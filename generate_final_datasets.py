import pandas as pd
import random
import json
import numpy as np
from typing import Dict, Optional, Set
from tqdm import tqdm
import logging
from functools import lru_cache
import gc
import os
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
with open('templates.json', 'r') as f:
    TEMPLATES = json.load(f) # Load templates

@lru_cache(maxsize=1000)
def clean_text(text: Optional[str], default: str = "") -> str:
    """Clean and format text with caching."""
    return default if pd.isna(text) else text.lower().strip('"').strip()

@lru_cache(maxsize=100)
def map_category(category: Optional[str]) -> str:
    """Map funding activity category to simplified category."""
    if pd.isna(category):
        return 'Other'
    category = category.lower()
    for key in TEMPLATES['purpose'].keys():
        if key.lower() in category:
            return key
        return 'Other'

def generate_grant_description(row: pd.Series) -> str:
    """Generate a descriptive paragraph for a grant opportunity."""
    agency = clean_text(row['agency_name'], "sponsoring agency")
    title = clean_text(row['opportunity_title'], "various programs and initiatives")
    category = map_category(row['category_of_funding_activity'])
    eligible = clean_text(row['eligible_applicants_type'], "qualified organizations")
    return f"{random.choice(TEMPLATES['intro']).format(agency=agency)} \
            {random.choice(TEMPLATES['purpose'][category]).format(title=title)}. \
            {random.choice(TEMPLATES['eligibility']).format(eligible_types=eligible)}"

def generate_mission_statement(row: pd.Series) -> str:
    """Generate a mission statement based on NTEE code."""
    category = ('OTHER' if pd.isna(row['NTEE_CD']) else row['NTEE_CD'][0].upper())
    if category not in TEMPLATES['ntee_missions']:
        category = 'OTHER'
    city = clean_text(row['CITY'], 'our area')
    state = clean_text(row['STATE'], 'our region')
    return random.choice(TEMPLATES['ntee_missions'][category]['templates']).format(
        city=city, state=state)

def process_chunk(chunk: pd.DataFrame, is_grant: bool = True) -> pd.DataFrame:
    """Process a chunk of data."""
    if is_grant:
        chunk['category'] = chunk['category_of_funding_activity'].apply(map_category)
        chunk['grant_description'] = chunk.apply(generate_grant_description, axis=1)
    else:
        chunk = chunk.copy()
        
        # Print debug info about income values
        print("\nDebug - Income values before processing:")
        print(f"Number of records: {len(chunk)}")
        print(f"Number of non-null INCOME_AMT values: {chunk['INCOME_AMT'].notna().sum()}")
        print(f"Sample of INCOME_AMT values:\n{chunk['INCOME_AMT'].head()}")
        
        # Convert income to numeric but don't fill missing values
        chunk['INCOME_AMT'] = pd.to_numeric(chunk['INCOME_AMT'], errors='coerce')
        
        # Print debug info after conversion
        print("\nDebug - Income values after numeric conversion:")
        print(f"Number of non-null INCOME_AMT values: {chunk['INCOME_AMT'].notna().sum()}")
        print(f"Sample of converted values:\n{chunk['INCOME_AMT'].head()}")
        print(f"Value counts of impact scores:\n{chunk['INCOME_AMT'].value_counts().head()}")
        
        def assign_impact_score(income):
            if pd.isna(income):
                return None  # Return None for missing income
            elif income <= 0:
                return None  # Return None for zero or negative income
            elif income <= 100000:  # $100k threshold for Low
                return 'Low'
            elif income <= 1000000:  # $1M threshold for Medium
                return 'Medium'
            else:
                return 'High'
        
        # Assign impact scores based on income
        chunk['impact_score'] = chunk['INCOME_AMT'].apply(assign_impact_score)
        
        # Print impact score distribution
        print("\nDebug - Impact score distribution:")
        print(chunk['impact_score'].value_counts())
        
        # Create numeric impact score
        impact_score_map = {'Low': 1.0, 'Medium': 2.0, 'High': 3.0}
        chunk['impact_score_numeric'] = chunk['impact_score'].map(impact_score_map)
        
        # Generate mission statement
        chunk['mission_statement'] = chunk.apply(generate_mission_statement, axis=1)
        
        # Set financial metric (use actual income)
        chunk['financial_metric'] = chunk['INCOME_AMT']
        
        # Calculate efficiency only for valid scores and income
        valid_mask = (
            chunk['impact_score_numeric'].notna() & 
            chunk['financial_metric'].notna() & 
            (chunk['financial_metric'] > 0)
        )
        chunk.loc[valid_mask, 'impact_efficiency'] = (
            chunk.loc[valid_mask, 'impact_score_numeric'] / 
            np.log10(chunk.loc[valid_mask, 'financial_metric'] + 10)
        )
    
    return chunk

def generate_funder_history(grants_df: pd.DataFrame, nonprofits_df: pd.DataFrame) -> Dict:
    """Generate synthetic funder history."""
    funder_history = {}
    logger.info("Starting funder history generation...")
    category_to_ntee = {
        'Science': ['A', 'B', 'D', 'H', 'U', 'V', 'W'],
        'Other': ['C', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'X', 'Y', 'Z']
    }
    try:
        unique_agencies = grants_df['agency_name'].dropna().unique()
        
        for agency in tqdm(unique_agencies, desc="Generating funder history"):
            try:
                agency_cats = set(grants_df[grants_df['agency_name'] == agency]['category'].unique())
                valid_ntee_prefixes = []
                for cat in agency_cats:
                    if cat in category_to_ntee:
                        valid_ntee_prefixes.extend(category_to_ntee[cat])
                matching_orgs = nonprofits_df[
                    (nonprofits_df['NTEE_CD'].notna()) &
                    (nonprofits_df['NTEE_CD'].str[0].isin(valid_ntee_prefixes))]
                if len(matching_orgs) > 0:
                    num_funded = min(random.randint(5, 15), len(matching_orgs))
                    funded_orgs = matching_orgs.sample(n=num_funded)
                    funder_history[agency] = [
                        {
                            'recipient_name': org['NAME'],
                            'amount': round(random.uniform(50000, 500000), 2),
                            'date': pd.Timestamp(
                                year=random.randint(2018, 2023),
                                month=random.randint(1, 12), day=random.randint(1, 28)).strftime('%Y-%m-%d'),
                            'purpose': random.choice(TEMPLATES['purpose'][random.choice(list(agency_cats))])
                                .format(title=org['NAME'].lower())}
                        for _, org in funded_orgs.iterrows()]
                if len(funder_history) % 100 == 0:
                    gc.collect()
            except Exception as e:
                logger.error(f"Error processing agency {agency}: {str(e)}")
                continue
        logger.info(f"Funder history generation completed. Total agencies processed: {len(funder_history)}")
        return funder_history
    except Exception as e:
        logger.error(f"Error in generate_funder_history: {str(e)}")
        return {}

def assess_data_quality(df):
    """Assess data quality for each nonprofit."""
    quality_df = df.copy()
    
    # Check for required fields
    quality_df['has_name'] = quality_df['NAME'].notna()
    quality_df['has_mission'] = quality_df['mission_statement'].notna() & (quality_df['mission_statement'].str.len() > 10)
    quality_df['has_ein'] = quality_df['EIN'].notna()
    quality_df['has_ntee'] = quality_df['NTEE_CD'].notna()
    
    # Check financial data - consider an organization to have financial data if it has a valid income amount
    quality_df['has_financial'] = quality_df['INCOME_AMT'].notna() & (quality_df['INCOME_AMT'] > 0)
    
    # Calculate quality score (weighted average)
    # Weights:
    # - mission_statement: 40%, INCOME_AMT: 30%, impact_score: 20%, Other fields (NAME, EIN, NTEE_CD): 10%
    quality_df['quality_score'] = (
        0.4 * quality_df['has_mission'].astype(float) +
        0.3 * quality_df['has_financial'].astype(float) +
        0.2 * quality_df['impact_score'].notna().astype(float) +
        0.1 * (quality_df['has_name'] & quality_df['has_ein'] & quality_df['has_ntee']).astype(float)
    )
    
    # Assign quality levels
    quality_df['data_quality'] = pd.cut(
        quality_df['quality_score'],
        bins=[-float('inf'), 0.3, 0.5, 0.7, float('inf')],
        labels=['poor', 'fair', 'good', 'excellent']
    )
    
    # Generate missing fields list
    def get_missing_fields(row):
        missing = []
        if not row['has_mission']:
            missing.append('mission_statement')
        if not row['has_financial']:
            missing.append('financial_data')
        if not row['impact_score']:
            missing.append('impact_score')
        if not (row['has_name'] and row['has_ein'] and row['has_ntee']):
            missing.append('basic_info')
        return ', '.join(missing) if missing else 'None'
    
    quality_df['missing_fields'] = quality_df.apply(get_missing_fields, axis=1)
    
    # Keep only necessary columns
    result_df = quality_df[['EIN', 'NAME', 'data_quality', 'quality_score', 'missing_fields']]
    result_df['EIN'] = result_df['EIN'].astype(str)
    
    return result_df

def detect_anomalous_impact_scores(df: pd.DataFrame, iqr_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Flag nonprofits with unusually high impact scores relative to their financial metrics.
    Uses multiple criteria for anomaly detection.
    """
    try:
        df = df.copy() # Create a copy to avoid SettingWithCopyWarning
        if 'EIN' in df.columns: # Ensure EIN is string
            df['EIN'] = df['EIN'].astype(str)
        impact_score_map = {'Low': 1, 'Medium': 2, 'High': 3} # Convert impact score categories to numeric values and handle missing values
        df['impact_score'] = df['impact_score'].fillna('Medium')  # Default missing scores to Medium instead of Low
        df['impact_score_numeric'] = df['impact_score'].map(impact_score_map).fillna(2)  # Default to 2 if mapping fails
        # Handle income data - convert to numeric and handle missing/zero values
        df['INCOME_AMT'] = pd.to_numeric(df['INCOME_AMT'], errors='coerce').fillna(0)
        # Set minimum income to $1 to avoid division by zero
        df['financial_metric'] = df['INCOME_AMT'].clip(lower=1)
        df['income_percentile'] = df['INCOME_AMT'].rank(pct=True)
        # Calculate multiple anomaly indicators - 1. Impact Efficiency (impact score relative to income)
        df['impact_efficiency'] = df['impact_score_numeric'] / np.log10(df['financial_metric'] + 10)
        # 2. Income Discrepancy (unusually low/high income)
        income_mean = df['INCOME_AMT'].mean()
        income_std = df['INCOME_AMT'].std()
        df['income_zscore'] = (df['INCOME_AMT'] - income_mean) / income_std
        # 3. Impact Score vs Income Mismatch
        df['expected_impact'] = pd.qcut(df['income_percentile'], q=3, 
                                      labels=['Low', 'Medium', 'High'])
        df['impact_mismatch'] = df['impact_score'] != df['expected_impact']
        # Remove any remaining infinite or NaN values
        df['impact_efficiency'] = df['impact_efficiency'].replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=['impact_efficiency'])
        valid_efficiencies = df['impact_efficiency'].dropna() # Calculate IQR statistics for impact efficiency
        if len(valid_efficiencies) > 0:
            Q1 = valid_efficiencies.quantile(0.25)
            Q3 = valid_efficiencies.quantile(0.75)
            IQR = Q3 - Q1
            efficiency_threshold = Q3 + iqr_multiplier * IQR
            df['is_anomalous'] = (  # Flag anomalies based on multiple criteria with more sensitive thresholds
                # High impact efficiency
                (df['impact_efficiency'] > Q3 + 1.5 * IQR) |  # Changed from efficiency_threshold
                (df['income_zscore'].abs() > 2) | # Extreme income
                ((df['impact_score'] == 'High') & (df['income_percentile'] < 0.2)) | # High impact score with low income
                ((df['impact_score'] == 'Low') & (df['income_percentile'] > 0.8))  # Low impact score with high income
                )
            df['anomaly_score'] = ( # Calculate anomaly scores
                # Normalize and combine multiple factors
                0.4 * ((df['impact_efficiency'] - Q3) / IQR) +
                0.3 * df['income_zscore'].abs() +
                0.3 * df['impact_mismatch'].astype(int))
            df['risk_level'] = pd.cut( # Assign risk levels based on anomaly score
                df['anomaly_score'],
                bins=[-float('inf'), 0.5, 1.0, 1.5, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical'])
            df['anomaly_type'] = 'normal' # Add detailed flags
            df.loc[df['is_anomalous'] & (df['impact_efficiency'] > Q3 + 1.5 * IQR), 'anomaly_type'] = 'high_impact_low_finance'
            df.loc[df['is_anomalous'] & (df['income_zscore'].abs() > 2), 'anomaly_type'] = 'extreme_income'
            df.loc[df['is_anomalous'] & ((df['impact_score'] == 'High') & (df['income_percentile'] < 0.2)), 'anomaly_type'] = 'suspicious_high_impact'
            df.loc[df['is_anomalous'] & ((df['impact_score'] == 'Low') & (df['income_percentile'] > 0.8)), 'anomaly_type'] = 'suspicious_low_impact'
        else:
            print("\nWarning: No valid efficiency values found for IQR analysis")
            df['is_anomalous'] = False
            df['anomaly_score'] = 0
            df['anomaly_type'] = 'unknown'
            df['risk_level'] = 'Low'
        # Return result with string EIN
        result_df = df[['NAME', 'EIN', 'impact_score', 'financial_metric', 
                       'impact_efficiency', 'is_anomalous', 'anomaly_score', 'anomaly_type', 'risk_level']]
        result_df['EIN'] = result_df['EIN'].astype(str)
        return result_df
    except Exception as e:
        logger.error(f"Error in anomaly detection: {str(e)}")
        print(f"Exception details: {str(e)}")
        raise

def process_data():
    """Process and save the final datasets."""
    logger.info("Loading and processing data...")
    
    # Load and process nonprofits data
    nonprofits_df = pd.read_csv('data/non-profits.csv')
    nonprofits_df = process_chunk(nonprofits_df, is_grant=False)
    nonprofits_df.to_csv('data/non-profits_final.csv', index=False)
    logger.info("Saved processed nonprofits data")
    
    # Load and process grants data
    grants_df = pd.read_csv('data/grants.csv')
    grants_df = process_chunk(grants_df, is_grant=True)
    grants_df.to_csv('data/grants_final.csv', index=False)
    logger.info("Saved processed grants data")
    
    # Generate quality data
    quality_df = assess_data_quality(nonprofits_df)
    quality_df.to_csv('data/nonprofit_quality.csv', index=False)
    logger.info("Saved nonprofit quality data")
    
    # Generate anomaly data
    anomalies_df = detect_anomalous_impact_scores(nonprofits_df)
    anomalies_df.to_csv('data/nonprofit_anomalies.csv', index=False)
    logger.info("Saved nonprofit anomalies data")

if __name__ == "__main__":
    process_data() 