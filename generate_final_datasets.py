import pandas as pd
import random
import json
import numpy as np
from typing import Dict, Optional, Set
from tqdm import tqdm
import logging
from functools import lru_cache
import gc

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
        chunk['REVENUE_AMT'] = pd.to_numeric(chunk['REVENUE_AMT'], errors='coerce').fillna(0)
        def assign_impact_score(revenue):
            if pd.isna(revenue) or revenue <= 0:
                return 'Low'
            elif revenue <= 100000:
                return 'Low'
            elif revenue <= 1000000:
                return 'Medium'
            else:
                return 'High'
        chunk['impact_score'] = chunk['REVENUE_AMT'].apply(assign_impact_score)
        chunk['mission_statement'] = chunk.apply(generate_mission_statement, axis=1)
        chunk['financial_metric'] = chunk['REVENUE_AMT'].clip(lower=1)
        chunk['impact_score_numeric'] = chunk['impact_score'].map({'Low': 1, 'Medium': 2, 'High': 3})
        chunk['impact_efficiency'] = chunk['impact_score_numeric'] / chunk['financial_metric']
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

def assess_data_quality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assess data quality by checking for missing critical fields and assign confidence scores.
    Critical fields are weighted as follows:
    - mission_statement: 40%, INCOME_AMT/REVENUE_AMT: 30%, impact_score: 20%, Other fields (NAME, EIN, NTEE_CD): 10%
    """
    quality_df = df.copy() # Initialize quality metrics
    if 'EIN' in quality_df.columns: # Ensure EIN remains as string
        quality_df['EIN'] = quality_df['EIN'].astype(str)
    numeric_cols = ['INCOME_AMT', 'REVENUE_AMT']
    for col in numeric_cols: # Ensure numeric columns are properly converted
        if col in quality_df.columns:
            quality_df[col] = pd.to_numeric(quality_df[col], errors='coerce')
    field_weights = { # Define critical fields and their weights
        'mission_statement': 0.4, 'financial': 0.3,  # Combined INCOME_AMT and REVENUE_AMT
        'impact_score': 0.2, 'basic': 0.1      # Combined NAME, EIN, NTEE_CD
    }
    # Check mission statement
    quality_df['has_mission'] = quality_df['mission_statement'].notna() & \
                               (quality_df['mission_statement'].str.len() > 10)
    # Check financial data
    quality_df['has_financial'] = quality_df[['INCOME_AMT', 'REVENUE_AMT']].notna().any(axis=1) & \
                                 (quality_df[['INCOME_AMT', 'REVENUE_AMT']].fillna(0) > 0).any(axis=1)
    quality_df['has_impact'] = quality_df['impact_score'].notna() # Check impact score
    # Check basic info
    quality_df['has_basic'] = quality_df[['NAME', 'EIN', 'NTEE_CD']].notna().all(axis=1)
    quality_df['confidence_score'] = ( # Calculate confidence score
        quality_df['has_mission'].astype(float) * field_weights['mission_statement'] +
        quality_df['has_financial'].astype(float) * field_weights['financial'] +
        quality_df['has_impact'].astype(float) * field_weights['impact_score'] +
        quality_df['has_basic'].astype(float) * field_weights['basic'])
    quality_df['data_quality'] = pd.cut( # Assign quality level
        quality_df['confidence_score'],
        bins=[-float('inf'), 0.3, 0.6, 0.8, float('inf')],
        labels=['poor', 'fair', 'good', 'excellent'])
    # List missing fields
    def get_missing_fields(row: pd.Series) -> Set[str]:
        missing = set()
        if not row['has_mission']:
            missing.add('mission_statement')
        if not row['has_financial']:
            missing.add('financial_data')
        if not row['has_impact']:
            missing.add('impact_score')
        if not row['has_basic']:
            missing.add('basic_info')
        return missing    
    quality_df['missing_fields'] = quality_df.apply(get_missing_fields, axis=1)
    # Return only necessary columns, ensuring EIN is string
    result_df = quality_df[['NAME', 'EIN', 'confidence_score', 'data_quality', 
                           'missing_fields', 'has_mission', 'has_financial',  'has_impact', 'has_basic']]
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
        # Handle revenue data - convert to numeric and handle missing/zero values
        df['REVENUE_AMT'] = pd.to_numeric(df['REVENUE_AMT'], errors='coerce').fillna(0)
        # Set minimum revenue to $1 to avoid division by zero
        df['financial_metric'] = df['REVENUE_AMT'].clip(lower=1)
        df['revenue_percentile'] = df['REVENUE_AMT'].rank(pct=True)
        # Calculate multiple anomaly indicators - 1. Impact Efficiency (impact score relative to revenue)
        df['impact_efficiency'] = df['impact_score_numeric'] / np.log10(df['financial_metric'] + 10)
        # 2. Revenue Discrepancy (unusually low/high revenue)
        revenue_mean = df['REVENUE_AMT'].mean()
        revenue_std = df['REVENUE_AMT'].std()
        df['revenue_zscore'] = (df['REVENUE_AMT'] - revenue_mean) / revenue_std
        # 3. Impact Score vs Revenue Mismatch
        df['expected_impact'] = pd.qcut(df['revenue_percentile'], q=3, 
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
                (df['revenue_zscore'].abs() > 2) | # Extreme revenue
                ((df['impact_score'] == 'High') & (df['revenue_percentile'] < 0.2)) | # High impact score with low revenue
                ((df['impact_score'] == 'Low') & (df['revenue_percentile'] > 0.8))  # Low impact score with high revenue
                )
            df['anomaly_score'] = ( # Calculate anomaly scores
                # Normalize and combine multiple factors
                0.4 * ((df['impact_efficiency'] - Q3) / IQR) +
                0.3 * df['revenue_zscore'].abs() +
                0.3 * df['impact_mismatch'].astype(int))
            df['risk_level'] = pd.cut( # Assign risk levels based on anomaly score
                df['anomaly_score'],
                bins=[-float('inf'), 0.5, 1.0, 1.5, float('inf')],
                labels=['Low', 'Medium', 'High', 'Critical'])
            df['anomaly_type'] = 'normal' # Add detailed flags
            df.loc[df['is_anomalous'] & (df['impact_efficiency'] > Q3 + 1.5 * IQR), 'anomaly_type'] = 'high_impact_low_finance'
            df.loc[df['is_anomalous'] & (df['revenue_zscore'].abs() > 2), 'anomaly_type'] = 'extreme_revenue'
            df.loc[df['is_anomalous'] & ((df['impact_score'] == 'High') & (df['revenue_percentile'] < 0.2)), 'anomaly_type'] = 'suspicious_high_impact'
            df.loc[df['is_anomalous'] & ((df['impact_score'] == 'Low') & (df['revenue_percentile'] > 0.8)), 'anomaly_type'] = 'suspicious_low_impact'
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

def process_data(df: pd.DataFrame, is_grant: bool = True) -> pd.DataFrame:
    """Process data in chunks without multiprocessing."""
    chunk_size = 1000
    processed_chunks = []
    for i in tqdm(range(0, len(df), chunk_size), desc="Processing " + ("grants" if is_grant else "nonprofits")):
        chunk = df[i:i + chunk_size]
        processed_chunk = process_chunk(chunk, is_grant)
        processed_chunks.append(processed_chunk)
        del chunk # Free memory
        gc.collect()
    result = pd.concat(processed_chunks, ignore_index=True)
    del processed_chunks
    gc.collect()
    return result

def main():
    """Main function to process datasets."""
    try:
        logger.info("Starting dataset generation...")
        dtypes = { # Define dtypes for efficient memory usage
            'NAME': 'string', 'EIN': 'string', 'NTEE_CD': 'string', 
            'CITY': 'string', 'STATE': 'string', 'ZIP': 'string',
            'REVENUE_AMT': 'float32', 'ASSET_AMT': 'float32', 'INCOME_AMT': 'float32'}
        logger.info("Processing grants data...") # Read and process grants
        grants_df = pd.read_csv('data/grants.csv', low_memory=False)
        grants_df = process_data(grants_df, is_grant=True)
        grants_df.to_csv('data/grants_final.csv', index=False)
        gc.collect() # Free memory
        logger.info("Processing nonprofits data...") # Read and process nonprofits
        nonprofits_df = pd.read_csv(
            'data/non-profits.csv', dtype=dtypes,
            na_values=['', 'nan', 'NaN', 'NULL'], low_memory=False)
        nonprofits_df = process_data(nonprofits_df, is_grant=False) # Process nonprofits data
        logger.info("Assessing nonprofit data quality...") # Assess data quality
        quality_df = assess_data_quality(nonprofits_df)
        quality_df.to_csv('data/nonprofit_quality.csv', index=False)
        quality_summary = quality_df['data_quality'].value_counts()
        logger.info("Data quality summary:")
        for quality_level, count in quality_summary.items():
            logger.info(f"{quality_level}: {count} nonprofits")
        logger.info("Analyzing impact score anomalies...") # Detect anomalous impact scores (use a lower confidence threshold)
        confidence_threshold = 0.5  # Lower threshold to include more data
        high_confidence_df = nonprofits_df[quality_df['confidence_score'] >= confidence_threshold].copy()
        anomalies_df = detect_anomalous_impact_scores(high_confidence_df)
        anomalies_df.to_csv('data/nonprofit_anomalies.csv', index=False)
        logger.info(f"Found {anomalies_df['is_anomalous'].sum()} anomalous nonprofits")
        logger.info("Generating funder history...") # Generate and save data
        try:
            funder_history = generate_funder_history(grants_df, nonprofits_df)
            if not funder_history:
                logger.warning("Generated funder history is empty!")
            logger.info("Saving funder history...")
            with open('data/funder_history.json', 'w') as f:
                json.dump(funder_history, f, indent=2)
            with open('data/funder_history.json', 'r') as f: # Verify the save
                saved_data = json.load(f)
                if not saved_data:
                    logger.warning("Saved funder history is empty!")
                else:
                    logger.info(f"Successfully saved funder history with {len(saved_data)} agencies")
        except Exception as e:
            logger.error(f"Error generating/saving funder history: {str(e)}", exc_info=True)
            if funder_history: # Create a backup of any generated data
                with open('data/funder_history.backup.json', 'w') as f:
                    json.dump(funder_history, f, indent=2)
        logger.info("Saving processed datasets...")
        nonprofits_df.to_csv('data/non-profits_final.csv', index=False)
        del grants_df, nonprofits_df, quality_df, anomalies_df, funder_history # Free memory
        gc.collect()
        logger.info("Dataset generation completed successfully!")
    except Exception as e:
        logger.error(f"Error during dataset generation: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main() 