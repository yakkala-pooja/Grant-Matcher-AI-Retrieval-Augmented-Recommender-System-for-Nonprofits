import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
import re
import os
from collections import defaultdict
import json

def clean_text(text: str) -> str:
    """Clean and preprocess text data."""
    if pd.isna(text):
        return ""
    if not isinstance(text, str):
        return str(text)
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', ' ', text.lower())).strip()

def convert_currency_to_float(value) -> float:
    """Convert currency string to float."""
    if isinstance(value, pd.Series):
        return value.apply(convert_currency_to_float)
    if pd.isna(value) or not value:
        return np.nan
    try:
        if isinstance(value, str):
            value = value.replace('$', '').replace(',', '')
        return float(value)
    except (ValueError, TypeError):
        return np.nan

def convert_date(date_str) -> pd.Timestamp:
    """Convert date string to pandas Timestamp."""
    if pd.isna(date_str):
        return pd.NaT
    date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y']
    try:
        for fmt in date_formats: # Try specific formats first
            try:
                return pd.to_datetime(date_str, format=fmt)
            except ValueError:
                continue
        return pd.to_datetime(date_str) # Fall back to pandas automatic parsing
    except:
        return pd.NaT

def load_funder_history(file_path: str) -> pd.DataFrame:
    """Load and preprocess funder history data."""
    with open(file_path, 'r') as f:
        data = json.load(f) # Read JSON file
    records = [] # Convert JSON to DataFrame format
    for funder_name, funding_records in data.items():
        for record in funding_records:
            record['funder_name'] = funder_name
            records.append(record)
    df = pd.DataFrame(records)
    str_columns = ['funder_name', 'recipient_name', 'purpose']
    for col in str_columns: # Clean string columns
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    amount_cols = [col for col in df.columns if 'amount' in col.lower()]
    for col in amount_cols: # Convert amount columns
        df[col] = df[col].apply(convert_currency_to_float)
    date_cols = [col for col in df.columns if 'date' in col.lower()]
    for col in date_cols: # Convert date columns
        df[col] = df[col].apply(convert_date)
    return df

def create_funder_nonprofit_mapping(funder_history_df: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Create a mapping of funders to their funded nonprofits with funding details."""
    funder_mapping = defaultdict(list)
    for _, row in funder_history_df.iterrows(): # Process each funding record
        funding_details = {
            'recipient_name': row['recipient_name'],
            'amount': row.get('amount', np.nan), 'date': row.get('date', pd.NaT),
            'purpose': row.get('purpose', ''), 'program_area': row.get('program_area', '')}
        # Add any additional fields
        extra_fields = set(row.index) - {'funder_name', 'recipient_name', 'amount', 'date', 'purpose', 'program_area'}
        funding_details.update({field: row[field] for field in extra_fields})
        funder_mapping[row['funder_name'].lower()].append(funding_details)
    return dict(funder_mapping)

def save_funder_nonprofit_mapping(mapping: Dict[str, List[Dict]], output_dir: str):
    """Save funder-nonprofit mapping to JSON files in a directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    summary_data = []
    for funder_name, funded_orgs in mapping.items(): # Process each funder's data
        safe_name = re.sub(r'[^\w\s-]', '', funder_name).strip().replace(' ', '_')
        if not safe_name:
            continue
        # Calculate summary statistics
        total_funding = sum(org['amount'] for org in funded_orgs if not pd.isna(org.get('amount', np.nan)))
        funder_data = { # Prepare funder data
            'funder_name': funder_name, 'total_funding': total_funding,
            'number_of_recipients': len(funded_orgs), 'funded_organizations': funded_orgs}
        summary_data.append({ # Add to summary
            'funder_name': funder_name, 'total_funding': total_funding,
            'number_of_recipients': len(funded_orgs), 'filename': f"{safe_name}.json"})
        # Save individual funder file
        with open(os.path.join(output_dir, f"{safe_name}.json"), 'w', encoding='utf-8') as f:
            json.dump(funder_data, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'funding_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, default=str) # Save summary file

def load_grants_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess grants data from CSV file."""
    df = pd.read_csv(file_path)
    # Validate required columns
    required_cols = ['opportunity_number', 'opportunity_title', 'grant_description', 'agency_name']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    text_cols = [ # Cleaning text columns
        'grant_description', 'opportunity_title', 'agency_name',
        'opportunity_category', 'funding_instrument_type', 'category']
    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    currency_cols = ['award_ceiling', 'award_floor', 'estimated_total_program_funding']
    for col in currency_cols: # Convert numeric columns
        if col in df.columns:
            df[col] = df[col].apply(convert_currency_to_float)
    if 'expected_number_of_awards' in df.columns:
        df['expected_number_of_awards'] = pd.to_numeric(df['expected_number_of_awards'], errors='coerce')
    date_cols = ['post_date', 'close_date', 'last_updated_date', 'archive_date']
    for col in date_cols: # Convert date columns
        if col in df.columns:
            df[col] = df[col].apply(convert_date)
    # Handle boolean columns
    if 'cost_sharing_or_matching_requirement' in df.columns:
        df['cost_sharing_or_matching_requirement'] = df['cost_sharing_or_matching_requirement'].map({
            'yes': True, 'no': False, 'Yes': True, 'No': False, True: True, False: False
        }).fillna(False)
    df = df.dropna(subset=['grant_description']) # Clean and validate data
    df = df[df['grant_description'].str.strip() != '']
    default_values = { # Fill missing values
        'agency_name': 'Unknown Agency', 'opportunity_title': 'Untitled Opportunity',
        'opportunity_category': 'Uncategorized',
        'funding_instrument_type': 'Not Specified'}
    df = df.fillna(default_values)
    if 'opportunity_id' in df.columns: # Ensure valid opportunity numbers
        df['opportunity_number'] = df['opportunity_number'].fillna(
            df['opportunity_id'].astype(str).apply(lambda x: f'GRANT-{x}'))
    return df

def load_nonprofits_data(file_path: str, funder_history_path: str = None) -> Tuple[pd.DataFrame, Dict[str, List[Dict]]]:
    """Load and preprocess nonprofits data and optionally include funder history."""
    df = pd.read_csv(file_path)
    required_cols = ['NAME', 'mission_statement'] # Validate required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    text_cols = ['NAME', 'mission_statement', 'STREET', 'CITY', 'STATE']
    for col in text_cols: # Clean text columns
        if col in df.columns:
            df[col] = df[col].apply(clean_text)
    
    # Handle revenue and financial data first
    financial_cols = ['ASSET_AMT', 'INCOME_AMT', 'REVENUE_AMT']
    for col in financial_cols:
        if col in df.columns:
            # Convert any currency strings to numeric
            df[col] = df[col].apply(convert_currency_to_float)
            # Ensure non-negative values
            df.loc[df[col] < 0, col] = np.nan
    
    # Other numeric columns
    other_numeric_cols = ['impact_score', 'anomaly_score']
    for col in other_numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    categorical_cols = ['risk_level', 'anomaly_type', 'impact_score']
    for col in categorical_cols: # Handle categorical columns
        if col in df.columns:
            df[col] = df[col].fillna('Low')  # Default to Low for missing values
    
    if 'is_anomalous' in df.columns: # Handle boolean columns
        df['is_anomalous'] = df['is_anomalous'].fillna(False)
    
    if 'ZIP' in df.columns:  # Handle ZIP codes
        df['ZIP'] = df['ZIP'].astype(str).str.zfill(5)
    
    df = df.dropna(subset=['mission_statement']) # Clean and validate data
    df = df[df['mission_statement'].str.strip() != '']
    
    default_values = { # Fill missing values
        'NAME': 'Unknown Organization', 'EIN': 'NOT_PROVIDED',
        'STATE': 'NA', 'risk_level': 'Low',
        'anomaly_type': 'normal', 'is_anomalous': False, 'anomaly_score': 0.0
    }
    df = df.fillna(default_values)
    
    funder_mapping = None # Process funder history if provided
    if funder_history_path and os.path.exists(funder_history_path):
        funder_history_df = load_funder_history(funder_history_path)
        funder_mapping = create_funder_nonprofit_mapping(funder_history_df)
        save_funder_nonprofit_mapping(funder_mapping, 'funder_history')
    
    return df, funder_mapping

def prepare_data_for_embedding(df: pd.DataFrame) -> Tuple[list, list]:
    """Prepare grant descriptions for embedding."""
    return [f"{row['opportunity_title']} {row['grant_description']}" for _, row in df.iterrows()], \
           df['opportunity_number'].tolist() 