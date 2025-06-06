import logging
from grant_recommender import GrantRecommender
from typing import List, Dict
import pandas as pd
from data_loader import load_nonprofits_data

# Configure logging
logging.basicConfig(level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def format_currency(value) -> str:
    """Format currency values with commas and dollar sign."""
    return "Not specified" if pd.isna(value) else f"${value:,.2f}"

def format_date(date_value) -> str:
    """Format datetime to string."""
    return "Not specified" if pd.isna(date_value) else date_value.strftime("%Y-%m-%d")

def format_recommendations(recommendations: List[Dict]) -> None:
    """Print recommendations in a readable format."""
    logger.info("\nTop matching grants:")
    logger.info("-" * 100)
    for i, rec in enumerate(recommendations, 1):
        grant_info = [ # Basic grant information
            f"\n{i}. {rec['title']}",
            f"   Opportunity Number: {rec['opportunity_number']}",
            f"   Agency: {rec['agency']}",
            f"   Award Ceiling: {format_currency(rec.get('award_ceiling'))}",
            f"   Award Floor: {format_currency(rec.get('award_floor'))}"]
        # Date information
        if 'post_date' in rec:
            grant_info.append(f"   Posted Date: {format_date(rec.get('post_date'))}")
        if 'close_date' in rec:
            grant_info.append(f"   Close Date: {format_date(rec.get('close_date'))}")
        grant_info.extend([ # Additional details
            f"   Funding Type: {rec.get('funding_instrument_type', 'Not specified')}",
            f"   Category: {rec.get('category', 'Not specified')}",
            f"   Match Score: {rec['similarity_score']:.4f}"])
        # Display ranking scores if available
        if 'original_similarity' in rec and 'funder_boost' in rec:
            grant_info.append(f"   Base Score: {rec['original_similarity']:.4f}")
            if rec['funder_boost'] > 0:
                grant_info.append(f"   Funder History Boost: +{rec['funder_boost']*100:.1f}%")
        grant_info.append(f"   Description: {rec['description'][:200]}...")
        logger.info('\n'.join(grant_info)) # Log grant information
        # Display match justification
        if 'match_justification' in rec:
            just = rec['match_justification']
            logger.info("\n   Match Analysis:")
            logger.info("   " + "-" * 20)
            # Display alignment summaries
            if just['alignment_summary']:
                for summary in just['alignment_summary']:
                    logger.info(f"   • {summary}")
            # Display funding alignment
            if just['funding_alignment']:
                logger.info("\n   Funding Alignment:")
                for alignment in just['funding_alignment']:
                    logger.info(f"   • {alignment}")
            # Display shared keywords
            if just['shared_keywords']:
                logger.info("\n   Key Matching Themes:")
                logger.info(f"   • {', '.join(just['shared_keywords'])}")
        logger.info("-" * 100)

def format_nonprofit_info(nonprofit: pd.Series, funder_mapping: Dict = None) -> None:
    """Print nonprofit information in a readable format."""
    info_lines = [ "\nNonprofit Organization Details:",
        "-" * 100, f"Name: {nonprofit['NAME']}",
        f"EIN: {nonprofit['EIN']}"]
    # Location information
    if 'CITY' in nonprofit and 'STATE' in nonprofit:
        location = f"{nonprofit['CITY']}, {nonprofit['STATE']}"
        if 'ZIP' in nonprofit:
            location += f" {nonprofit['ZIP']}"
        info_lines.append(f"Location: {location}")
    # Risk and anomaly information
    if 'risk_level' in nonprofit:
        info_lines.append(f"Risk Level: {nonprofit['risk_level']}")
    if 'anomaly_type' in nonprofit:
        info_lines.append(f"Anomaly Type: {nonprofit['anomaly_type']}")
    if 'anomaly_score' in nonprofit and not pd.isna(nonprofit['anomaly_score']):
        info_lines.append(f"Anomaly Score: {nonprofit['anomaly_score']:.2f}")
    # Financial information
    if 'impact_score' in nonprofit and not pd.isna(nonprofit['impact_score']):
        info_lines.append(f"Impact Score: {nonprofit['impact_score']}")
    if 'ASSET_AMT' in nonprofit and not pd.isna(nonprofit['ASSET_AMT']):
        info_lines.append(f"Assets: {format_currency(nonprofit['ASSET_AMT'])}")
    if 'REVENUE_AMT' in nonprofit and not pd.isna(nonprofit['REVENUE_AMT']):
        info_lines.append(f"Revenue: {format_currency(nonprofit['REVENUE_AMT'])}")
    info_lines.append(f"\nMission Statement: {nonprofit['mission_statement']}")

    logger.info('\n'.join(info_lines)) # Log basic information
    if funder_mapping: # Display funding history if available
        nonprofit_name = nonprofit['NAME'].lower()
        received_funding = [] # Collect funding history
        for funder, funded_orgs in funder_mapping.items():
            for org in funded_orgs:
                if org['recipient_name'].lower() == nonprofit_name:
                    received_funding.append({
                        'funder': funder, 'amount': org['amount'],
                        'date': org['date'], 'purpose': org.get('purpose', ''),
                        'program_area': org.get('program_area', '')
                    })
        if received_funding: # Display funding history
            logger.info("\nFunding History:")
            logger.info("-" * 80)
            for funding in received_funding:
                funding_details = [
                    f"\nFunder: {funding['funder'].title()}",
                    f"Amount: {format_currency(funding['amount'])}",
                    f"Date: {format_date(funding['date'])}"]
                if funding['purpose']:
                    funding_details.append(f"Purpose: {funding['purpose']}")
                if funding['program_area']:
                    funding_details.append(f"Program Area: {funding['program_area']}")
                logger.info('\n'.join(funding_details))
    logger.info("-" * 100)

def main():
    logger.info("Loading and fitting the model...") # Initialize and load data
    recommender = GrantRecommender()
    recommender.fit('data/grants_final.csv')
    logger.info("Loading nonprofits and funder history data...")
    nonprofits_df, funder_mapping = load_nonprofits_data(
        'data/non-profits_final.csv',
        'data/funder_history.json')
    logger.info("Loading anomaly data...") # Load and merge anomaly data
    try:
        anomalies_df = pd.read_csv('data/nonprofit_anomalies.csv')
        nonprofits_df = pd.merge( # Merge anomaly data with nonprofits data
            nonprofits_df,
            anomalies_df[['EIN', 'risk_level', 'anomaly_type', 'anomaly_score', 'is_anomalous']],
            on='EIN', how='left'
        )
        # Fill missing values
        nonprofits_df['risk_level'] = nonprofits_df['risk_level'].fillna('Low')
        nonprofits_df['anomaly_type'] = nonprofits_df['anomaly_type'].fillna('normal')
        nonprofits_df['anomaly_score'] = nonprofits_df['anomaly_score'].fillna(0)
        nonprofits_df['is_anomalous'] = nonprofits_df['is_anomalous'].fillna(False)
        # Get a sample of nonprofits with different risk levels
        sample_size = 2  # Number of nonprofits to show per risk level
        samples = []
        for risk_level in ['High', 'Medium', 'Critical', 'Low']:
            risk_sample = nonprofits_df[nonprofits_df['risk_level'] == risk_level].sample(min(sample_size, len(nonprofits_df[nonprofits_df['risk_level'] == risk_level])))
            samples.append(risk_sample)
        sample_df = pd.concat(samples)
        for idx, nonprofit in sample_df.iterrows(): # Process sample nonprofits
            format_nonprofit_info(nonprofit, funder_mapping) # Display nonprofit information
            nonprofit_info = { # Prepare nonprofit info for recommendations
                'name': nonprofit['NAME'], 'mission_statement': nonprofit['mission_statement'],
                'ntee_code': nonprofit.get('NTEE_CD', ''), 'ein': nonprofit.get('EIN', ''),
                'state': nonprofit.get('STATE', ''), 'city': nonprofit.get('CITY', ''),
                'revenue': nonprofit.get('REVENUE_AMT'), 'assets': nonprofit.get('ASSET_AMT'),
                'risk_level': nonprofit.get('risk_level', 'Low'), 'anomaly_type': nonprofit.get('anomaly_type', 'normal'),
                'anomaly_score': nonprofit.get('anomaly_score', 0)}
            recommendations = recommender.get_recommendations( # Get and display recommendations
                mission_statement=nonprofit['mission_statement'],
                nonprofit_info=nonprofit_info,
                top_n=3, min_similarity=0.3)
            format_recommendations(recommendations)
            logger.info("\n" + "=" * 100 + "\n")
    except Exception as e:
        logger.warning(f"Could not load anomaly data: {str(e)}")
    logger.info("\nSaving the model...")
    recommender.save_model('model_output')
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    main() 