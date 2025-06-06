import streamlit as st
import pandas as pd
import numpy as np
from grant_recommender import GrantRecommender
import plotly.express as px
import plotly.graph_objects as go
import logging

st.set_page_config(
    page_title="GrantMatch AI", page_icon="🎯", layout="wide")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if 'model_weights' not in st.session_state: # Initialize session state
    logger.info("Initializing model weights in session state")
    st.session_state.model_weights = {
        'similarity_weight': 0.6, 'impact_weight': 0.2, 
        'financial_weight': 0.2}
if 'recommender' not in st.session_state:
    logger.info("Initializing recommender in session state")
    st.session_state.recommender = None

@st.cache_resource
def load_recommender():
    """Load the recommender model with caching."""
    logger.info("Attempting to load recommender model...")
    try:
        recommender = GrantRecommender.load_model('model_output')
        logger.info("Successfully loaded recommender model")
        return recommender
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}", exc_info=True)
        st.error(f"Error loading model: {str(e)}")
        st.error("Please ensure model files exist in 'model_output' directory.")
        return None
    
@st.cache_data
def load_data():
    """Load all necessary datasets."""
    try:
        logger.info("Starting data loading process...")
        dtypes = { # Define dtypes for efficient memory usage
            'NAME': 'string', 'EIN': 'string',
            'NTEE_CD': 'string', 'CITY': 'string',
            'STATE': 'string', 'ZIP': 'string',
            'REVENUE_AMT': 'float32', 'ASSET_AMT': 'float32',
            'INCOME_AMT': 'float32', 'impact_score': 'string'  # Changed from float32 to string
        }
        logger.info("Loading nonprofits data...")
        usecols = list(dtypes.keys()) + ['mission_statement'] # Only load necessary columns
        # Load nonprofits data with optimized settings
        nonprofits = pd.read_csv(
            'data/non-profits_final.csv',
            dtype=dtypes, usecols=usecols,
            na_values=['', 'nan', 'NaN', 'NULL'], low_memory=False)
        logger.info(f"Loaded {len(nonprofits)} nonprofits")
        # Convert impact_score to numeric if it's a number
        nonprofits['impact_score_numeric'] = pd.to_numeric(nonprofits['impact_score'], errors='coerce')
        logger.info("Loading quality data...")
        try: # Try to load quality data, generate if not exists
            quality = pd.read_csv(
                'data/nonprofit_quality.csv', dtype={'EIN': 'string'}  # Ensure EIN is loaded as string
            )
            logger.info(f"Loaded quality data with {len(quality)} records")
        except FileNotFoundError:
            logger.warning("Quality data not found, generating...")
            st.warning("Generating nonprofit quality data...")
            from generate_final_datasets import assess_data_quality
            quality = assess_data_quality(nonprofits)
            quality.to_csv('data/nonprofit_quality.csv', index=False)
        logger.info("Loading anomalies data...")
        try: # Try to load anomalies data, generate if not exists
            anomalies = pd.read_csv(
                'data/nonprofit_anomalies.csv', dtype={'EIN': 'string'}  # Ensure EIN is loaded as string
            )
            logger.info(f"Loaded anomalies data with {len(anomalies)} records")
        except FileNotFoundError:
            logger.warning("Anomalies data not found, generating...")
            st.warning("Generating anomaly detection data...")
            from generate_final_datasets import detect_anomalous_impact_scores
            high_quality = nonprofits[nonprofits['EIN'].isin(
                quality[quality['data_quality'].isin(['good', 'excellent'])]['EIN'])]
            anomalies = detect_anomalous_impact_scores(high_quality)
            anomalies.to_csv('data/nonprofit_anomalies.csv', index=False)
        logger.info("Data loading completed successfully")
        return nonprofits, quality, anomalies
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}", exc_info=True)
        st.error(f"Error loading data: {str(e)}")
        st.error("Please run generate_final_datasets.py first to prepare the required data files.")
        raise
if 'data_loaded' not in st.session_state: # Initialize session state for data
    logger.info("Loading data into session state...")
    st.session_state.nonprofits, st.session_state.quality, st.session_state.anomalies = load_data()
    st.session_state.data_loaded = True # Initialize session state for data

def display_grant_matches():
    """Display the grant matching interface."""
    st.header("Grant Matching")
    if st.session_state.recommender is None: # Load recommender model
        logger.info("Loading recommender model...")
        st.session_state.recommender = load_recommender()
    if st.session_state.recommender is None:
        logger.error("Failed to load recommender model")
        return
    logger.info("Using data from session state...") # Use data from session state
    nonprofits = st.session_state.nonprofits
    quality = st.session_state.quality
    st.subheader("Select Nonprofit") # Nonprofit selection
    acceptable_quality = nonprofits[nonprofits['EIN'].isin(
        quality[quality['data_quality'].isin(['good', 'fair'])]['EIN'])]
    logger.info(f"Found {len(acceptable_quality)} nonprofits with acceptable quality")
    selected_nonprofit = st.selectbox(
        "Choose a nonprofit organization:",
        options=acceptable_quality['NAME'].unique(),
        help="Showing nonprofits with good or fair data quality")
    if selected_nonprofit:
        nonprofit_data = acceptable_quality[acceptable_quality['NAME'] == selected_nonprofit].iloc[0]
        col1, col2 = st.columns(2) # Display nonprofit info
        with col1:
            st.write("**Mission Statement:**")
            st.write(nonprofit_data['mission_statement'])
        with col2:
            st.write("**Financial Information:**")
            revenue = pd.to_numeric(nonprofit_data['REVENUE_AMT'], errors='coerce')
            if pd.notna(revenue):
                st.write(f"Revenue: ${revenue:,.2f}")
            else:
                st.write("Revenue: Not available")
            # Display impact score based on type
            if pd.notna(nonprofit_data['impact_score_numeric']):
                st.write(f"Impact Score: {nonprofit_data['impact_score_numeric']:.2f}")
            elif pd.notna(nonprofit_data['impact_score']):
                st.write(f"Impact Score: {nonprofit_data['impact_score']}")
            else:
                st.write("Impact Score: Not available")
        if st.button("Find Matching Grants"): # Get recommendations
            with st.spinner("Finding best matches..."):
                recommendations = st.session_state.recommender.get_recommendations(
                    mission_statement=nonprofit_data['mission_statement'],
                    nonprofit_info={
                        'name': nonprofit_data['NAME'],
                        'mission_statement': nonprofit_data['mission_statement'],
                        'ntee_code': nonprofit_data.get('NTEE_CD', ''),
                        'ein': nonprofit_data.get('EIN', '')}, top_n=5)
                for i, rec in enumerate(recommendations, 1): # Display recommendations
                    with st.expander(f"{i}. {rec['title']} (Match Score: {rec['similarity_score']:.2f})"):
                        st.write("**Description:**", rec['description'])
                        st.write("**Agency:**", rec['agency'])
                        award_ceiling = pd.to_numeric(rec.get('award_ceiling'), errors='coerce')
                        if pd.notna(award_ceiling):
                            st.write(f"**Award Ceiling:** ${award_ceiling:,.2f}")
                        if 'match_justification' in rec: # Match justification
                            st.write("**Why This Matches:**")
                            for reason in rec['match_justification']['alignment_summary']:
                                if reason:
                                    st.write(f"- {reason}")

def display_fraud_alerts():
    """Display the fraud alerts interface."""
    st.header("Fraud & Anomaly Detection")
    # Use data from session state
    quality = st.session_state.quality
    anomalies = st.session_state.anomalies
    nonprofits = st.session_state.nonprofits
    col1, col2, col3 = st.columns(3) # Summary metrics
    with col1:
        st.metric("Total Anomalies", len(anomalies[anomalies['is_anomalous']]))
    with col2:
        st.metric("Extreme Cases", len(anomalies[anomalies['anomaly_type'] == 'extreme']))
    with col3:
        st.metric("Low Quality Data", len(quality[quality['data_quality'] == 'poor']))    
    st.subheader("Impact Score vs Financial Metric") # Anomaly visualization
    fig = px.scatter(
        anomalies, 
        x='financial_metric', y='impact_score', color='risk_level', 
        symbol='anomaly_type', hover_data=['NAME', 'anomaly_type', 'anomaly_score'],
        log_x=True, title="Anomaly Detection Results",
        color_discrete_map={
            'Critical': '#ff0000', 'High': '#ff7f00',
            'Medium': '#ffff00', 'Low': '#00ff00'})
    st.plotly_chart(fig)
    st.subheader("Risk Level Distribution") # Risk level distribution
    risk_counts = anomalies['risk_level'].value_counts()
    fig = px.pie(
        values=risk_counts.values, names=risk_counts.index,
        title="Risk Level Distribution", color=risk_counts.index,
        color_discrete_map={
            'Critical': '#ff0000', 'High': '#ff7f00',
            'Medium': '#ffff00', 'Low': '#00ff00'})
    st.plotly_chart(fig)
    st.subheader("Data Quality Distribution") # Data quality summary
    quality_counts = quality['data_quality'].value_counts()
    fig = px.pie(
        values=quality_counts.values, names=quality_counts.index,
        title="Data Quality Distribution")
    st.plotly_chart(fig)
    st.subheader("Nonprofit Alert Details") # Create combined alerts table
    alerts_df = quality[['EIN', 'NAME', 'data_quality', 'missing_fields']].copy()
    alerts_df['quality_alert'] = alerts_df['data_quality'].apply(
        lambda x: 'High Risk' if x == 'poor' 
        else 'Medium Risk' if x == 'fair' 
        else 'Low Risk')
    alerts_df = alerts_df.merge( # Add financial information
        nonprofits[['EIN', 'REVENUE_AMT', 'ASSET_AMT', 'impact_score']],
        on='EIN', how='left')
    if not anomalies.empty: # Add anomaly information
        anomaly_info = anomalies[['EIN', 'anomaly_type', 'anomaly_score', 'risk_level']].copy()
        # Map anomaly types to alert levels
        anomaly_info['anomaly_alert'] = anomaly_info['anomaly_type'].apply(
            lambda x: 'Critical' if x in ['extreme_revenue', 'high_impact_low_finance']
            else 'High Risk' if x in ['suspicious_high_impact', 'suspicious_low_impact']
            else 'Medium Risk' if x != 'normal'
            else 'Normal')
        alerts_df = alerts_df.merge(
            anomaly_info[['EIN', 'anomaly_alert', 'anomaly_score', 'risk_level']],
            on='EIN', how='left')
        # Use the risk level from anomalies data
        alerts_df['risk_level'] = alerts_df['risk_level'].fillna('Low')
        alerts_df['anomaly_alert'] = alerts_df['anomaly_alert'].fillna('Normal')
        alerts_df['anomaly_score'] = alerts_df['anomaly_score'].fillna(0)
    else:
        alerts_df['anomaly_alert'] = 'No Data'
        alerts_df['anomaly_score'] = np.nan
        alerts_df['risk_level'] = 'Low'
    alerts_df['risk_level'] = alerts_df['risk_level'].fillna('Low')
    st.write("Filter Alerts:") # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        risk_filter = st.multiselect(
            "Risk Level",
            options=['Critical', 'High', 'Medium', 'Low'],
            default=['Critical', 'High', 'Medium'])
    with col2:
        quality_filter = st.multiselect(
            "Quality Alert Level",
            options=['High Risk', 'Medium Risk', 'Low Risk'],
            default=['High Risk', 'Medium Risk'])
    with col3:
        anomaly_filter = st.multiselect(
            "Anomaly Alert Level",
            options=['Critical', 'High Risk', 'Medium Risk', 'Normal', 'No Data'],
            default=['Critical', 'High Risk', 'Medium Risk'])
    filtered_df = alerts_df[ # Apply filters
        (alerts_df['risk_level'].isin(risk_filter)) &
        (alerts_df['quality_alert'].isin(quality_filter)) &
        (alerts_df['anomaly_alert'].isin(anomaly_filter))]
    # Format financial columns
    filtered_df['REVENUE_AMT'] = filtered_df['REVENUE_AMT'].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "Not Available")
    filtered_df['ASSET_AMT'] = filtered_df['ASSET_AMT'].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "Not Available")
    # Display table
    st.write(f"Showing {len(filtered_df)} nonprofits matching selected filters")
    st.dataframe(
        filtered_df[[
            'NAME', 'EIN', 'risk_level', 'quality_alert', 'anomaly_alert',
            'REVENUE_AMT', 'ASSET_AMT', 'impact_score', 'missing_fields'
        ]].sort_values('risk_level', ascending=False),
        hide_index=True, use_container_width=True)

def display_model_settings():
    """Display the model settings interface."""
    st.header("Model Settings")
    st.write("""Adjust the weights used in the grant matching algorithm. 
    These weights determine the importance of different factors in finding matches.""")
    st.subheader("Matching Weights") # Weight sliders
    similarity_weight = st.slider(
        "Mission Similarity Weight",
        min_value=0.0, max_value=1.0,
        value=st.session_state.model_weights['similarity_weight'],
        help="Weight given to mission statement similarity")
    impact_weight = st.slider(
        "Impact Score Weight",
        min_value=0.0, max_value=1.0,
        value=st.session_state.model_weights['impact_weight'],
        help="Weight given to nonprofit's impact score")
    financial_weight = st.slider(
        "Financial Match Weight",
        min_value=0.0, max_value=1.0,
        value=st.session_state.model_weights['financial_weight'],
        help="Weight given to financial compatibility")
    # Normalize weights
    total = similarity_weight + impact_weight + financial_weight
    if total > 0:
        weights = {
            'similarity_weight': similarity_weight / total,
            'impact_weight': impact_weight / total,
            'financial_weight': financial_weight / total}
        if st.button("Save Weights"): # Update session state
            st.session_state.model_weights = weights
            st.success("Weights updated successfully!")  
            fig = go.Figure(data=[go.Pie( # Visualize weight distribution
                labels=list(weights.keys()),
                values=list(weights.values()), hole=.3)])
            fig.update_layout(title="Weight Distribution")
            st.plotly_chart(fig)

def main():
    """Main function to run the Streamlit app."""
    st.title("GrantMatch AI Dashboard")
    tab1, tab2, tab3 = st.tabs([ # Create tabs
        "Grant Matching", "Fraud Alerts", "Model Settings"])
    # Display content for each tab
    with tab1:
        display_grant_matches()
    with tab2:
        display_fraud_alerts()
    with tab3:
        display_model_settings()

if __name__ == "__main__":
    main() 