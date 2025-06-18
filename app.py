import streamlit as st
import pandas as pd
import numpy as np
from grant_recommender import GrantRecommender
import plotly.express as px
import plotly.graph_objects as go
import logging
import os
import pickle

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
    """Load the grant recommender model."""
    try:
        logger.info("Attempting to load recommender model...")
        
        # Load the pre-trained model from model_output directory
        recommender = GrantRecommender.load_model('model_output')
        
        logger.info("Successfully loaded recommender model")
        return recommender
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
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
            'INCOME_AMT': 'float32', 'ASSET_AMT': 'float32',
            'impact_score': 'string'
        }
        logger.info("Loading nonprofits data...")
        usecols = list(dtypes.keys()) + ['mission_statement', 'impact_score_numeric']
        # Load nonprofits data with optimized settings
        nonprofits = pd.read_csv(
            'data/non-profits_final.csv',
            dtype=dtypes, usecols=usecols,
            na_values=['', 'nan', 'NaN', 'NULL'], low_memory=False)
        logger.info(f"Loaded {len(nonprofits)} nonprofits")
        
        # Ensure numeric columns are properly handled
        for col in ['INCOME_AMT', 'ASSET_AMT']:
            nonprofits[col] = pd.to_numeric(nonprofits[col], errors='coerce')
        
        # Handle impact scores
        if 'impact_score_numeric' not in nonprofits.columns:
            # Create numeric impact score if not present
            impact_score_map = {'Low': 1.0, 'Medium': 2.0, 'High': 3.0}
            nonprofits['impact_score_numeric'] = nonprofits['impact_score'].map(impact_score_map)
        
        logger.info("Loading quality data...")
        try:
            quality = pd.read_csv('data/nonprofit_quality.csv', dtype={'EIN': 'string'})
            logger.info(f"Loaded quality data with {len(quality)} records")
        except FileNotFoundError:
            logger.warning("Quality data not found, generating...")
            st.warning("Generating nonprofit quality data...")
            from generate_final_datasets import assess_data_quality
            quality = assess_data_quality(nonprofits)
            quality.to_csv('data/nonprofit_quality.csv', index=False)
            
        logger.info("Loading anomalies data...")
        try:
            anomalies = pd.read_csv('data/nonprofit_anomalies.csv', dtype={'EIN': 'string'})
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
    
    if st.session_state.recommender is None:
        logger.info("Loading recommender model...")
        st.session_state.recommender = load_recommender()
        
    if st.session_state.recommender is None:
        st.error("Failed to load recommender model. Please check the logs for details.")
        return
    
    logger.info("Using data from session state...")
    nonprofits = st.session_state.nonprofits
    quality = st.session_state.quality
    
    # Filter out organizations with no income or zero income first
    valid_nonprofits = nonprofits[
        (nonprofits['INCOME_AMT'].notna()) &
        (nonprofits['INCOME_AMT'] > 0)
    ]
    
    # Quality filter - include both good and fair quality data
    quality_mask = quality['data_quality'].isin(['good', 'fair', 'excellent'])
    valid_nonprofits = valid_nonprofits[valid_nonprofits['EIN'].isin(
        quality[quality_mask]['EIN']
    )]
    
    if valid_nonprofits.empty:
        st.error("No nonprofits found with valid income data and acceptable quality. Please check the data generation process.")
        return
    
    # Add filters for income range and impact score
    st.sidebar.header("Filters")
    
    # Income range filter with better formatting
    income_min = float(valid_nonprofits['INCOME_AMT'].min())
    income_max = float(valid_nonprofits['INCOME_AMT'].max())
    income_step = (income_max - income_min) / 100
    selected_income_range = st.sidebar.slider(
        "Annual Income Range ($)",
        min_value=income_min,
        max_value=income_max,
        value=(income_min, income_max),
        step=income_step,
        format="$%.0f"
    )
    
    # Impact score filter
    impact_scores = valid_nonprofits['impact_score'].dropna().unique().tolist()
    impact_options = ['All'] + sorted(impact_scores)
    selected_impact = st.sidebar.selectbox(
        "Impact Score",
        options=impact_options,
        index=0
    )
    
    st.subheader("Select Nonprofit")
    
    # Apply filters
    filtered_nonprofits = valid_nonprofits[
        (valid_nonprofits['INCOME_AMT'] >= selected_income_range[0]) &
        (valid_nonprofits['INCOME_AMT'] <= selected_income_range[1])
    ]
    
    if selected_impact != 'All':
        filtered_nonprofits = filtered_nonprofits[filtered_nonprofits['impact_score'] == selected_impact]
    
    # Sort by income for better browsing
    filtered_nonprofits = filtered_nonprofits.sort_values('INCOME_AMT', ascending=False)
    
    logger.info(f"Found {len(filtered_nonprofits)} nonprofits with acceptable quality")
    
    # Add income info to the display name
    filtered_nonprofits['display_name'] = filtered_nonprofits.apply(
        lambda x: f"{x['NAME']} (Annual Income: ${x['INCOME_AMT']:,.2f})", axis=1
    )
    
    selected_nonprofit = st.selectbox(
        "Choose a nonprofit organization:",
        options=filtered_nonprofits['display_name'].unique(),
        help="Showing nonprofits with acceptable data quality, sorted by annual income"
    )
    
    if selected_nonprofit:
        try:
            # Extract the actual name from the display name
            actual_name = selected_nonprofit.split(" (Annual Income")[0]
            nonprofit_data = filtered_nonprofits[filtered_nonprofits['NAME'] == actual_name].iloc[0]
            
            # Display nonprofit details
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Nonprofit Details")
                st.write(f"**Name:** {nonprofit_data['NAME']}")
                st.write(f"**EIN:** {nonprofit_data['EIN']}")
                st.write(f"**Annual Income:** ${nonprofit_data['INCOME_AMT']:,.2f}")
                st.write(f"**Impact Score:** {nonprofit_data['impact_score']}")
                st.write(f"**NTEE Code:** {nonprofit_data['NTEE_CD']}")
            
            with col2:
                st.subheader("Mission Statement")
                st.write(nonprofit_data['mission_statement'])
            
            # Find matching grants
            st.subheader("Matching Grants")
            with st.spinner("Finding matching grants..."):
                matches = st.session_state.recommender.find_matches(
                    nonprofit_data['mission_statement'],
                    k=5
                )
                
                if matches:
                    for i, match in enumerate(matches, 1):
                        score = match['similarity_score']
                        semantic_score = match.get('semantic_score', 0)
                        tfidf_score = match.get('tfidf_score', 0)
                        
                        with st.expander(f"Match {i}: {match['opportunity_title']} (Overall Score: {score:.3f})"):
                            st.write("**Scoring Breakdown:**")
                            st.write(f"- Semantic Similarity: {semantic_score:.3f} (70% weight)")
                            st.write(f"- Text Matching: {tfidf_score:.3f} (30% weight)")
                            
                            st.write("\n**Match Analysis:**")
                            justification = match.get('match_justification', {})
                            if justification:
                                for summary in justification.get('alignment_summary', []):
                                    st.write(f"- {summary}")
                                
                                shared_keywords = justification.get('shared_keywords', [])
                                if shared_keywords:
                                    st.write("\n**Shared Keywords:**")
                                    st.write(", ".join(shared_keywords))
                            
                            st.write("\n**Grant Details:**")
                            st.write(f"**Description:** {match['grant_description']}")
                            st.write(f"**Category:** {match['category']}")
                            st.write(f"**Agency:** {match['agency_name']}")
                            
                            funding_details = []
                            if 'award_ceiling' in match and pd.notna(match['award_ceiling']):
                                funding_details.append(f"Award Ceiling: ${match['award_ceiling']:,.2f}")
                            if 'funding_instrument_type' in match and match['funding_instrument_type']:
                                funding_details.append(f"Funding Type: {match['funding_instrument_type']}")
                            
                            if funding_details:
                                st.write("\n**Funding Information:**")
                                for detail in funding_details:
                                    st.write(f"- {detail}")
                else:
                    st.warning("No matching grants found.")
                    
        except Exception as e:
            st.error(f"Error processing nonprofit data: {str(e)}")
            logger.error(f"Error in display_grant_matches: {str(e)}")

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
        nonprofits[['EIN', 'INCOME_AMT', 'ASSET_AMT', 'impact_score']],
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
    filtered_df['INCOME_AMT'] = filtered_df['INCOME_AMT'].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "Not Available")
    filtered_df['ASSET_AMT'] = filtered_df['ASSET_AMT'].apply(
        lambda x: f"${x:,.2f}" if pd.notna(x) else "Not Available")

    # Display table
    st.write(f"Showing {len(filtered_df)} nonprofits matching selected filters")
    st.dataframe(
        filtered_df[[
            'NAME', 'EIN', 'risk_level', 'quality_alert', 'anomaly_alert',
            'INCOME_AMT', 'ASSET_AMT', 'impact_score', 'missing_fields'
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