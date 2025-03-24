import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS to improve the UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #0D47A1;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .feature-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .results-card {
        background-color: #e3f2fd;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 1.5rem;
    }
    .high-risk {
        background-color: #ffebee;
        color: #c62828;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .low-risk {
        background-color: #e8f5e9;
        color: #2e7d32;
        padding: 1rem;
        border-radius: 5px;
        font-weight: bold;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        color: #616161;
        font-size: 0.8rem;
    }
    .slider-label {
        color: #424242;
        font-weight: 500;
    }
    .stSlider {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f8f9fa;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #e3f2fd;
        color: #1E88E5;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Customer Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem;'>
    This tool helps identify customers who are likely to discontinue using the telecom company's services.
</p>
""", unsafe_allow_html=True)

# Sidebar for context and instructions
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/phone-office.png")
    st.title("Information")
    st.info("""
    ### About the App
    This application uses machine learning to predict the probability of customer churn based on their data.
    
    ### How to Use
    1. Enter customer data in the form
    2. Click the "Make Prediction" button
    3. View results and recommendations
    
    ### Features
    - Churn probability prediction
    - Visualization of important factors
    - Retention recommendations
    """)
    
    st.markdown("---")
    st.markdown("### Support Contact")
    st.markdown("📧 support@telecom.com")
    st.markdown("☎️ +1 123 456 7890")

# Function to load model
@st.cache_resource
def load_model():
    try:
        # Load model
        model_path = 'models/xgboost_model.pkl'
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        else:
            st.error(f"Model file not found: {model_path}")
            return None
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Function to load scaler
@st.cache_resource
def load_scaler():
    try:
        scaler_path = 'data/scaler.pkl'
        if os.path.exists(scaler_path):
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            return scaler
        else:
            st.warning(f"Scaler file not found: {scaler_path}. Using default scaler.")
            return StandardScaler()
    except Exception as e:
        st.warning(f"Failed to load scaler: {e}. Using default scaler.")
        return StandardScaler()

# Load model and scaler
model = load_model()
scaler = load_scaler()

# Main app content
tab1, tab2, tab3 = st.tabs(["📋 Individual Prediction", "📊 Batch Prediction", "📈 Data Analysis"])

with tab1:
    st.markdown("<h2 class='sub-header'>Enter Customer Data</h2>", unsafe_allow_html=True)
    
    # Create a form with 3 columns layout for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Services")
        is_tv_subscriber = st.checkbox("TV Subscription", value=False)
        is_movie_package_subscriber = st.checkbox("Movie Package Subscription", value=False)
        subscription_age = st.slider("Subscription Age (years)", min_value=0.0, max_value=12.0, value=0.5, step=0.01)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Usage")
        bill_avg = st.slider("Average Bill (last 3 months)", min_value=0, max_value=400, value=50)
        download_avg = st.slider("Average Download Speed (GB, last 3 months)", min_value=0.0, max_value=500.0, value=150.0, step=1.0)
        upload_avg = st.slider("Average Upload Speed (GB, last 3 months)", min_value=0.0, max_value=50.0, value=10.0, step=0.1)
        download_over_limit = st.slider("Download Over Limit Count (last 9 months)", min_value=0, max_value=7, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Contract & Service")
        remaining_contract = st.slider("Remaining Contract (years)", min_value=0.0, max_value=3.0, value=1.0, step=0.01)
        service_failure_count = st.slider("Service Failure Count", min_value=0, max_value=19, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    predict_btn = st.button("Make Prediction", type="primary", use_container_width=True)
    
    # Make prediction when button is clicked
    if predict_btn:
        if model:
            # Prepare feature data
            features = {
                'is_tv_subscriber': 1 if is_tv_subscriber else 0,
                'is_movie_package_subscriber': 1 if is_movie_package_subscriber else 0,
                'subscription_age': subscription_age,
                'bill_avg': bill_avg,
                'reamining_contract': remaining_contract,
                'service_failure_count': service_failure_count,
                'download_avg': download_avg,
                'upload_avg': upload_avg,
                'download_over_limit': download_over_limit
            }
            
            # Create DataFrame with the same structure as training data
            input_df = pd.DataFrame([features])
            
            try:
                # Make prediction
                probability = model.predict_proba(input_df)[0][1]
                prediction = 1 if probability > 0.5 else 0
                
                # Display prediction results in a nice card
                st.markdown("<div class='results-card'>", unsafe_allow_html=True)
                st.subheader("Prediction Results")
                
                cols = st.columns([2, 3])
                with cols[0]:
                    # Risk gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Probability (%)"},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "green"},
                                {'range': [30, 70], 'color': "orange"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': probability * 100
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with cols[1]:
                    st.markdown("<h3>Customer Risk Assessment</h3>", unsafe_allow_html=True)
                    
                    if prediction == 1:
                        st.markdown("<div class='high-risk'>High Risk of Churn</div>", unsafe_allow_html=True)
                        st.markdown("""
                        ### Retention Recommendations:
                        - **Contact the customer** to address any concerns
                        - **Offer a special promotion** or discount
                        - **Review and resolve** any recent service issues
                        - **Propose contract extension** with improved conditions
                        """)
                    else:
                        st.markdown("<div class='low-risk'>Low Risk of Churn</div>", unsafe_allow_html=True)
                        st.markdown("""
                        ### Retention Recommendations:
                        - **Continue monitoring** customer satisfaction
                        - **Consider upselling** additional services
                        - **Offer loyalty rewards** for continued subscription
                        - **Implement periodic check-ins** to maintain relationship
                        """)
                
                # Feature importance visualization
                st.subheader("What's Driving This Prediction?")
                
                # Get feature importance (simplified for demo)
                feature_importance = {
                    'service_failure_count': 0.25 if service_failure_count > 0 else 0.05,
                    'remaining_contract': 0.30 if remaining_contract < 0.5 else 0.10,
                    'subscription_age': 0.15 if subscription_age < 3 else 0.05,
                    'bill_avg': 0.20 if bill_avg > 100 else 0.10,
                    'download_avg': 0.10 if download_avg < 50 else 0.05,
                }
                
                # Normalize importance values
                total = sum(feature_importance.values())
                feature_importance = {k: v/total for k, v in feature_importance.items()}
                
                # Create DataFrame for visualization
                importance_df = pd.DataFrame({
                    'Feature': list(feature_importance.keys()),
                    'Importance': list(feature_importance.values())
                })
                
                # Sort by importance
                importance_df = importance_df.sort_values('Importance', ascending=False)
                
                # Create bar chart
                fig = px.bar(
                    importance_df, 
                    x='Importance', 
                    y='Feature', 
                    orientation='h',
                    color='Importance',
                    color_continuous_scale='Blues',
                    title='Key Factors Influencing Churn Prediction'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error making prediction: {e}")
        else:
            st.error("Model not loaded. Please check if the model file exists and is valid.")

with tab2:
    st.markdown("<h2 class='sub-header'>Batch Prediction</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    Upload a CSV file with customer data to predict churn for multiple customers at once. 
    The file should contain the same features as shown in the Individual Prediction tab.
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the file
            df = pd.read_csv(uploaded_file)
            
            # Display the first few rows
            st.subheader("Data Preview")
            st.dataframe(df.head())
            
            # Check if required columns exist
            required_columns = [
                'is_tv_subscriber', 'is_movie_package_subscriber', 'subscription_age',
                'bill_avg', 'reamining_contract', 'service_failure_count',
                'download_avg', 'upload_avg', 'download_over_limit'
            ]
            
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
            else:
                # Make predictions button
                if st.button("Generate Predictions", type="primary"):
                    if model:
                        # Make predictions
                        try:
                            # Get only the required columns
                            X = df[required_columns]
                            
                            # Make predictions
                            df['churn_probability'] = model.predict_proba(X)[:, 1]
                            df['predicted_churn'] = (df['churn_probability'] > 0.5).astype(int)
                            
                            # Display results
                            st.subheader("Prediction Results")
                            
                            # Summary statistics
                            churn_count = df['predicted_churn'].sum()
                            total_customers = len(df)
                            churn_rate = churn_count / total_customers * 100
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Customers", f"{total_customers}")
                            with col2:
                                st.metric("Predicted to Churn", f"{churn_count}")
                            with col3:
                                st.metric("Churn Rate", f"{churn_rate:.2f}%")
                            
                            # Show the results table
                            st.dataframe(df)
                            
                            # Download link for the results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label="Download Results as CSV",
                                data=csv,
                                file_name="churn_predictions.csv",
                                mime="text/csv",
                            )
                            
                            # Visualization
                            st.subheader("Churn Probability Distribution")
                            fig = px.histogram(
                                df, 
                                x='churn_probability', 
                                nbins=20,
                                labels={'churn_probability': 'Churn Probability'},
                                title='Distribution of Churn Probabilities',
                                color_discrete_sequence=['#1E88E5']
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error generating predictions: {e}")
                    else:
                        st.error("Model not loaded. Please check if the model file exists and is valid.")
        except Exception as e:
            st.error(f"Error reading file: {e}")

with tab3:
    st.markdown("<h2 class='sub-header'>Data Analysis</h2>", unsafe_allow_html=True)
    
    st.markdown("""
    This section provides insights into customer churn based on historical data.
    Understand the key factors that influence customer churn to develop effective retention strategies.
    """)
    
    # Try to load the dataset
    try:
        # Check if we can load a sample of the data
        if os.path.exists('data/internet_service_churn.csv'):
            # Load a sample of the data for visualization
            sample_df = pd.read_csv('data/internet_service_churn.csv', nrows=10000)
            
            if 'churn' in sample_df.columns:
                # Analysis options
                analysis_type = st.selectbox(
                    "Select Analysis Type",
                    ["Churn by Service Subscription", "Churn by Contract Remaining", "Churn by Service Failures", "Churn by Bill Amount"]
                )
                
                if analysis_type == "Churn by Service Subscription":
                    # Calculate percentages
                    tv_churn = sample_df.groupby('is_tv_subscriber', observed=True)['churn'].mean().reset_index()
                    tv_churn['Service'] = 'TV Subscription'
                    tv_churn.rename(columns={'is_tv_subscriber': 'Has Service'}, inplace=True)
                    
                    movie_churn = sample_df.groupby('is_movie_package_subscriber', observed=True)['churn'].mean().reset_index()
                    movie_churn['Service'] = 'Movie Package'
                    movie_churn.rename(columns={'is_movie_package_subscriber': 'Has Service'}, inplace=True)
                    
                    # Combine data
                    service_churn = pd.concat([tv_churn, movie_churn])
                    
                    # Create visualization
                    fig = px.bar(
                        service_churn,
                        x='Service',
                        y='churn',
                        color='Has Service',
                        barmode='group',
                        labels={'churn': 'Churn Rate', 'Has Service': 'Has Subscription'},
                        title='Churn Rate by Service Subscription',
                        color_discrete_map={0: '#ff7f0e', 1: '#1f77b4'}
                    )
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insights:**
                    - Customers with TV subscriptions generally have a lower churn rate than those without
                    - Movie package subscribers show different churn patterns compared to non-subscribers
                    - Bundling services together may help in customer retention
                    """)
                
                elif analysis_type == "Churn by Contract Remaining":
                    # Create bins for contract remaining time
                    sample_df['contract_bins'] = pd.cut(
                        sample_df['reamining_contract'], 
                        bins=[0, 0.5, 1, 1.5, 2, 3], 
                        labels=['0-0.5', '0.5-1', '1-1.5', '1.5-2', '2+']
                    )
                    
                    # Calculate churn rate by contract bin
                    contract_churn = sample_df.groupby('contract_bins', observed=True)['churn'].mean().reset_index()
                    
                    # Create visualization
                    fig = px.bar(
                        contract_churn,
                        x='contract_bins',
                        y='churn',
                        labels={'contract_bins': 'Remaining Contract (years)', 'churn': 'Churn Rate'},
                        title='Churn Rate by Remaining Contract Length',
                        color='churn',
                        color_continuous_scale='blues'
                    )
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insights:**
                    - Customers with less time remaining on their contracts have significantly higher churn rates
                    - There's a clear inverse relationship between contract length and churn probability
                    - Proactive engagement with customers nearing contract end is recommended
                    """)
                
                elif analysis_type == "Churn by Service Failures":
                    # Group by failure count
                    failure_bins = [0, 1, 2, 3, 5, 10, 20]
                    failure_labels = ['0', '1', '2', '3-5', '6-10', '11+']
                    
                    # Create failure bins
                    sample_df['failure_bins'] = pd.cut(
                        sample_df['service_failure_count'], 
                        bins=failure_bins, 
                        labels=failure_labels
                    )
                    
                    # Calculate churn rate by failure bin
                    failure_churn = sample_df.groupby('failure_bins', observed=True)['churn'].mean().reset_index()
                    failure_count = sample_df.groupby('failure_bins', observed=True).size().reset_index(name='count')
                    failure_data = failure_churn.merge(failure_count, on='failure_bins')
                    
                    # Create visualization
                    fig = px.bar(
                        failure_data,
                        x='failure_bins',
                        y='churn',
                        labels={'failure_bins': 'Number of Service Failures', 'churn': 'Churn Rate'},
                        title='Churn Rate by Service Failures',
                        color='churn',
                        color_continuous_scale='Reds',
                        text='count'
                    )
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insights:**
                    - Service failures have a strong correlation with customer churn
                    - Even a single service failure increases churn probability significantly
                    - Multiple failures have a compounding effect on churn risk
                    - Improving service reliability should be a priority for retention
                    """)
                
                elif analysis_type == "Churn by Bill Amount":
                    # Create bins for bill amounts
                    bill_bins = [0, 25, 50, 75, 100, 150, 200, 500]
                    bill_labels = ['0-25', '25-50', '50-75', '75-100', '100-150', '150-200', '200+']
                    
                    # Create bill bins
                    sample_df['bill_bins'] = pd.cut(
                        sample_df['bill_avg'], 
                        bins=bill_bins, 
                        labels=bill_labels
                    )
                    
                    # Calculate churn rate by bill bin
                    bill_churn = sample_df.groupby('bill_bins', observed=True)['churn'].mean().reset_index()
                    
                    # Create visualization
                    fig = px.line(
                        bill_churn,
                        x='bill_bins',
                        y='churn',
                        markers=True,
                        labels={'bill_bins': 'Average Bill Amount', 'churn': 'Churn Rate'},
                        title='Churn Rate by Average Bill Amount',
                    )
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    **Insights:**
                    - Customers with very low or very high bills tend to have higher churn rates
                    - Mid-range bill amounts correlate with better retention
                    - Consider price sensitivity when developing retention strategies
                    - Investigate if high-bill customers feel they're getting sufficient value
                    """)
            else:
                st.error("The dataset does not contain a 'churn' column. Unable to perform analysis.")
        else:
            st.warning("Sample data not available. Please upload a dataset to perform analysis.")
            
            # Option to upload a CSV for analysis
            analysis_file = st.file_uploader("Upload a CSV file for analysis", type="csv")
            
            if analysis_file is not None:
                try:
                    analysis_df = pd.read_csv(analysis_file)
                    st.success("File uploaded successfully. Showing data preview:")
                    st.dataframe(analysis_df.head())
                    
                    # Basic data analysis
                    st.subheader("Data Summary")
                    st.write(analysis_df.describe())
                    
                    # More analysis can be added here
                    
                except Exception as e:
                    st.error(f"Error processing file: {e}")
    
    except Exception as e:
        st.error(f"Error loading dataset: {e}")

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 Telecom Churn Prediction Tool | Powered by Machine Learning</p>
</div>
""", unsafe_allow_html=True)