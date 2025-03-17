import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Телеком Прогноз Відтоку",
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
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<h1 class='main-header'>Прогнозування Відтоку Клієнтів</h1>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align: center; font-size: 1.2rem;'>
    Цей інструмент допомагає виявити клієнтів, які можуть припинити користування послугами телекомунікаційної компанії.
</p>
""", unsafe_allow_html=True)

# Sidebar for context and instructions
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/phone-office.png")
    st.title("Інформація")
    st.info("""
    ### Про додаток
    Цей додаток використовує машинне навчання для прогнозування ймовірності відтоку клієнтів на основі їх даних.
    
    ### Як використовувати
    1. Введіть дані клієнта у форму
    2. Натисніть кнопку "Здійснити прогноз"
    3. Перегляньте результати та рекомендації
    
    ### Функції
    - Прогноз ймовірності відтоку
    - Візуалізація важливості факторів
    - Рекомендації щодо утримання клієнта
    """)
    
    st.markdown("---")
    st.markdown("### Контакти підтримки")
    st.markdown("📧 support@telecom.com")
    st.markdown("☎️ +380 44 123 45 67")

# Function to load model
@st.cache_resource
def load_model():
    # Check if model exists, for this demo we'll create a simple placeholder
    if not os.path.exists('models/churn_model.pkl'):
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Create a simple placeholder model for demo purposes
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Save placeholder model
        with open('models/churn_model.pkl', 'wb') as f:
            pickle.dump(model, f)
    
    # Load model
    with open('models/churn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    return model

# Load model
try:
    model = load_model()
except Exception as e:
    st.error(f"Не вдалося завантажити модель: {e}")
    model = None

# Main app content
tab1, tab2 = st.tabs(["📋 Прогнозування", "📊 Аналітика"])

with tab1:
    st.markdown("<h2 class='sub-header'>Введіть дані клієнта</h2>", unsafe_allow_html=True)
    
    # Create a form with 3 columns layout for better organization
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Послуги")
        is_tv_subscriber = st.checkbox("Підписка на ТВ", value=False)
        is_movie_package_subscriber = st.checkbox("Підписка на кінопакет", value=False)
        subscription_age = st.slider("Вік підписки (місяців)", min_value=0, max_value=24, value=12, step=1)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Використання")
        bill_avg = st.slider("Середній рахунок (грн)", min_value=0, max_value=100, value=25)
        download_avg = st.slider("Середня швидкість завантаження (Мбіт/с)", min_value=0.0, max_value=30.0, value=8.0, step=0.1)
        upload_avg = st.slider("Середня швидкість вивантаження (Мбіт/с)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
        download_over_limit = st.checkbox("Перевищення ліміту завантаження", value=False)
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='feature-section'>", unsafe_allow_html=True)
        st.subheader("Контракт і сервіс")
        reamining_contract = st.slider("Залишок контракту (років)", min_value=0, max_value=5, value=2, step=1)
        service_failure_count = st.slider("Кількість збоїв сервісу", min_value=0, max_value=10, value=0)
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    predict_btn = st.button("Здійснити прогноз", type="primary", use_container_width=True)
    
    # Make prediction when button is clicked
    if predict_btn:
        if model:
            # Prepare feature data
            features = {
                'is_tv_subscriber': 1 if is_tv_subscriber else 0,
                'is_movie_package_subscriber': 1 if is_movie_package_subscriber else 0,
                'subscription_age': subscription_age,
                'bill_avg': bill_avg,
                'reamining_contract': reamining_contract,
                'service_failure_count': service_failure_count,
                'download_avg': download_avg,
                'upload_avg': upload_avg,
                'download_over_limit': 1 if download_over_limit else 0
            }
            
            # Create DataFrame with the same structure as training data
            input_df = pd.DataFrame([features])
            
            # For demo purposes only - random prediction
            # In production, this would use the actual trained model
            # prediction = model.predict(input_df)[0]
            # probability = model.predict_proba(input_df)[0][1]
            
            # Placeholder for demo - using logic based on important features
            if reamining_contract > 0.8 and is_tv_subscriber:
                prediction = 0
                probability = 0.2
            elif service_failure_count > 2 and reamining_contract < 0.3:
                prediction = 1
                probability = 0.8
            else:
                # Use some simple heuristics to generate a probability
                base_prob = 0.5
                factors = [
                    -0.3 if is_tv_subscriber else 0,
                    -0.2 if is_movie_package_subscriber else 0,
                    -0.01 * subscription_age,
                    0.005 * bill_avg,
                    -0.5 * reamining_contract,
                    0.05 * service_failure_count,
                    -0.01 * download_avg,
                    -0.01 * upload_avg,
                    0.1 if download_over_limit else 0
                ]
                probability = base_prob + sum(factors)
                probability = max(min(probability, 0.95), 0.05)  # Keep between 0.05 and 0.95
                prediction = 1 if probability > 0.5 else 0
            
            # Display prediction results in a nice card
            st.markdown("<div class='results-card'>", unsafe_allow_html=True)
            st.subheader("Результати прогнозування")
            
            cols = st.columns([2, 3])
            with cols[0]:
                # Risk gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Ймовірність відтоку (%)"},
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
                
                # Risk classification
                if prediction == 1:
                    st.markdown("<div class='high-risk'>ВИСОКИЙ РИЗИК ВІДТОКУ</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<div class='low-risk'>НИЗЬКИЙ РИЗИК ВІДТОКУ</div>", unsafe_allow_html=True)
            
            with cols[1]:
                st.subheader("Важливі фактори")
                
                # Display factors affecting the prediction
                factors_data = {
                    'Фактор': [
                        'Залишок контракту',
                        'Кількість збоїв',
                        'Підписка на ТВ',
                        'Перевищення ліміту',
                        'Вік підписки'
                    ],
                    'Вплив': [
                        -0.5 * reamining_contract * 100,
                        0.05 * service_failure_count * 100,
                        -0.3 * (1 if is_tv_subscriber else 0) * 100,
                        0.1 * (1 if download_over_limit else 0) * 100,
                        -0.01 * subscription_age * 100
                    ]
                }
                
                df_factors = pd.DataFrame(factors_data)
                fig = px.bar(
                    df_factors, 
                    x='Вплив', 
                    y='Фактор',
                    orientation='h',
                    color='Вплив',
                    color_continuous_scale=['green', 'yellow', 'red'],
                    range_color=[-50, 50]
                )
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations
                st.subheader("Рекомендації")
                if prediction == 1:
                    recommendations = []
                    if reamining_contract < 0.3:
                        recommendations.append("Запропонувати новий контракт з кращими умовами")
                    if not is_tv_subscriber:
                        recommendations.append("Запропонувати ТВ-пакет зі знижкою")
                    if service_failure_count > 2:
                        recommendations.append("Провести технічну перевірку якості послуг")
                    if bill_avg > 50:
                        recommendations.append("Переглянути тарифний план клієнта")
                    
                    for rec in recommendations:
                        st.markdown(f"✅ {rec}")
                else:
                    st.markdown("✅ Клієнт має низький ризик відтоку. Рекомендуємо підтримувати якість обслуговування.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.error("Модель не завантажена. Будь ласка, перезавантажте сторінку або зверніться до адміністратора.")

with tab2:
    st.markdown("<h2 class='sub-header'>Аналітика відтоку клієнтів</h2>", unsafe_allow_html=True)
    
    # Load sample data for demo visualizations
    try:
        df = pd.read_csv("data/internet_service_churn.csv")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Churn distribution
            fig = px.pie(
                df, 
                names='churn',
                title='Розподіл відтоку клієнтів',
                color_discrete_sequence=px.colors.sequential.Blues_r,
                hole=0.4,
                labels={'0': 'Залишились', '1': 'Відтік'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # TV subscriber vs. churn
            tv_churn = df.groupby(['is_tv_subscriber', 'churn']).size().unstack().reset_index()
            tv_churn.columns = ['ТВ-підписка', 'Залишились', 'Відтік']
            tv_churn['ТВ-підписка'] = tv_churn['ТВ-підписка'].map({0: 'Немає', 1: 'Є'})
            fig = px.bar(
                tv_churn, 
                x='ТВ-підписка', 
                y=['Залишились', 'Відтік'],
                title='Вплив ТВ-підписки на відтік',
                barmode='group',
                color_discrete_sequence=['#1E88E5', '#E53935']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Contract remaining vs. churn
            fig = px.scatter(
                df, 
                x='reamining_contract',
                y='bill_avg',
                color='churn',
                color_discrete_sequence=['#1E88E5', '#E53935'],
                opacity=0.7,
                title='Залежність відтоку від залишку контракту та середнього рахунку'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Service failures impact
            df_failures = df.groupby('service_failure_count')['churn'].mean().reset_index()
            df_failures.columns = ['Кількість збоїв', 'Середній відтік']
            fig = px.line(
                df_failures, 
                x='Кількість збоїв', 
                y='Середній відтік', 
                markers=True,
                title='Вплив кількості збоїв на відтік',
                color_discrete_sequence=['#E53935']
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)
    
    except Exception as e:
        st.error(f"Не вдалося завантажити дані для аналітики: {e}")
        st.info("Демо-режим: показуються вигадані дані")
        
        # Create sample data
        st.markdown("### Симуляція даних для демонстрації")
        
        # Synthetic charts
        col1, col2 = st.columns(2)
        with col1:
            labels = ['Залишились', 'Відтік']
            values = [75, 25]
            fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.4, marker_colors=['#1E88E5', '#E53935'])])
            fig.update_layout(title_text='Розподіл відтоку клієнтів')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            x = ['0-6 місяців', '6-12 місяців', '1-2 роки', '2+ роки']
            y = [0.42, 0.28, 0.15, 0.05]
            fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color='#E53935')])
            fig.update_layout(title_text='Відтік за тривалістю користування')
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("""
<div class='footer'>
    <p>© 2023 Телекомунікаційна компанія. Всі права захищені.</p>
    <p>Розроблено службою аналітики даних</p>
</div>
""", unsafe_allow_html=True)