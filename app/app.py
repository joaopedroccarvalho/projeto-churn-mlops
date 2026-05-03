import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

# Configuração da página
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

# Carregar modelo e scaler
@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model = joblib.load(os.path.join(base_path, 'models', 'gradient_boosting_churn.pkl'))
    scaler = joblib.load(os.path.join(base_path, 'models', 'scaler.pkl'))
    feature_names = joblib.load(os.path.join(base_path, 'models', 'feature_names.pkl'))
    return model, scaler, feature_names

model, scaler, feature_names = load_model()

# Título
st.title("📊 Churn Predictor — Previsão de Cancelamento")
st.markdown("Preencha os dados do cliente para prever a probabilidade de cancelamento.")
st.divider()

# Sidebar com inputs
st.sidebar.header("Dados do Cliente")

tenure = st.sidebar.slider("Tempo como cliente (meses)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Valor mensal (R$)", 18.0, 120.0, 65.0)
total_charges = monthly_charges * tenure

contract = st.sidebar.selectbox("Tipo de contrato", 
    ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Serviço de internet",
    ["DSL", "Fiber optic", "No"])
payment_method = st.sidebar.selectbox("Forma de pagamento",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
tech_support = st.sidebar.selectbox("Suporte técnico", ["No", "Yes", "No internet service"])
online_security = st.sidebar.selectbox("Segurança online", ["No", "Yes", "No internet service"])
senior_citizen = st.sidebar.selectbox("Idoso (65+)?", ["Não", "Sim"])
partner = st.sidebar.selectbox("Tem parceiro(a)?", ["Yes", "No"])
dependents = st.sidebar.selectbox("Tem dependentes?", ["Yes", "No"])
multiple_lines = st.sidebar.selectbox("Múltiplas linhas?", ["No", "Yes", "No phone service"])
online_backup = st.sidebar.selectbox("Backup online?", ["No", "Yes", "No internet service"])
device_protection = st.sidebar.selectbox("Proteção do dispositivo?", ["No", "Yes", "No internet service"])
streaming_tv = st.sidebar.selectbox("Streaming TV?", ["No", "Yes", "No internet service"])
streaming_movies = st.sidebar.selectbox("Streaming filmes?", ["No", "Yes", "No internet service"])
paperless_billing = st.sidebar.selectbox("Fatura digital?", ["Yes", "No"])

# Montar o dicionário de features
input_data = {
    'SeniorCitizen': 1 if senior_citizen == "Sim" else 0,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'MultipleLines': multiple_lines,
    'InternetService': internet_service,
    'OnlineSecurity': online_security,
    'OnlineBackup': online_backup,
    'DeviceProtection': device_protection,
    'TechSupport': tech_support,
    'StreamingTV': streaming_tv,
    'StreamingMovies': streaming_movies,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Converter para dataframe e fazer encoding
input_df = pd.DataFrame([input_data])

# Encoding manual das categóricas
categoricas = ['Partner', 'Dependents', 'MultipleLines', 'InternetService',
               'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
               'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
               'PaymentMethod']

input_encoded = pd.get_dummies(input_df, columns=categoricas, drop_first=True)

# Alinhar com as features do modelo
input_aligned = pd.DataFrame(0, index=[0], columns=feature_names)
for col in input_encoded.columns:
    if col in feature_names:
        input_aligned[col] = input_encoded[col].values

# Normalizar e prever
input_scaled = scaler.transform(input_aligned)
proba = model.predict_proba(input_scaled)[0][1]
prediction = 1 if proba > 0.5 else 0

# Exibir resultado
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Probabilidade de Churn", f"{proba*100:.1f}%")

with col2:
    risco = "🔴 Alto" if proba > 0.6 else "🟡 Médio" if proba > 0.3 else "🟢 Baixo"
    st.metric("Nível de Risco", risco)

with col3:
    st.metric("Tempo como cliente", f"{tenure} meses")

st.divider()

# Gauge chart
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=proba * 100,
    title={'text': "Probabilidade de Cancelamento (%)"},
    gauge={
        'axis': {'range': [0, 100]},
        'bar': {'color': "#EF4444" if proba > 0.5 else "#10B981"},
        'steps': [
            {'range': [0, 30], 'color': "#D1FAE5"},
            {'range': [30, 60], 'color': "#FEF3C7"},
            {'range': [60, 100], 'color': "#FEE2E2"}
        ],
        'threshold': {
            'line': {'color': "black", 'width': 4},
            'thickness': 0.75,
            'value': 50
        }
    }
))

fig.update_layout(height=350, template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# Recomendações
st.subheader("💡 Recomendações")

if proba > 0.6:
    st.error("**Alto risco de cancelamento!** Ações recomendadas:")
    if contract == "Month-to-month":
        st.write("• Oferecer desconto para migrar para contrato anual")
    if payment_method == "Electronic check":
        st.write("• Incentivar migração para pagamento automático")
    if tech_support == "No":
        st.write("• Oferecer período gratuito de suporte técnico")
    if tenure < 12:
        st.write("• Ativar programa de fidelidade para clientes novos")
elif proba > 0.3:
    st.warning("**Risco médio.** Monitorar e engajar proativamente.")
else:
    st.success("**Baixo risco.** Cliente com bom perfil de retenção.")