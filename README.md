# Churn Prediction — ML + Deep Learning + MLOps

## Sobre o Projeto
Pipeline completo de Machine Learning para prever cancelamento de clientes 
em uma empresa de telecomunicações. O projeto cobre desde análise estatística 
até deploy de aplicação interativa, passando por Deep Learning e MLOps.

## Demo
🚀 **[Acessar o app online](SEU_LINK_AQUI)**

## Pipeline Completo
Dados → Análise Estatística → ML Clássico → Deep Learning → MLOps → App Streamlit

## Tecnologias
- **Análise:** Python, Pandas, SciPy (testes t e chi-quadrado)
- **ML:** scikit-learn (Logistic Regression, Random Forest, Gradient Boosting)
- **Balanceamento:** SMOTE (imbalanced-learn)
- **Otimização:** GridSearchCV
- **Explicabilidade:** SHAP
- **Deep Learning:** PyTorch (rede neural com 14k parâmetros)
- **MLOps:** MLflow (tracking de experimentos e versionamento)
- **App:** Streamlit
- **Versionamento:** Git + GitHub

## Estrutura
projeto-churn-mlops/
├── data/              → Dataset Telco Customer Churn
├── notebooks/
│   ├── 01_analise_estatistica.ipynb
│   ├── 02_machine_learning.ipynb
│   ├── 03_deep_learning.ipynb
│   └── 04_mlops_mlflow.ipynb
├── models/            → Modelos treinados (.pkl, .pt)
├── app/
│   └── app.py         → Aplicação Streamlit
├── mlruns/            → Experimentos MLflow
└── requirements.txt

## Resultados

### Análise Estatística
- Testes t e chi-quadrado identificaram variáveis significativas
- **Contrato mensal:** 42.7% de churn vs 2.8% no contrato de 2 anos
- **Electronic check:** 45.3% de churn — maior risco entre formas de pagamento
- **Fibra óptica:** 41.9% de churn — serviço premium com maior insatisfação

### Machine Learning

| Modelo | Acurácia | AUC-ROC |
|--------|----------|---------|
| Logistic Regression | 74.8% | 0.808 |
| Random Forest | 76.7% | 0.836 |
| Gradient Boosting (otimizado) | 77.7% | 0.819 |
| Rede Neural PyTorch | 77.5% | 0.828 |

**Modelo em produção:** Gradient Boosting otimizado com GridSearchCV

### MLOps
- 3 experimentos registrados no MLflow com parâmetros, métricas e artefatos
- Reprodutibilidade garantida de todos os experimentos

## Como Reproduzir
1. Clone o repositório
2. Baixe o dataset no [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
   e coloque em `data/`
3. Crie o ambiente virtual:
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
4. Execute os notebooks em ordem (01 → 04)
5. Rode o app:
```bash
cd app
streamlit run app.py