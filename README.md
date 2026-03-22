# 💼 Previsor de Salários — Brasil

Projeto de Machine Learning para previsão de salários usando dados do mercado brasileiro.

## 📋 Pré-requisitos

- Python 3.8+
- pip ou conda

## 🚀 Setup Inicial

1. **Clonar o repositório:**
```bash
git clone https://github.com/andradefisio/grupo10-desafio-versao-final.git
cd grupo10-desafio-versao-final
```

2. **Criar ambiente virtual e instalar dependências:**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux

pip install -r requirements.txt
```

## 🔄 Gerar Arquivos Necessários

Execute os scripts Python **em ordem**:

```bash
# 1. Análise Exploratória de Dados (EDA)
python 1_eda.py

# 2. Transformação e Limpeza dos Dados
python 2_transformacao.py

# 3. Modelagem e Treinamento
python 3_modelagem.py
```

Isso vai gerar:
- `dataset_limpo.csv` — Dataset processado
- `X_modelagem.csv` e `y_modelagem.csv` — Features e target para modelagem
- `modelo_final.pkl` — Modelo treinado
- `scaler.pkl` — Scaler para normalização
- `escolaridade_map.pkl` — Mapeamento de escolaridade
- `nome_modelo.pkl` — Nome do modelo utilizado
- `eda_*.png` — Gráficos da análise exploratória

## 🎨 Iniciar a Interface Web

```bash
streamlit run app.py
```

A aplicação abrirá em `http://localhost:8501`

## 📊 Estrutura do Projeto

```
.
├── 1_eda.py                    # Análise exploratória
├── 2_transformacao.py          # Limpeza e transformação
├── 3_modelagem.py              # Treinamento do modelo
├── app.py                      # Interface Streamlit
├── dataset_salarios_brasil.csv # Dataset original
├── requirements.txt            # Dependências
└── README.md                   # Este arquivo
```

## ⚠️ Aviso

Este é um projeto **acadêmico** desenvolvido para fins de estudo em Machine Learning. As previsões são estimativas experimentais e não devem ser usadas como referência profissional.
