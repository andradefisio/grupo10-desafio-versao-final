# Previsor de Salários — Brasil

> Projeto acadêmico da disciplina de Machine Learning — Modelo Linear Generalizado (GLM) para previsão de salários com deploy em Streamlit.

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange) ![Streamlit](https://img.shields.io/badge/Streamlit-deploy-red?logo=streamlit) ![GLM](https://img.shields.io/badge/Modelo-GLM%20Tweedie-1A936F)

---

## Resultados em Destaque

| Métrica | Valor |
|---|---|
| **Modelo final** | GLM Tweedie (power=1,5) |
| **R² no teste** | 0,669 |
| **MAE** | R$ 1.706 |
| **MAPE** | 17,6% |
| **CV-R² (5-fold)** | 0,535 ± 0,086 |

---

## Estrutura do Repositório

```
.
├── dataset_salarios_brasil.csv   # Dataset original
├── 1_eda.py                      # Etapa 1: Análise Exploratória
├── 2_transformacao.py            # Etapa 2: Transformação dos dados
├── 3_modelagem.py                # Etapa 3: Treinamento dos modelos
├── app.py                        # Interface web (Streamlit)
├── requirements.txt              # Dependências
│
│   # Gerados ao rodar os scripts:
├── dataset_limpo.csv
├── X_modelagem.csv
├── y_modelagem.csv
├── scaler.pkl
├── colunas_modelo.pkl
├── escolaridade_map.pkl
├── modelo_final.pkl
├── nome_modelo.pkl
└── eda_*.png                     # 12 gráficos gerados pelo 1_eda.py
```

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/seu-usuario/seu-repo.git
cd seu-repo
```

### 2. Criar o ambiente virtual com Python 3.12

```bash
python3.12 -m venv .venv

# Ativar (Mac/Linux)
source .venv/bin/activate

# Ativar (Windows)
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Rodar os scripts na ordem

```bash
python 1_eda.py
python 2_transformacao.py
python 3_modelagem.py
```

### 5. Iniciar o app

```bash
streamlit run app.py
```

---

## 🔍 Etapa 1 — Análise Exploratória (EDA)

### Dataset

| Atributo | Valor |
|---|---|
| Arquivo | `dataset_salarios_brasil.csv` |
| Total de registros brutos | 15.000 linhas |
| Número de colunas | 8 variáveis |
| Variável alvo | Salário (R$) |

### Variáveis

| Coluna | Tipo | Uso no Modelo |
|---|---|---|
| Idade | Numérica | ❌ **REMOVIDA** (r ≈ 0,013) |
| Anos_Experiencia | Numérica | ✅ Mantida (r ≈ 0,29) |
| Escolaridade | Categ. Ordinal | ✅ Encoding ordinal 1–7 |
| Segunda_Lingua | Categ. Nominal | ✅ Binário (tem/não tem) |
| Terceira_Lingua | Categ. Nominal | ✅ Binário (tem/não tem) |
| Regiao | Categ. Nominal | ✅ Dummies (one-hot) |
| Profissao | Categ. Nominal | ✅ Dummies (one-hot) |
| Salario | Numérica | 🎯 Variável ALVO |

### Qualidade dos Dados

| Problema | Coluna | Quantidade | Ação |
|---|---|---|---|
| Valores texto `"erro"` | Salário | 200 | Convertidos para NaN e removidos |
| Valores ausentes (NaN) | Salário | 297 | Removidos (alvo não imputável) |
| Valores ausentes | Anos_Experiencia | 298 | Imputados pela mediana |
| Outliers extremos (top 1%) | Salário | 146 | Removidos (acima de R$ 118.328) |
| **Dataset final** | | **14.357 registros** | |

### Distribuição do Salário

| Estatística | Valor |
|---|---|
| Mínimo | R$ 1.472 |
| 1º Quartil (Q1) | R$ 6.064 |
| Mediana | R$ 9.260 |
| Média | R$ 13.370 |
| 3º Quartil (Q3) | R$ 14.077 |
| Máximo (após corte p99) | R$ 118.328 |

> A distribuição é fortemente assimétrica à direita. Após `log(Salário+1)`, a distribuição se aproxima de uma normal — justificando o GLM com link logarítmico.

### Correlação com log(Salário)

| Variável | r de Pearson | p-value | Decisão |
|---|---|---|---|
| Idade | 0,013 | 0,10 (não significativo) | ❌ Removida |
| Anos_Experiencia | 0,293 | < 0,001 | ✅ Mantida |

---

## Etapa 2 — Transformação dos Dados

### Decisões de Encoding

| Variável | Estratégia | Justificativa |
|---|---|---|
| Idade | **REMOVIDA** | r=0,013 com log(salário), sem poder preditivo |
| Escolaridade | Ordinal (1–7) | Progressão salarial clara: Fundamental → Doutorado |
| Segunda/Terceira Língua | Binário (0/1) | O idioma específico não importa, apenas ter ou não |
| Região | One-hot (drop_first) | Sem ordem natural entre regiões |
| Profissão | One-hot (drop_first) | 19 profissões sem hierarquia |
| Anos_Experiencia | StandardScaler | Única contínua restante; GLM sensível à escala |

### Resultado

```
Dataset bruto          → 15.000 registros × 8 colunas
Após limpeza           → 14.503 registros × 8 colunas
Após corte de outliers → 14.357 registros × 8 colunas
Após encoding          → 14.357 registros × 26 features + alvo
```

---

## Etapa 3 — Modelagem

### Por que GLM?

- Salário é contínuo, positivo e com distribuição assimétrica à direita
- A variância cresce com a média (padrão da família Tweedie/Gamma)
- O link logarítmico garante predições sempre positivas
- Coeficientes interpretáveis como efeitos multiplicativos no salário

### Família Tweedie — Parâmetro `power`

| power (p) | Distribuição | Aplicação típica |
|---|---|---|
| p = 0 | Gaussiana | Regressão linear clássica |
| p = 1 | Poisson | Contagens |
| **p = 1,5** | **Compound Poisson-Gamma** | **Valores monetários (VENCEDOR)** |
| p = 2 | Gamma | Valores positivos com cauda direita |

### Resultados Comparativos

| Modelo | R² Treino | R² Teste | MAE (R$) | RMSE (R$) | MAPE | CV-R² |
|---|---|---|---|---|---|---|
| GLM Gamma (p=2) | 0,229 | 0,291 | R$ 4.002 | R$ 5.942 | 47,8% | 0,242±0,037 |
| **GLM Tweedie (p=1,5)** | **0,502** | **0,669** | **R$ 1.706** | **R$ 4.063** | **17,6%** | **0,535±0,086** |
| Random Forest | 0,548 | 0,623 | R$ 2.136 | R$ 4.333 | 24,0% | 0,502±0,085 |

> ✅ O GLM Tweedie venceu em todas as métricas. A escolha correta da família estatística superou o modelo não-linear mais complexo.

### Interpretação dos Coeficientes (GLM Gamma)

Com link logarítmico: `exp(β)` = fator multiplicativo no salário.

| Coeficiente | β | exp(β) | Interpretação |
|---|---|---|---|
| Escolaridade (+1 nível) | +0,125 | ×1,134 | Cada nível a mais = +13,4% no salário |
| Anos Experiência | +0,116 | ×1,123 | +1 desvio-padrão = +12,3% |
| Região: Sudeste | +0,045 | ×1,046 | +4,6% vs. Centro-Oeste |
| Profissão: Médico | +0,051 | ×1,053 | +5,3% vs. referência |
| Profissão: Engenheiro | +0,056 | ×1,058 | +5,8% vs. referência |
| Profissão: Motorista | -0,053 | ×0,949 | -5,1% vs. referência |
| Profissão: Vendedor | -0,047 | ×0,954 | -4,6% vs. referência |

---

## Interface Web (Streamlit)

O app possui dois painéis:

- **Esquerdo (formulário):** Anos de experiência, escolaridade, profissão, região, línguas → botão Prever
- **Direito (resultado):** Salário previsto, nome do modelo, intervalo ±30% (baseado no MAPE)
- **4 abas de EDA:** Escolaridade · Experiência & Idade · Profissão & Região · Correlações

> ⚠️ O campo `Idade` foi removido do formulário intencionalmente — não entra no modelo.

---

## Critérios de Avaliação

| Critério | Entregue | Peso |
|---|---|---|
| Análise Exploratória | 12 gráficos, correlações, decisão de exclusão de variável | 10% |
| Transformação | Encoding ordinal, binário, dummies, scaler, artefatos pkl | 10% |
| Modelos testados | GLM Gamma, GLM Tweedie, Random Forest | 10% |
| Métrica de teste | R²=0,669 · MAE=R$1.706 · MAPE=17,6% | 10% |
| Interface Web | Streamlit com formulário, resultado e EDA | 10% |
| Organização Git | Scripts separados, .gitignore, estrutura limpa | 10% |

---

##  Dependências

```
pandas>=2.0.0
numpy>=1.26.0
scikit-learn>=1.4.0
statsmodels>=0.14.0
matplotlib>=3.8.0
seaborn>=0.13.0
streamlit>=1.32.0
pillow>=10.0.0
scipy>=1.12.0
```

---

> **Aviso:** Projeto acadêmico. Os valores previstos são estimativas experimentais e não devem ser usados como referência profissional.
