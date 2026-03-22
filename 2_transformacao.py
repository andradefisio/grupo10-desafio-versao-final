"""
=============================================================
ETAPA 2 — TRANSFORMAÇÃO E PREPARAÇÃO DOS DADOS
=============================================================
Entrada : dataset_salarios_brasil.csv
Saídas  :
  • dataset_limpo.csv      — dados limpos (para o Streamlit)
  • X_modelagem.csv        — features prontas para o modelo
  • y_modelagem.csv        — variável alvo (Salario)
  • scaler.pkl             — StandardScaler ajustado
  • colunas_modelo.pkl     — lista ordenada de colunas
  • escolaridade_map.pkl   — mapeamento ordinal de Escolaridade

DECISÕES fundamentadas na EDA:
  • 'Idade' removida (r ≈ 0.013 com log-salário, sem poder preditivo)
  • 'Escolaridade' → encoding ordinal (tem ordem natural clara)
  • Línguas → binárias: 1=fala alguma língua, 0=Nenhuma
  • 'Regiao' e 'Profissao' → dummies (one-hot, drop_first=True)
  • Outliers do top-1% de salário removidos (evitar distorção)
  • StandardScaler aplicado apenas em 'Anos_Experiencia'
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("  ETAPA 2 — TRANSFORMAÇÃO DE DADOS")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. CARREGAMENTO
# ──────────────────────────────────────────────────────────────
print("\n[1/6] Carregando dados brutos...")
df = pd.read_csv("dataset_salarios_brasil.csv")
print(f"      {df.shape[0]:,} linhas × {df.shape[1]} colunas carregadas.")

# ──────────────────────────────────────────────────────────────
# 2. CORREÇÃO DE TIPOS (valores "erro" viram NaN)
# ──────────────────────────────────────────────────────────────
print("[2/6] Corrigindo tipos de dados...")
df["Salario"]          = pd.to_numeric(df["Salario"],          errors="coerce")
df["Idade"]            = pd.to_numeric(df["Idade"],            errors="coerce")
df["Anos_Experiencia"] = pd.to_numeric(df["Anos_Experiencia"], errors="coerce")

# ──────────────────────────────────────────────────────────────
# 3. REMOÇÃO DE NULOS E DUPLICATAS
# ──────────────────────────────────────────────────────────────
print("[3/6] Removendo registros inválidos e duplicados...")
linhas_antes = len(df)

df = df.drop_duplicates()
df = df.dropna(subset=["Salario"])          # Salario NaN → inutilizável
df["Anos_Experiencia"] = df["Anos_Experiencia"].fillna(
    df["Anos_Experiencia"].median())         # Imputação conservadora

removidos = linhas_antes - len(df)
print(f"      → {removidos} registros removidos/imputados. "
      f"Restam: {len(df):,}")

# ──────────────────────────────────────────────────────────────
# 4. TRATAMENTO DE OUTLIERS — TOP 1% SALARIAL
# ──────────────────────────────────────────────────────────────
print("[4/6] Removendo outliers (top 1% salarial)...")
linhas_antes_out = len(df)
limite_superior  = df["Salario"].quantile(0.99)
df = df[df["Salario"] <= limite_superior].copy()
print(f"      → {linhas_antes_out - len(df)} outliers removidos "
      f"(acima de R$ {limite_superior:,.2f}).")
print(f"      → Dataset limpo: {len(df):,} registros.")

# ──────────────────────────────────────────────────────────────
# SALVAR DATASET LIMPO (usado pelo Streamlit para preencher
# os dropdowns com valores válidos do dataset)
# ──────────────────────────────────────────────────────────────
df.to_csv("dataset_limpo.csv", index=False)
print("      ✔  'dataset_limpo.csv' salvo.")

# ──────────────────────────────────────────────────────────────
# 5. ENGENHARIA DE FEATURES
# ──────────────────────────────────────────────────────────────
print("[5/6] Aplicando encoding e transformações...")

df_feat = df.copy()

# A) REMOVER 'Idade' — correlação desprezível com log(Salário)
df_feat = df_feat.drop(columns=["Idade"])

# B) ESCOLARIDADE — Encoding Ordinal
#    Justificativa: existe progressão salarial clara de
#    Fundamental → Doutorado, então números ordenados são mais
#    informativos do que colunas binárias independentes.
ESCOLARIDADE_MAP = {
    "Fundamental": 1,
    "Médio":       2,
    "Técnico":     3,
    "Superior":    4,
    "Pós":         5,
    "Mestrado":    6,
    "Doutorado":   7,
}
df_feat["Escolaridade"] = df_feat["Escolaridade"].map(ESCOLARIDADE_MAP)

# C) LÍNGUAS — Binárias
#    Justificativa: o que importa é ter competência em outro idioma,
#    não qual idioma específico (distribuição salarial similar entre
#    os idiomas na EDA).
df_feat["Tem_2a_Lingua"] = (df_feat["Segunda_Lingua"] != "Nenhuma").astype(int)
df_feat["Tem_3a_Lingua"] = (df_feat["Terceira_Lingua"] != "Nenhuma").astype(int)
df_feat = df_feat.drop(columns=["Segunda_Lingua", "Terceira_Lingua"])

# D) REGIÃO e PROFISSÃO — One-Hot Encoding
#    Sem ordem natural → dummies (drop_first evita multicolinearidade perfeita)
df_feat = pd.get_dummies(df_feat, columns=["Regiao", "Profissao"],
                         drop_first=True)

# ──────────────────────────────────────────────────────────────
# 6. SEPARAR X e y — ESCALAR VARIÁVEL CONTÍNUA
# ──────────────────────────────────────────────────────────────
X = df_feat.drop(columns=["Salario"])
y = df_feat["Salario"]

# StandardScaler em 'Anos_Experiencia' (única contínua restante)
scaler = StandardScaler()
X[["Anos_Experiencia"]] = scaler.fit_transform(X[["Anos_Experiencia"]])

# Garantir que bool → int (get_dummies gera bool no pandas ≥ 1.5)
bool_cols = X.select_dtypes(include="bool").columns
X[bool_cols] = X[bool_cols].astype(int)

print(f"      → Features finais: {X.shape[1]} colunas")
print(f"      → Colunas: {list(X.columns)}")

# ──────────────────────────────────────────────────────────────
# 7. SALVAR ARTEFATOS
# ──────────────────────────────────────────────────────────────
print("[6/6] Salvando artefatos para modelagem e deploy...")

X.to_csv("X_modelagem.csv", index=False)
y.to_csv("y_modelagem.csv", index=False)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("colunas_modelo.pkl", "wb") as f:
    pickle.dump(list(X.columns), f)

with open("escolaridade_map.pkl", "wb") as f:
    pickle.dump(ESCOLARIDADE_MAP, f)

print()
print("=" * 60)
print("  ETAPA 2 CONCLUÍDA — Artefatos salvos:")
print("=" * 60)
print("""
  • dataset_limpo.csv      → usado pelo Streamlit (dropdowns)
  • X_modelagem.csv        → features para treino dos modelos
  • y_modelagem.csv        → variável alvo (Salario)
  • scaler.pkl             → StandardScaler ajustado
  • colunas_modelo.pkl     → ordem exata das colunas do modelo
  • escolaridade_map.pkl   → mapeamento ordinal de Escolaridade
""")
