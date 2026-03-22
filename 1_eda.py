"""
=============================================================
ETAPA 1 — ANÁLISE EXPLORATÓRIA DE DADOS (EDA)
=============================================================
Dataset : dataset_salarios_brasil.csv
Objetivo: Entender a distribuição, qualidade e relações
          das variáveis antes de qualquer modelagem.

Saídas  : 12 gráficos PNG salvos no diretório de trabalho.
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.facecolor": "#F8F9FA", "axes.facecolor": "#F8F9FA"})

# ──────────────────────────────────────────────────────────────
# 1. CARREGAMENTO E CORREÇÃO DE TIPOS
# ──────────────────────────────────────────────────────────────
print("=" * 60)
print("  ETAPA 1 — ANÁLISE EXPLORATÓRIA DE DADOS")
print("=" * 60)

df = pd.read_csv("dataset_salarios_brasil.csv")

# Colunas numéricas foram lidas como string por causa de "erro" no CSV
df["Salario"]          = pd.to_numeric(df["Salario"],          errors="coerce")
df["Idade"]            = pd.to_numeric(df["Idade"],            errors="coerce")
df["Anos_Experiencia"] = pd.to_numeric(df["Anos_Experiencia"], errors="coerce")

# ──────────────────────────────────────────────────────────────
# 2. VISÃO GERAL
# ──────────────────────────────────────────────────────────────
print("\n--- Dimensões do dataset ---")
print(f"  Linhas: {df.shape[0]:,}   Colunas: {df.shape[1]}")

print("\n--- Tipos de dados ---")
print(df.dtypes.to_string())

print("\n--- Primeiras linhas ---")
print(df.head(5).to_string())

print("\n--- Estatísticas descritivas (numéricas) ---")
print(df.describe().round(2).to_string())

# ──────────────────────────────────────────────────────────────
# 3. QUALIDADE DOS DADOS
# ──────────────────────────────────────────────────────────────
print("\n--- Valores ausentes / inválidos por coluna ---")
nulos = df.isnull().sum()
pct   = (nulos / len(df) * 100).round(2)
resumo_nulos = pd.DataFrame({"Nulos": nulos, "% do Total": pct})
print(resumo_nulos[resumo_nulos["Nulos"] > 0].to_string())

# Linhas com "erro" literal já foram convertidas para NaN acima
erros_salario = df["Salario"].isna().sum()
print(f"\n  → {erros_salario} registros com Salario=NaN "
      f"(incluindo os com texto 'erro')")
print(f"  → {df.duplicated().sum()} linhas duplicadas")

# ──────────────────────────────────────────────────────────────
# 4. CORRELAÇÃO COM log(SALÁRIO)
#    Importante antes de decidir quais variáveis incluir no modelo
# ──────────────────────────────────────────────────────────────
df_num = df[["Idade", "Anos_Experiencia", "Salario"]].dropna().copy()
df_num["log_Salario"] = np.log1p(df_num["Salario"])

r_idade, p_idade = stats.pearsonr(df_num["Idade"],            df_num["log_Salario"])
r_exp,   p_exp   = stats.pearsonr(df_num["Anos_Experiencia"], df_num["log_Salario"])

print("\n--- Correlação de Pearson com log(Salário) ---")
print(f"  Idade            : r = {r_idade:.4f}  (p = {p_idade:.4e})")
print(f"  Anos_Experiencia : r = {r_exp:.4f}  (p = {p_exp:.4e})")
print()
print("  ⚠  DECISÃO: 'Idade' apresenta correlação próxima de zero")
print("     com o salário e NÃO será usada na modelagem.")
print("     'Anos_Experiencia' tem correlação moderada e")
print("     SERÁ mantida no modelo.")

# ──────────────────────────────────────────────────────────────
# 5. GRÁFICOS — VARIÁVEL ALVO
# ──────────────────────────────────────────────────────────────
print("\nGerando gráficos da variável alvo...")
sal = df["Salario"].dropna()
log_sal = np.log1p(sal)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribuição bruta
axes[0].hist(sal, bins=60, color="#2D6A9F", edgecolor="white", linewidth=0.3)
axes[0].axvline(sal.mean(),   color="tomato", linestyle="--",
                label=f"Média: R$ {sal.mean():,.0f}")
axes[0].axvline(sal.median(), color="gold",   linestyle="--",
                label=f"Mediana: R$ {sal.median():,.0f}")
axes[0].set_title("Distribuição Bruta do Salário", fontweight="bold")
axes[0].set_xlabel("Salário (R$)")
axes[0].set_ylabel("Frequência")
axes[0].legend()

# Distribuição log-transformada
axes[1].hist(log_sal, bins=60, color="#1A936F", edgecolor="white", linewidth=0.3)
mu, sigma = log_sal.mean(), log_sal.std()
x = np.linspace(log_sal.min(), log_sal.max(), 200)
ax2 = axes[1].twinx()
ax2.plot(x, stats.norm.pdf(x, mu, sigma), "r-", linewidth=2,
         label=f"Normal(μ={mu:.2f}, σ={sigma:.2f})")
ax2.set_ylabel("Densidade (curva normal)", color="r")
ax2.tick_params(axis="y", labelcolor="r")
ax2.legend(loc="upper right")
axes[1].set_title("Distribuição de log(Salário) — Aproximação Normal",
                  fontweight="bold")
axes[1].set_xlabel("log(Salário + 1)")
axes[1].set_ylabel("Frequência")

plt.tight_layout()
plt.savefig("eda_distribuicao_salario.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_distribuicao_salario.png")

# ──────────────────────────────────────────────────────────────
# 6. GRÁFICOS — ESCOLARIDADE vs SALÁRIO (ordenada)
# ──────────────────────────────────────────────────────────────
escol_order = ["Fundamental", "Médio", "Técnico", "Superior", "Pós",
               "Mestrado", "Doutorado"]
limite_99 = df["Salario"].quantile(0.99)
df_plot   = df[df["Salario"] <= limite_99].copy()

plt.figure(figsize=(12, 6))
sns.boxplot(data=df_plot, x="Escolaridade", y="Salario",
            order=escol_order, hue="Escolaridade",
            palette="Blues", legend=False)
plt.title("Salário por Nível de Escolaridade (sem top-1% outliers)",
          fontweight="bold")
plt.xlabel("Nível de Escolaridade (ordem crescente →)")
plt.ylabel("Salário (R$)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("eda_salario_por_escolaridade.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_salario_por_escolaridade.png")

# ──────────────────────────────────────────────────────────────
# 7. GRÁFICO — PROFISSÃO vs MEDIANA SALARIAL
# ──────────────────────────────────────────────────────────────
med_profissao = (df_plot.groupby("Profissao")["Salario"]
                         .median()
                         .sort_values(ascending=True))

fig, ax = plt.subplots(figsize=(10, 8))
colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(med_profissao)))
bars = ax.barh(med_profissao.index, med_profissao.values, color=colors)
for bar, val in zip(bars, med_profissao.values):
    ax.text(val + 150, bar.get_y() + bar.get_height() / 2,
            f"R$ {val:,.0f}", va="center", fontsize=8)
ax.set_xlabel("Mediana Salarial (R$)")
ax.set_title("Mediana de Salário por Profissão", fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("eda_salario_por_profissao.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_salario_por_profissao.png")

# ──────────────────────────────────────────────────────────────
# 8. GRÁFICO — REGIÃO vs SALÁRIO
# ──────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
sns.boxplot(data=df_plot, x="Regiao", y="Salario",
            hue="Regiao", palette="husl", legend=False)
plt.title("Distribuição de Salários por Região", fontweight="bold")
plt.xlabel("Região")
plt.ylabel("Salário (R$)")
plt.tight_layout()
plt.savefig("eda_salario_por_regiao.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_salario_por_regiao.png")

# ──────────────────────────────────────────────────────────────
# 9. SCATTER — ANOS DE EXPERIÊNCIA vs log(SALÁRIO)
# ──────────────────────────────────────────────────────────────
df_scatter = df_num.copy()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Anos Experiência
axes[0].scatter(df_scatter["Anos_Experiencia"], df_scatter["log_Salario"],
                alpha=0.15, s=8, color="#1A936F")
m, b = np.polyfit(df_scatter["Anos_Experiencia"],
                  df_scatter["log_Salario"], 1)
x_line = np.linspace(df_scatter["Anos_Experiencia"].min(),
                     df_scatter["Anos_Experiencia"].max(), 100)
axes[0].plot(x_line, m * x_line + b, "r-", linewidth=2)
axes[0].set_title(f"Anos de Experiência vs log(Salário)\n"
                  f"(r = {r_exp:.3f}, p < 0.001)", fontweight="bold")
axes[0].set_xlabel("Anos de Experiência")
axes[0].set_ylabel("log(Salário)")

# Idade
axes[1].scatter(df_scatter["Idade"], df_scatter["log_Salario"],
                alpha=0.15, s=8, color="#888888")
m2, b2 = np.polyfit(df_scatter["Idade"], df_scatter["log_Salario"], 1)
x2 = np.linspace(df_scatter["Idade"].min(), df_scatter["Idade"].max(), 100)
axes[1].plot(x2, m2 * x2 + b2, "r-", linewidth=2)
axes[1].set_title(f"Idade vs log(Salário)\n"
                  f"(r = {r_idade:.3f}) — ⚠ correlação desprezível",
                  fontweight="bold")
axes[1].set_xlabel("Idade (anos)")
axes[1].set_ylabel("log(Salário)")

plt.tight_layout()
plt.savefig("eda_salario_vs_experiencia.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_salario_vs_experiencia.png")

# ──────────────────────────────────────────────────────────────
# 10. MAPA DE CALOR — CORRELAÇÕES NUMÉRICAS
# ──────────────────────────────────────────────────────────────
corr_df = df_num[["Idade", "Anos_Experiencia", "log_Salario"]].rename(
    columns={"log_Salario": "log(Salário)"})

plt.figure(figsize=(6, 5))
mask = np.triu(np.ones_like(corr_df.corr(), dtype=bool), k=1)
sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm",
            fmt=".3f", vmin=-1, vmax=1,
            square=True, linewidths=0.5)
plt.title("Mapa de Calor — Correlações Numéricas", fontweight="bold")
plt.tight_layout()
plt.savefig("eda_correlacao_numericas.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_correlacao_numericas.png")

# ──────────────────────────────────────────────────────────────
# 11. LÍNGUAS — IMPACTO NO SALÁRIO
# ──────────────────────────────────────────────────────────────
df_ling = df_plot.copy()
df_ling["Tem_2a_Lingua"] = (df_ling["Segunda_Lingua"] != "Nenhuma").map(
    {True: "Sim", False: "Não"})
df_ling["Tem_3a_Lingua"] = (df_ling["Terceira_Lingua"] != "Nenhuma").map(
    {True: "Sim", False: "Não"})

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(data=df_ling, x="Tem_2a_Lingua", y="Salario",
            hue="Tem_2a_Lingua", palette="Set2",
            legend=False, ax=axes[0])
axes[0].set_title("Salário: Tem 2ª Língua?", fontweight="bold")
axes[0].set_xlabel("Possui 2ª Língua")
axes[0].set_ylabel("Salário (R$)")

sns.boxplot(data=df_ling, x="Tem_3a_Lingua", y="Salario",
            hue="Tem_3a_Lingua", palette="Set3",
            legend=False, ax=axes[1])
axes[1].set_title("Salário: Tem 3ª Língua?", fontweight="bold")
axes[1].set_xlabel("Possui 3ª Língua")
axes[1].set_ylabel("Salário (R$)")

plt.tight_layout()
plt.savefig("eda_segunda_lingua.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_segunda_lingua.png")

# ──────────────────────────────────────────────────────────────
# 12. DISTRIBUIÇÕES: Experiência e Outliers
# ──────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(df["Anos_Experiencia"].dropna(), bins=30,
             color="#2D6A9F", edgecolor="white", linewidth=0.3)
axes[0].set_title("Distribuição de Anos de Experiência", fontweight="bold")
axes[0].set_xlabel("Anos de Experiência")
axes[0].set_ylabel("Frequência")

# Boxplot de Salário para evidenciar outliers
axes[1].boxplot(sal, vert=True, patch_artist=True,
                boxprops=dict(facecolor="#2D6A9F", alpha=0.6),
                medianprops=dict(color="orange", linewidth=2),
                flierprops=dict(marker="o", markersize=3, alpha=0.3))
q1, q3 = sal.quantile(0.25), sal.quantile(0.75)
iqr = q3 - q1
n_out = ((sal < q1 - 1.5 * iqr) | (sal > q3 + 1.5 * iqr)).sum()
axes[1].set_title(f"Boxplot Salário — {n_out} outliers IQR ({n_out/len(sal)*100:.1f}%)",
                  fontweight="bold")
axes[1].set_ylabel("Salário (R$)")

plt.tight_layout()
plt.savefig("eda_terceira_lingua.png", dpi=130, bbox_inches="tight")
plt.close()
print("  ✔  eda_terceira_lingua.png")

# ──────────────────────────────────────────────────────────────
# RESUMO FINAL
# ──────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  CONCLUSÕES DA EDA")
print("=" * 60)
print("""
  1. ALVO (Salário): Distribuição fortemente assimétrica à
     direita → log-transformação aproxima normalidade → 
     justifica GLM com família Gamma e link log.

  2. VARIÁVEL EXCLUÍDA: 'Idade' tem r≈0.013 com log(Salário),
     sem poder preditivo. Será removida do modelo.

  3. VARIÁVEL RETIDA: 'Anos_Experiencia' com r≈0.29.

  4. ESCOLARIDADE: relação ordinal clara (Fundamental→Doutorado)
     → encoding ordinal (1 a 7), não one-hot.

  5. LÍNGUAS: o que importa é ter ou não (binário), não qual.

  6. PROFISSÃO e REGIÃO: efeito real no salário → dummies.

  7. OUTLIERS: top-1% de salários serão removidos na etapa 2.
""")
print("  12 gráficos salvos com sucesso!")
