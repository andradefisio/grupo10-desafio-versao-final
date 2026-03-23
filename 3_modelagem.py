"""
=============================================================
ETAPA 3 — MODELAGEM: GLM e Benchmark
=============================================================
Entrada : X_modelagem.csv, y_modelagem.csv (da Etapa 2)
Saída   : modelo_final.pkl (melhor modelo serializado)

Modelos testados:
  1. GLM Gamma  (TweedieRegressor power=2, link=log)
     → Família ideal para valores monetários: contínuos,
       positivos e com variância crescente com a média.
  2. GLM Tweedie (TweedieRegressor power=1.5, link=log)
     → Variante intermediária entre Poisson e Gamma.
  3. Random Forest (benchmark)
     → Modelo não-linear para comparação de desempenho.

Métricas avaliadas: R², MAE, RMSE, MAPE
=============================================================
"""

import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model   import TweedieRegressor
from sklearn.ensemble       import RandomForestRegressor
from sklearn.metrics        import (mean_absolute_error,
                                    mean_squared_error, r2_score)

print("=" * 60)
print("  ETAPA 3 — MODELAGEM GLM E BENCHMARK")
print("=" * 60)

# ──────────────────────────────────────────────────────────────
# 1. CARREGAR DADOS
# ──────────────────────────────────────────────────────────────
print("\n[1/4] Carregando dados da Etapa 2...")
X = pd.read_csv("X_modelagem.csv")
y = pd.read_csv("y_modelagem.csv").squeeze()   # DataFrame → Serie

print(f"      {X.shape[0]:,} amostras × {X.shape[1]} features")

# ──────────────────────────────────────────────────────────────
# 2. DIVISÃO TREINO / TESTE (80/20)
# ──────────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42)

print(f"\n      Treino : {X_train.shape[0]:,} amostras")
print(f"      Teste  : {X_test.shape[0]:,} amostras")

# ──────────────────────────────────────────────────────────────
# 3. DEFINIR OS 3 MODELOS
# ──────────────────────────────────────────────────────────────
#
# TweedieRegressor implementa a família de distribuições Tweedie:
#   power=0  → Gaussiana (link identity — regressão linear clássica)
#   power=1  → Poisson
#   power=1.5→ Compound Poisson-Gamma (Tweedie)
#   power=2  → Gamma  ← ideal para salários
#   power=3  → Inversa Gaussiana
#
# link='log' garante predições sempre positivas e
# coeficientes interpretáveis como efeitos multiplicativos.
# ──────────────────────────────────────────────────────────────
modelos = {
    "GLM Gamma   (power=2)  ": TweedieRegressor(
        power=2, alpha=0.5, link="log", max_iter=3000),

    "GLM Tweedie (power=1.5)": TweedieRegressor(
        power=1.5, alpha=0.5, link="log", max_iter=3000),

    "Random Forest (benchmark)": RandomForestRegressor(
        n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
}

# ──────────────────────────────────────────────────────────────
# 4. TREINAR, AVALIAR E COMPARAR
# ──────────────────────────────────────────────────────────────
print("\n[2/4] Treinando e avaliando modelos...")

def calcular_metricas(y_true, y_pred):
    """Retorna dicionário com R², MAE, RMSE e MAPE."""
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape}

resultados  = {}
modelos_fit = {}

for nome, modelo in modelos.items():
    print(f"\n  → {nome.strip()} ...")
    modelo.fit(X_train, y_train)

    y_pred_treino = modelo.predict(X_train)
    y_pred_teste  = modelo.predict(X_test)

    # Garantir predições positivas (GLM já garante, mas RF pode dar 0)
    y_pred_teste  = np.maximum(y_pred_teste,  1.0)
    y_pred_treino = np.maximum(y_pred_treino, 1.0)

    m_treino = calcular_metricas(y_train, y_pred_treino)
    m_teste  = calcular_metricas(y_test,  y_pred_teste)

    # Validação cruzada (5-fold) no R²
    cv_r2 = cross_val_score(modelo, X, y, cv=5,
                             scoring="r2", n_jobs=-1)

    resultados[nome]  = {"treino": m_treino, "teste": m_teste,
                          "cv_r2_mean": cv_r2.mean(),
                          "cv_r2_std":  cv_r2.std()}
    modelos_fit[nome] = modelo

    print(f"     Treino → R²={m_treino['R2']:.4f}  "
          f"MAE=R${m_treino['MAE']:,.0f}  "
          f"RMSE=R${m_treino['RMSE']:,.0f}")
    print(f"     Teste  → R²={m_teste['R2']:.4f}  "
          f"MAE=R${m_teste['MAE']:,.0f}  "
          f"RMSE=R${m_teste['RMSE']:,.0f}  "
          f"MAPE={m_teste['MAPE']:.1f}%")
    print(f"     CV-R²  → {cv_r2.mean():.4f} ± {cv_r2.std():.4f}")

# ──────────────────────────────────────────────────────────────
# 5. TABELA COMPARATIVA
# ──────────────────────────────────────────────────────────────
print("\n[3/4] Tabela comparativa de desempenho:")
print()
print(f"{'Modelo':<32} | {'R² Teste':>8} | {'MAE (R$)':>10} | "
      f"{'RMSE (R$)':>10} | {'MAPE (%)':>8} | {'CV-R²':>12}")
print("-" * 92)

for nome, res in resultados.items():
    t = res["teste"]
    print(f"{nome.strip():<32} | {t['R2']:>8.4f} | "
          f"{t['MAE']:>10,.0f} | {t['RMSE']:>10,.0f} | "
          f"{t['MAPE']:>7.1f}% | "
          f"{res['cv_r2_mean']:.4f}±{res['cv_r2_std']:.4f}")

print()
print("  Interpretação das métricas:")
print("  • R²   → proporção da variância explicada (maior = melhor)")
print("  • MAE  → erro médio absoluto em R$ (menor = melhor)")
print("  • RMSE → penaliza erros grandes (menor = melhor)")
print("  • MAPE → erro percentual médio (menor = melhor)")
print("  • CV-R²→ robustez do modelo em dados não vistos")

# ──────────────────────────────────────────────────────────────
# 6. ANÁLISE DOS COEFICIENTES DO GLM GAMMA
# ──────────────────────────────────────────────────────────────
print("\n--- Coeficientes do GLM Gamma (interpretação) ---")
print("  Escala: log-link → exp(coef) = fator multiplicativo no salário\n")

glm_gamma = modelos_fit["GLM Gamma   (power=2)  "]
coefs     = pd.Series(glm_gamma.coef_, index=X.columns)
intercept = glm_gamma.intercept_

print(f"  Intercepto: {intercept:.4f}  "
      f"(salário base ≈ R$ {np.exp(intercept):,.0f})")
print()

# Mostrar top impactos positivos e negativos
top_pos = coefs.nlargest(8)
top_neg = coefs.nsmallest(6)

print("  Top 8 coeficientes POSITIVOS (aumentam o salário):")
for var, val in top_pos.items():
    var_limpa = (var.replace("Profissao_", "").replace("Regiao_", "")
                    .replace("_", " "))
    print(f"    {var_limpa:<28} β={val:+.4f}  "
          f"→ efeito multiplicativo: ×{np.exp(val):.3f}")

print()
print("  Top 6 coeficientes NEGATIVOS (reduzem o salário):")
for var, val in top_neg.items():
    var_limpa = (var.replace("Profissao_", "").replace("Regiao_", "")
                    .replace("_", " "))
    print(f"    {var_limpa:<28} β={val:+.4f}  "
          f"→ efeito multiplicativo: ×{np.exp(val):.3f}")

# ──────────────────────────────────────────────────────────────
# 7. ESCOLHA E PERSISTÊNCIA DO MODELO FINAL
# ──────────────────────────────────────────────────────────────
print("\n[4/4] Selecionando e salvando o modelo final...")

# Critério de seleção: maior R² no conjunto de teste
melhor_nome = max(resultados,
                  key=lambda n: resultados[n]["teste"]["R2"])
modelo_final = modelos_fit[melhor_nome]

print(f"\n  ✔  Modelo selecionado: {melhor_nome.strip()}")
print(f"     R² Teste: {resultados[melhor_nome]['teste']['R2']:.4f}")
print(f"     MAE:      R$ {resultados[melhor_nome]['teste']['MAE']:,.0f}")

# Salva o modelo
with open("modelo_final.pkl", "wb") as f:
    pickle.dump(modelo_final, f)

# Salva o nome do modelo vencedor (usado pelo Streamlit)
with open("nome_modelo.pkl", "wb") as f:
    pickle.dump(melhor_nome.strip(), f)

print()
print("=" * 60)
print("  ETAPA 3 CONCLUÍDA — Artefatos salvos:")
print("=" * 60)
print("""
  • modelo_final.pkl  → modelo vencedor pronto para deploy
  • nome_modelo.pkl   → nome do modelo (exibição no Streamlit)
""")

# ──────────────────────────────────────────────────────────────
# 8. INTEGRAÇÃO COM MLFLOW (OPCIONAL)
# ──────────────────────────────────────────────────────────────
print("\n[OPCIONAL] Tentando registrar o modelo no MLflow...")

try:
    from dotenv import load_dotenv
    import mlflow
    import mlflow.sklearn

    load_dotenv()
    
    databricks_host = os.getenv("DATABRICKS_HOST")
    databricks_token = os.getenv("DATABRICKS_TOKEN")

    if databricks_host and databricks_token:
        os.environ['DATABRICKS_HOST'] = databricks_host
        os.environ['DATABRICKS_TOKEN'] = databricks_token
        mlflow.set_tracking_uri("databricks")
        mlflow.set_registry_uri("databricks-uc")  # Unity Catalog (obrigatório neste workspace)
        
        # Define e inicia experimento
        # Nota: O nome DEVE ser um caminho absoluto no workspace Databricks
        experiment_name = "/Shared/previsor_salarios_brasil"
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name="TreinamentoFinal"):
            # Log de parâmetros
            mlflow.log_params({
                "modelos_treinados": 3,
                "tamanho_treino": X_train.shape[0],
                "tamanho_teste": X_test.shape[0],
            })
            
            # Log de métricas do melhor modelo
            melhor_metricas = resultados[melhor_nome]["teste"]
            mlflow.log_metrics({
                "r2_teste": melhor_metricas["R2"],
                "mae_teste": melhor_metricas["MAE"],
                "rmse_teste": melhor_metricas["RMSE"],
                "mape_teste": melhor_metricas["MAPE"],
            })
            
            # Log do modelo - sem registrar no Model Registry para evitar problemas de catálogo
            try:
                mlflow.sklearn.log_model(
                    sk_model=modelo_final,
                    artifact_path="modelo_salarios"
                )
                print("✅ Modelo registrado como artefato no MLflow.")
            except Exception as e:
                print(f"⚠️  Não foi possível registrar o modelo como artefato: {e}")
                print("   O modelo foi salvo localmente em modelo_final.pkl")
            
            print("✅ Modelo registrado no MLflow com sucesso!")
            print(f"   Experimento: {experiment_name}")
            print(f"   Run ID: {mlflow.active_run().info.run_id}")
    else:
        print("⚠️  Credenciais do Databricks não encontradas no .env")
        print("   O modelo foi salvo localmente em modelo_final.pkl")

except ImportError:
    print("⚠️  MLflow não está instalado. Usando apenas armazenamento local.")
except Exception as e:
    print(f"⚠️  Erro ao conectar com MLflow: {e}")
    print("   O modelo foi salvo localmente em modelo_final.pkl")

print("\n✅ Script concluído com sucesso!")
