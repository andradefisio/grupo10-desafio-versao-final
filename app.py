"""
=============================================================
APP — INTERFACE WEB: Previsor de Salários (Streamlit)
=============================================================
Dependências: rodar 1_eda.py, 2_transformacao.py e
              3_modelagem.py antes de iniciar o app.

Execução local:
  streamlit run app.py
=============================================================
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# ──────────────────────────────────────────────────────────────
# CONFIGURAÇÃO DA PÁGINA
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Previsor de Salários · Brasil",
    page_icon="💼",
    layout="wide",
)

# CSS mínimo para aprimorar a aparência
st.markdown("""
<style>
    .big-metric  { font-size: 2.4rem; font-weight: 700; color: #1A936F; }
    .section-tag { background:#2D6A9F; color:white;
                   padding:4px 12px; border-radius:6px;
                   font-size:0.85rem; font-weight:600; }
    .warn-box    { background:#fff3cd; border-left:4px solid #f0ad4e;
                   padding:10px 14px; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# CARREGAMENTO DE ARTEFATOS (cacheado para não recarregar a cada clique)
# ──────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_recursos():
    with open("modelo_final.pkl",    "rb") as f: modelo      = pickle.load(f)
    with open("colunas_modelo.pkl",  "rb") as f: colunas     = pickle.load(f)
    with open("scaler.pkl",          "rb") as f: scaler      = pickle.load(f)
    with open("escolaridade_map.pkl","rb") as f: escol_map   = pickle.load(f)
    with open("nome_modelo.pkl",     "rb") as f: nome_modelo = pickle.load(f)
    df_limpo = pd.read_csv("dataset_limpo.csv")
    return modelo, colunas, scaler, escol_map, nome_modelo, df_limpo

try:
    modelo, colunas_modelo, scaler, escol_map, nome_modelo, df_raw = \
        carregar_recursos()
except FileNotFoundError as e:
    st.error(
        f"**Arquivo não encontrado:** `{e.filename}`\n\n"
        "Certifique-se de ter executado, nesta ordem:\n"
        "```\npython 2_transformacao.py\npython 3_modelagem.py\n```"
    )
    st.stop()

# ──────────────────────────────────────────────────────────────
# CABEÇALHO
# ──────────────────────────────────────────────────────────────
st.title("💼 Previsor de Salários — Brasil")
st.markdown(
    '<div class="warn-box">⚠️  <strong>Aviso acadêmico:</strong> '
    'Este app foi desenvolvido para fins de estudo em Machine Learning. '
    'Os valores previstos são estimativas experimentais e não devem '
    'ser usados como referência profissional.</div>',
    unsafe_allow_html=True,
)
st.markdown("---")

# ──────────────────────────────────────────────────────────────
# LAYOUT PRINCIPAL: formulário | resultado + EDA
# ──────────────────────────────────────────────────────────────
col_form, col_result = st.columns([0.65, 2.35], gap="large")

# ──────────────── FORMULÁRIO ────────────────────────────────
with col_form:
    st.markdown('<span class="section-tag">📝 Perfil do Profissional</span>',
                unsafe_allow_html=True)
    st.write("")

    experiencia = st.number_input(
        "Anos de Experiência na Área",
        min_value=0, max_value=60, value=5, step=1,
        help="'Idade' não é usada no modelo (correlação ≈ 0 com o salário).")

    # Escolaridade em ordem crescente
    escol_ordem = sorted(escol_map, key=lambda k: escol_map[k])
    escolaridade = st.selectbox("Nível de Escolaridade", escol_ordem)

    profissao = st.selectbox(
        "Profissão",
        sorted(df_raw["Profissao"].unique()))

    regiao = st.selectbox(
        "Região do País",
        sorted(df_raw["Regiao"].unique()))

    segunda_lingua = st.selectbox(
        "Segunda Língua",
        sorted(df_raw["Segunda_Lingua"].unique()),
        help="O modelo usa apenas se possui ou não uma segunda língua.")

    terceira_lingua = st.selectbox(
        "Terceira Língua",
        sorted(df_raw["Terceira_Lingua"].unique()))

    st.write("")
    btn = st.button("🔮 Prever Salário", use_container_width=True,
                    type="primary")

# ──────────────── RESULTADO + ABAS EDA ─────────────────────
with col_result:

    # ── Área de resultado ──
    resultado_placeholder = st.empty()

    if btn:
        # ── Replicar EXATAMENTE a transformação da Etapa 2 ──────────
        # 1. Escolaridade → ordinal
        escol_num = escol_map[escolaridade]

        # 2. Línguas → binárias
        tem_2a = int(segunda_lingua  != "Nenhuma")
        tem_3a = int(terceira_lingua != "Nenhuma")

        # 3. Montar DataFrame base (sem Idade, sem línguas originais)
        input_raw = pd.DataFrame([{
            "Anos_Experiencia": experiencia,
            "Escolaridade":     escol_num,
            "Tem_2a_Lingua":    tem_2a,
            "Tem_3a_Lingua":    tem_3a,
            "Regiao":           regiao,
            "Profissao":        profissao,
        }])

        # 4. One-hot encoding de Regiao e Profissao
        input_enc = pd.get_dummies(input_raw,
                                   columns=["Regiao", "Profissao"])

        # 5. Alinhar colunas com as do modelo (adiciona 0 onde faltar)
        for col in colunas_modelo:
            if col not in input_enc.columns:
                input_enc[col] = 0
        input_enc = input_enc[colunas_modelo]

        # 6. Garantir tipos corretos (bool → int)
        bool_cols = input_enc.select_dtypes(include="bool").columns
        input_enc[bool_cols] = input_enc[bool_cols].astype(int)

        # 7. Escalar Anos_Experiencia (mesmo scaler do treino)
        input_enc[["Anos_Experiencia"]] = scaler.transform(
            input_enc[["Anos_Experiencia"]])

        # 8. Prever
        salario_previsto = float(modelo.predict(input_enc)[0])
        salario_previsto = max(salario_previsto, 0.0)

        # ── Exibir resultado ─────────────────────────────────────────
        with resultado_placeholder.container():
            st.success("✅ Previsão calculada com sucesso!")

            c1, c2, c3 = st.columns(3)
            c1.metric("💰 Salário Previsto",
                      f"R$ {salario_previsto:,.2f}")
            c2.metric("🤖 Modelo Utilizado",
                      nome_modelo.split("(")[0].strip())
            c3.metric("📊 Intervalo ±30%",
                      f"R$ {salario_previsto*0.7:,.0f} – "
                      f"R$ {salario_previsto*1.3:,.0f}")

            st.caption(
                f"Modelo: **{nome_modelo}** · "
                "Nota: o intervalo informal ±30% reflete o MAPE "
                "observado no conjunto de teste.")
            st.divider()
    else:
        with resultado_placeholder.container():
            st.info("👈 Preencha o perfil ao lado e clique em **Prever Salário**.")
            st.divider()

    # ── Abas da EDA ──────────────────────────────────────────────
    st.markdown('<span class="section-tag">📊 Painel de Insights — EDA</span>',
                unsafe_allow_html=True)
    st.write("")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Escolaridade",
        "Experiência & Idade",
        "Profissão & Região",
        "Correlações",
    ])

    def show_img(path, caption=""):
        if os.path.exists(path):
            st.image(Image.open(path), caption=caption,
                     use_container_width=True)
        else:
            st.warning(f"Imagem não encontrada: `{path}`. "
                       "Execute `python 1_eda.py` primeiro.")

    with tab1:
        show_img("eda_salario_por_escolaridade.png",
                 "Escolaridade tem relação ordinal clara com o salário "
                 "→ encoding ordinal (1–7) foi escolhido no lugar de one-hot.")

    with tab2:
        show_img("eda_salario_vs_experiencia.png",
                 "Anos de Experiência (r≈0.29) tem poder preditivo moderado. "
                 "Idade (r≈0.013) foi excluída do modelo por correlação "
                 "desprezível com o salário.")

    with tab3:
        col_a, col_b = st.columns(2)
        with col_a:
            show_img("eda_salario_por_profissao.png",
                     "Médico, Engenheiro e Advogado no topo.")
        with col_b:
            show_img("eda_salario_por_regiao.png",
                     "Sudeste com medianas superiores às demais regiões.")

    with tab4:
        show_img("eda_correlacao_numericas.png",
                 "Correlações lineares entre as variáveis numéricas e "
                 "log(Salário). Ausência de multicolinearidade entre "
                 "Idade e Experiência confirma que a exclusão de Idade "
                 "foi por falta de poder preditivo, não por redundância.")
