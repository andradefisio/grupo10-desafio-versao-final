# Previsor de Salários — Brasil

Aplica Machine Learning para prever salários no mercado brasileiro com base no perfil profissional.

---

## Quick Start
Obs.: No meu ambiente de desenvolvimento, dada a estrutura de arquivos, após ativar o ambiente virtual ( .venv\Scripts\Activate.ps1 ), é necessário entrar na pasta rodando no terminal "cd grupo10-desafio-versao-final". Se chamar a aplicação com "Streamlit run app.py" ela não roda. É necessário estar nesta pasta. 

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar a aplicação
```bash
streamlit run app.py
```

Abra o navegador em: **http://localhost:8501**

### 3. Usar o previsor
- Preencha o perfil profissional (experiência, escolaridade, profissão, região, idiomas)
- Clique em **"🔮 Prever Salário"**
- Visualize a previsão e os gráficos de análise

---

## Primeira Execução

Se for a **primeira vez**, execute os scripts de processamento na ordem:

```bash
python 1_eda.py          # Análise exploratória
python 2_transformacao.py # Limpeza dos dados
python 3_modelagem.py     # Treina o modelo
```

Após isso, use `streamlit run app.py` normalmente.

---

## ⚠️ Importante

- Projeto acadêmico para fins de estudo
- Previsões são estimativas experimentais
- Não use como referência profissional oficial
