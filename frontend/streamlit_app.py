import streamlit as st
import requests

# -------------------------------------------------------------
# Configuração da página Streamlit (Tem visual Moderno e Clean)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Vanilla RAG - Chat",
    page_icon="🤖",
    layout="centered" # Layout centralizado fica mais legível para chat
)

API_URL = "http://127.0.0.1:8000"

# -------------------------------------------------------------
# UI: CASCATA LATERAL (SIDEBAR) - Área de Controles e Upload
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/8649/8649591.png", width=60) # Ícone amigável
    st.title("Configurações")
    st.markdown("Gerencie o cérebro do seu RAG.")
    
    st.divider()
    
    st.header("📄 1. Alimentar o RAG")
    uploaded_file = st.file_uploader("Upload de Conhecimento (PDF)", type=["pdf"], label_visibility="collapsed")
    
    if uploaded_file is not None:
        if st.button("🧠 Processar e Memorizar", use_container_width=True):
            with st.spinner("Extraindo e Vetorizando..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_URL}/upload_pdf", files=files)
                    if response.status_code == 200:
                        st.success("Documento aprendido com sucesso!")
                        st.rerun() # Atualiza a lista
                    else:
                        st.error(f"Erro: {response.text}")
                except Exception as e:
                    st.error("Servidor Offline.")

    # Listagem de Documentos
    st.markdown("---")
    st.subheader("📚 Documentos na Base")
    try:
        l_resp = requests.get(f"{API_URL}/list_docs")
        if l_resp.status_code == 200:
            docs = l_resp.json().get("documents", [])
            if docs:
                for doc in docs:
                    cols = st.columns([0.8, 0.2])
                    cols[0].caption(f"📄 {doc}")
                    if cols[1].button("🗑️", key=f"del_{doc}", help=f"Remover {doc}"):
                        try:
                            d_resp = requests.delete(f"{API_URL}/remove_doc?filename={doc}")
                            if d_resp.status_code == 200:
                                st.success(f"Removido!")
                                st.rerun()
                            else:
                                st.error("Erro ao remover")
                        except Exception:
                            st.error("Erro na API")
            else:
                st.info("Nenhum documento carregado.")
        else:
            st.error("Erro ao listar docs.")
    except Exception:
        st.caption("API Offline para listagem.")

    st.divider()

    st.header("⚙️ 2. Motor de IA")
    if st.button("Verificar Status do Servidor", use_container_width=True):
        with st.spinner("Ping..."):
            try:
                h_resp = requests.get(f"{API_URL}/health")
                if h_resp.status_code == 200:
                    data = h_resp.json()
                    st.success("🟢 Online e Operante")
                    st.caption(f"**LLM:** {data['llm']} | **DB:** {data['vectordb']}")
                else:
                    st.error("🔴 Problema na Engine")
            except Exception:
                st.error("🔴 API Offline")

    st.divider()
    st.header("🧹 3. Manutenção")
    if st.button("Limpar Base de Dados", use_container_width=True, type="secondary", help="Remove todos os documentos e vetores aprendidos."):
        if st.checkbox("Confirma a exclusão total?"):
            with st.spinner("Limpando..."):
                try:
                    c_resp = requests.post(f"{API_URL}/clear_db")
                    if c_resp.status_code == 200:
                        st.success("Base limpa!")
                        st.session_state.messages = [] # Limpa o chat também
                        st.rerun()
                    else:
                        st.error("Erro ao limpar base.")
                except Exception:
                    st.error("API Offline")
        else:
            st.warning("Marque a confirmação acima.")


# -------------------------------------------------------------
# UI: ÁREA PRINCIPAL - Apresentação e Interação
# -------------------------------------------------------------
st.title("🤖 Chat Vanilla RAG")
st.markdown("Esta é a sua interface de testes. Alimente a IA com um PDF pela barra lateral e converse com ela aqui baseada nos dados do arquivo.")

# Inicializar o estado das mensagens
if "messages" not in st.session_state:
    st.session_state.messages = []

# Exibir histórico na tela
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Fontes consultadas", expanded=False):
                for src in message["sources"]:
                    st.caption(f"- {src}")

# Campo de input ancorado no fundo
if prompt := st.chat_input("Pergunte algo sobre o documento armazenado..."):
    # Renderiza a mensagem do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Renderiza a resposta do LLM
    with st.chat_message("assistant"):
        with st.spinner("Refletindo sobre o documento..."):
            try:
                response = requests.post(f"{API_URL}/chat", json={"question": prompt})
                if response.status_code == 200:
                    chat_data = response.json()
                    answer = chat_data["answer"]
                    sources = chat_data.get("sources", [])
                    
                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Fontes consultadas", expanded=False):
                            for src in sources:
                                st.caption(f"- {src}")
                                
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": answer,
                        "sources": sources
                    })
                else:
                    st.error("Ops, ocorreu um erro de conexão com a API.")
            except Exception as e:
                st.error(f"Sistema indisponível: {str(e)}")

