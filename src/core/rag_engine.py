import os
import subprocess
import time
import fitz  # PyMuPDF
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
import requests
from dotenv import load_dotenv

# Carrega configurações do .env se existir
from .llm_factory import get_llm, resolve_model, resolve_provider

load_dotenv()


def ensure_ollama_running():
    """Verifica se Ollama está rodando, se não, inicia."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=2)
        if response.status_code == 200:
            print("[RAG] Ollama já está rodando.")
            return True
    except Exception:
        pass
    
    print("[RAG] Iniciando Ollama...")
    try:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3)
        
        for _ in range(10):
            try:
                response = requests.get(f"{ollama_host}/api/tags", timeout=2)
                if response.status_code == 200:
                    print("[RAG] Ollama iniciado com sucesso.")
                    return True
            except Exception:
                time.sleep(1)
        
        print("[RAG] Erro ao iniciar Ollama.")
        return False
    except Exception as e:
        print(f"[RAG] Erro ao iniciar Ollama: {e}")
        return False


def ensure_model_loaded(model_name: str):
    """Verifica se o modelo especificado está disponível, se não, fazpull."""
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    
    try:
        response = requests.get(f"{ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", m.get("model", "")) for m in models]
            
            if any(model_name in m for m in model_names):
                print(f"[RAG] Modelo '{model_name}' já disponível.")
                return True
            
            print(f"[RAG] Baixando modelo '{model_name}'...")
            result = subprocess.run(["ollama", "pull", model_name], capture_output=True, text=True)
            if result.returncode == 0:
                print(f"[RAG] Modelo '{model_name}' baixado com sucesso.")
                return True
            else:
                print(f"[RAG] Erro ao baixar modelo: {result.stderr}")
                return False
    except Exception as e:
        print(f"[RAG] Erro ao verificar modelo: {e}")
        return False

# Configurações do RAG vindas das variáveis de ambiente ou com valores padrão de fallback
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text") 
LLM_MODEL = os.getenv("MODEL_NAME", "llama3") 
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Configuração Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

# Descobrindo a raiz do projeto (três níveis acima de src/core/rag_engine.py)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DB_DIR = os.path.join(PROJECT_ROOT, "data", "vector_db")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

class RagEngine:
    def __init__(self, provider: str = None, model: str = None):
        os.makedirs(DB_DIR, exist_ok=True)
        os.makedirs(UPLOADS_DIR, exist_ok=True)
        
        if not ensure_ollama_running():
            print("[RAG] AVISO: Ollama não está disponível. A aplicação pode não funcionar corretamente.")
        
        if not ensure_model_loaded(LLM_MODEL):
            print(f"[RAG] AVISO: Modelo '{LLM_MODEL}' não pôde ser carregado.")
        
        try:
            self.embeddings = OllamaEmbeddings(
                model=EMBEDDING_MODEL,
                base_url=OLLAMA_BASE_URL
            )
            # provider/modelo vêm da fábrica: sem argumento, usa LLM_PROVIDER do .env.
            self.provider = resolve_provider(provider)
            self.model_name = resolve_model(self.provider, model)
            self.llm = get_llm(self.provider, self.model_name, temperature=0.1)
            self.vectorstore = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embeddings
            )
            
            self._setup_chain()
            self.status = "initialized"
        except Exception as e:
            print(f"Erro ao inicializar RAG Engine: {e}")
            self.status = "error"

    def _get_message_history(self, session_id: str) -> RedisChatMessageHistory:
        """Gerencia persistência do histórico via Redis por ID de Sessão."""
        return RedisChatMessageHistory(session_id, url=REDIS_URL)

    def _setup_chain(self):
        """Inicializa e reconecta todos os runnables da Chain Principal do LangChain."""
        # Retriver com MMR para melhor diversidade
        self.retriever = self.vectorstore.as_retriever(
            search_type="mmr", 
            search_kwargs={"k": 5, "fetch_k": 10}
        )
        
        # O prompt agora é focado em mensagens do Chat (ChatPromptTemplate) para suportar memory
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Você é um assistente prestativo que responde perguntas baseado APENAS no contexto fornecido.\n\nContexto extraído:\n{context}\n\nLembre-se das interações anteriores se necessário para entender o usuário, mas a resposta principal DEVE focar no contexto extraído. Se a resposta não estiver no contexto, diga: 'Não encontrei essa informação no documento.' Não invente respostas livres."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
            
        # Corrente do RAG contextualizando
        rag_chain = (
            RunnablePassthrough.assign(
                context=(lambda x: x["question"]) | self.retriever | format_docs
            )
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        # Memória Conversacional do LangChain envelopando o rag_chain
        # Auto-injeta a keyword `history` resgatando do Redis baseando-se por chamadas UUID.
        self.conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            self._get_message_history,
            input_messages_key="question",
            history_messages_key="history"
        )


    def process_pdf(self, file_path: str):
        """Lê um PDF, divide em chunks e salva no VectorDB."""
        if not os.path.exists(file_path):
            raise FileNotFoundError("PDF file not found.")

        # Carregar com PyMuPDFLoader
        loader = PyMuPDFLoader(file_path)
        docs = loader.load()

        # Split em Chunks Menores
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=100
        )
        splits = text_splitter.split_documents(docs)

        # Adicionar ao Banco de Dados (Chroma)
        self.vectorstore.add_documents(documents=splits)
        return len(splits)

    def query(self, question: str, session_id: str = "default_session"):
        """Faz uma pergunta à chain conversacional via memória Redis e recupera as FONTES (Vectors)."""
        if self.vectorstore._collection.count() == 0:
            return {"answer": "Nenhum documento foi processado ainda. Faça upload de um PDF primeiro.", "sources": []}
            
        print(f"[RAG-MEMORY] Buscando contexto na sessão '{session_id}': {question}")
        
        # Invocação envelopada informando dinamicamente a ID de sessão (ChatHistory)
        response = self.conversational_rag_chain.invoke(
            {"question": question},
            config={"configurable": {"session_id": session_id}}
        )
        
        # O self.retriever continua sendo o do VectorDB local para referências
        source_docs = self.retriever.invoke(question)
        sources = [f"Página {doc.metadata.get('page', 'N/A')} do arquivo {os.path.basename(doc.metadata.get('source', 'Desconhecido'))}" for doc in source_docs]
        
        return {
            "answer": response,
            "sources": list(set(sources))
        }

    def clear_database(self):
        """Remove todos os documentos do VectorDB e deleta arquivos de upload."""
        try:
            # Deletar coleção no Chroma
            self.vectorstore.delete_collection()
            # Reinicializar o vectorstore
            self.vectorstore = Chroma(
                persist_directory=DB_DIR, 
                embedding_function=self.embeddings
            )
            
            # Reconecta todo o pipeline de Memory novamente
            self._setup_chain()

            # Limpar pasta de uploads
            for file in os.listdir(UPLOADS_DIR):
                file_path = os.path.join(UPLOADS_DIR, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            
            print("[RAG] Banco de dados e uploads limpos com sucesso.")
            return True
        except Exception as e:
            print(f"[RAG] Erro ao limpar banco de dados: {e}")
            return False

    def list_documents(self):
        """Retorna uma lista de nomes de arquivos extraídos diretamente do VectorDB."""
        try:
            all_data = self.vectorstore._collection.get()
            sources = set()
            for metadata in all_data['metadatas']:
                source_path = metadata.get('source', '')
                if source_path:
                    sources.add(os.path.basename(source_path))
            return sorted(list(sources))
        except Exception as e:
            print(f"[RAG] Erro ao listar documentos do DB: {e}")
            return []

    def remove_document(self, filename: str):
        """Remove um documento específico do VectorDB e apaga o arquivo físico."""
        try:
            full_path = os.path.join(UPLOADS_DIR, filename)
            
            all_data = self.vectorstore._collection.get()
            ids_to_delete = [
                all_data['ids'][i] 
                for i, metadata in enumerate(all_data['metadatas']) 
                if metadata.get('source', '').endswith(filename)
            ]
            
            if ids_to_delete:
                self.vectorstore._collection.delete(ids=ids_to_delete)
                print(f"[RAG] {len(ids_to_delete)} fragmentos removidos do VectorDB.")
            
            if os.path.exists(full_path):
                os.remove(full_path)
            
            return True
        except Exception as e:
            print(f"[RAG] Erro ao remover documento '{filename}': {e}")
            return False

# Instância Singleton Engine
# Instância padrão (provider do .env), mantida para o código existente.
rag_engine_instance = RagEngine()

# Cache por combinação provider+modelo: montar o motor é caro, e trocar de
# motor na tela não pode devolver o motor antigo.
_motores: dict = {}


def get_engine(provider: str = None, model: str = None) -> RagEngine:
    """Motor da combinação pedida. Sem argumentos, usa o padrão do .env."""
    chave = (resolve_provider(provider), resolve_model(resolve_provider(provider), model))
    if chave not in _motores:
        _motores[chave] = RagEngine(provider=chave[0], model=chave[1])
    return _motores[chave]
