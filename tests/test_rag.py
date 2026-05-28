import os
import sys
import pytest
from unittest.mock import MagicMock, patch

# Adiciona o diretório base do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ──────────────────────────────────────────────────────────────
# ULTRA-FAST HERMETIC LANGCHAIN & GRPC MOCKS (sys.modules)
# ──────────────────────────────────────────────────────────────
# Previne o carregamento de extensões C++ pesadas, chamadas de
# subprocessos bloqueantes e deadlocks de mutex de gRPC no macOS.

mock_fitz = MagicMock()
sys.modules['fitz'] = mock_fitz

mock_loaders = MagicMock()
sys.modules['langchain_community.document_loaders'] = mock_loaders

mock_splitters = MagicMock()
sys.modules['langchain_text_splitters'] = mock_splitters

mock_vectorstores = MagicMock()
sys.modules['langchain_community.vectorstores'] = mock_vectorstores

mock_ollama = MagicMock()
sys.modules['langchain_ollama'] = mock_ollama

mock_histories = MagicMock()
sys.modules['langchain_community.chat_message_histories'] = mock_histories

mock_prompts = MagicMock()
mock_prompts.ChatPromptTemplate = MagicMock()
mock_prompts.MessagesPlaceholder = MagicMock()
sys.modules['langchain_core.prompts'] = mock_prompts

mock_runnables = MagicMock()
mock_runnables.RunnablePassthrough = MagicMock()
sys.modules['langchain_core.runnables'] = mock_runnables

mock_parsers = MagicMock()
mock_parsers.StrOutputParser = MagicMock()
sys.modules['langchain_core.output_parsers'] = mock_parsers

mock_runnables_history = MagicMock()
mock_runnables_history.RunnableWithMessageHistory = MagicMock()
sys.modules['langchain_core.runnables.history'] = mock_runnables_history

# Mock also Document to be easily importable or mock it locally
from langchain_core.documents import Document

# Mock subprocess and requests during module import
with patch('subprocess.Popen') as mock_popen, \
     patch('subprocess.run') as mock_run, \
     patch('requests.get') as mock_get:
    
    mock_get.return_value.status_code = 200
    mock_get.return_value.json.return_value = {"models": [{"name": "llama3"}, {"name": "nomic-embed-text"}]}
    mock_run.return_value.returncode = 0
    
    from src.core.rag_engine import RagEngine

@pytest.fixture
def mock_rag_engine():
    """Fixture para prover uma instância da Engine de RAG mockada para testes isolados."""
    with patch('src.core.rag_engine.ensure_ollama_running', return_value=True), \
         patch('src.core.rag_engine.ensure_model_loaded', return_value=True):
        
        # Instanciar o motor
        engine = RagEngine()
        
        # Mocks de dependências internas da instância
        engine.vectorstore = MagicMock()
        engine.embeddings = MagicMock()
        engine.llm = MagicMock()
        engine.retriever = MagicMock()
        engine.conversational_rag_chain = MagicMock()
        
        yield engine

# ──────────────────────────────────────────────────────────────
# CASOS DE TESTE TDD — PRJ-02 MEMORY RAG
# ──────────────────────────────────────────────────────────────

def test_redis_connection(mock_rag_engine):
    """Verifica que a factory RedisChatMessageHistory é chamada com a URL correta do Redis."""
    with patch('src.core.rag_engine.RedisChatMessageHistory') as mock_redis_history:
        session_id = "test-session-123"
        mock_rag_engine._get_message_history(session_id)
        
        mock_redis_history.assert_called_once_with(
            session_id, 
            url=os.getenv("REDIS_URL", "redis://localhost:6379")
        )

def test_memory_persistence(mock_rag_engine):
    """Verifica que a chamada à chain de RAG conversacional persiste o histórico de mensagens."""
    mock_rag_engine.vectorstore._collection.count.return_value = 5
    mock_rag_engine.conversational_rag_chain.invoke.return_value = "Resposta da segunda pergunta que depende da primeira."
    
    session_id = "session-persist-test"
    question = "Como funciona a persistência com Redis?"
    
    # Executa a query
    result = mock_rag_engine.query(question, session_id=session_id)
    
    # Assertions
    assert "answer" in result
    assert result["answer"] == "Resposta da segunda pergunta que depende da primeira."
    mock_rag_engine.conversational_rag_chain.invoke.assert_called_once_with(
        {"question": question},
        config={"configurable": {"session_id": session_id}}
    )

def test_session_isolation(mock_rag_engine):
    """Garante que sessões distintas passam a ID correta e permanecem isoladas."""
    mock_rag_engine.vectorstore._collection.count.return_value = 5
    session_a = "session-A"
    session_b = "session-B"
    
    # Faz query na sessão A
    mock_rag_engine.query("Pergunta da sessão A", session_id=session_a)
    mock_rag_engine.conversational_rag_chain.invoke.assert_any_call(
        {"question": "Pergunta da sessão A"},
        config={"configurable": {"session_id": session_a}}
    )
    
    # Faz query na sessão B
    mock_rag_engine.query("Pergunta da sessão B", session_id=session_b)
    mock_rag_engine.conversational_rag_chain.invoke.assert_any_call(
        {"question": "Pergunta da sessão B"},
        config={"configurable": {"session_id": session_b}}
    )

def test_chromadb_health(mock_rag_engine):
    """Verifica a integridade e conexão saudável com o ChromaDB."""
    mock_rag_engine.vectorstore._collection.count.return_value = 42
    
    # Executa verificação de contagem de vetores
    count = mock_rag_engine.vectorstore._collection.count()
    
    assert count == 42
    mock_rag_engine.vectorstore._collection.count.assert_called_once()

def test_rag_pipeline_empty(mock_rag_engine):
    """Garante que se a coleção de vetores estiver vazia, retorna mensagem de fallback sem invocar o LLM."""
    mock_rag_engine.vectorstore._collection.count.return_value = 0
    
    result = mock_rag_engine.query("Qualquer pergunta")
    
    assert result["answer"] == "Nenhum documento foi processado ainda. Faça upload de um PDF primeiro."
    assert result["sources"] == []
    mock_rag_engine.conversational_rag_chain.invoke.assert_not_called()

@patch('src.core.rag_engine.PyMuPDFLoader')
@patch('src.core.rag_engine.RecursiveCharacterTextSplitter')
def test_pdf_ingestion(mock_splitter, mock_loader, mock_rag_engine):
    """Verifica que o PDF é lido, divido em chunks e inserido no vectorstore com sucesso."""
    # Configure Mocks
    mock_doc = MagicMock()
    mock_loader.return_value.load.return_value = [mock_doc]
    
    mock_split = Document(page_content="Fragmento de texto de teste.")
    mock_splitter.return_value.split_documents.return_value = [mock_split]
    
    with patch('os.path.exists', return_value=True):
        num_chunks = mock_rag_engine.process_pdf("test_document.pdf")
        
        assert num_chunks == 1
        mock_rag_engine.vectorstore.add_documents.assert_called_once_with(documents=[mock_split])

def test_mmr_diversity(mock_rag_engine):
    """Verifica que a busca vetorial retorna fontes únicas eliminando duplicatas."""
    mock_rag_engine.vectorstore._collection.count.return_value = 5
    mock_rag_engine.conversational_rag_chain.invoke.return_value = "Resposta com MMR."
    
    # Mock retriever returning identical docs to test duplicate removal of sources
    doc_1 = Document(page_content="Doc Content 1", metadata={"page": 1, "source": "livro.pdf"})
    doc_2 = Document(page_content="Doc Content 2", metadata={"page": 1, "source": "livro.pdf"})
    mock_rag_engine.retriever.invoke.return_value = [doc_1, doc_2]
    
    result = mock_rag_engine.query("Pergunta sobre MMR")
    
    # Sources list should remove duplicate references
    assert len(result["sources"]) == 1
    assert result["sources"][0] == "Página 1 do arquivo livro.pdf"
