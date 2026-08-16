from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import os
import shutil

# Importações dos modelos e engine (local do projeto src/)
from .api.schemas import QueryRequest, QueryResponse, HealthResponse
from .core.rag_engine import rag_engine_instance, get_engine, UPLOADS_DIR
from .core.llm_factory import (
    LLMConfigError,
    PROVIDERS,
    describe_provider,
    list_provider_status,
    modelos_ollama,
    probe_provider,
    resolve_provider,
)


# Inicialização do App FastAPI
app = FastAPI(
    title="RAG Vanilla API (Projeto 01)",
    description="API para testes base da arquitetura RAG Vanilla rodando 100% local com Ollama e ChromaDB.",
    version="1.0.0"
)

# Adicionando CORS para permitir acesso de qualquer interface web (Gradio, Streamlit ou Vue/React)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/providers", tags=["LLM"])
async def providers(probe: bool = False):
    """
    Motores de LLM e o estado de cada um.

    `?probe=true` faz uma chamada real — única forma de detectar conta sem
    crédito ou chave revogada. Também devolve os modelos instalados no Ollama
    para a tela oferecer a lista em vez de campo livre.
    """
    if probe:
        status = []
        for nome in PROVIDERS:
            try:
                status.append(probe_provider(nome))
            except LLMConfigError:
                status.append(describe_provider(nome))
    else:
        status = list_provider_status()

    return {
        "default": resolve_provider(),
        "providers": [vars(s) for s in status],
        "ollama_models": modelos_ollama(),
    }

@app.get("/", tags=["Basic"])
async def root():
    return {"message": "Bem-vindo ao Vanilla RAG API! Acesse /docs para ver o Swagger completo."}


@app.get("/health", response_model=HealthResponse, tags=["Health Operations"])
async def check_health():
    """
    Checa o status do motor do RAG (Ollama conectado, ChromaDB pronto).
    """
    engine_status = rag_engine_instance.status
    if engine_status == "error":
        return HealthResponse(
            status="error",
            llm="Offline ou Erro de Inicialização",
            vectordb="Offline ou Erro de Inicialização"
        )
    else:
        # Se chegou aqui, as instâncias instanciaram sem crashar.
        # Em um uso avançado checaríamos ping no DB e no Ollama.
        return HealthResponse(
            status="online",
            llm="Active",
            vectordb="Active"
        )


@app.post("/upload_pdf", tags=["Document Processing"])
async def upload_document(file: UploadFile = File(...)):
    """
    Faz o upload de um arquivo PDF, extrai seus fragmentos e insere no Vector Database.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são suportados nestes testes Vanilla.")
        
    try:
        # Salvar o arquivo recebido fisicamente na pasta uploads/
        file_location = os.path.join(UPLOADS_DIR, file.filename)
        with open(file_location, "wb+") as file_object:
            shutil.copyfileobj(file.file, file_object)
            
        # Acionar a Engine de RAG para processar e embarcar(embed) no DB
        num_chunks = rag_engine_instance.process_pdf(file_location)
        
        return {
            "info": f"Arquivo '{file.filename}' processado com sucesso.",
            "chunks_gerados": num_chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=QueryResponse, tags=["Interaction"])
async def perform_query(request: QueryRequest):
    """
    Realiza uma requisição na corrente(chain) RAG, buscando contexto no ChromaDB para embasar a resposta do Ollama.
    """
    if rag_engine_instance.status == "error":
        raise HTTPException(status_code=503, detail="RAG Engine não inicializou adequadamente. Verifique se o Ollama está rodando.")
        
    try:
        motor = get_engine(request.provider, request.model) if (request.provider or request.model) else rag_engine_instance
        retorno = motor.query(request.question, request.session_id)
        return QueryResponse(
            answer=retorno["answer"],
            sources=retorno["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno de processamento LLM: {str(e)}")


@app.post("/clear_db", tags=["Maintenance"])
async def clear_database():
    """
    Remove todos os vetores e documentos do ecossistema para resetar o projeto.
    """
    success = rag_engine_instance.clear_database()
    if success:
        return {"message": "Banco de dados e arquivos de upload limpos com sucesso."}
    else:
        raise HTTPException(status_code=500, detail="Erro ao limpar o banco de dados.")


@app.get("/list_docs", tags=["Document Processing"])
async def list_documents():
    """
    Lista todos os nomes de arquivos PDF que foram processados e estão na base.
    """
    docs = rag_engine_instance.list_documents()
    return {"documents": docs}


@app.delete("/remove_doc", tags=["Document Processing"])
async def remove_document(filename: str):
    """
    Remove um arquivo específico do banco de dados vetorial e da pasta de uploads.
    """
    success = rag_engine_instance.remove_document(filename)
    if success:
        return {"message": f"Documento '{filename}' removido com sucesso."}
    else:
        raise HTTPException(status_code=500, detail=f"Erro ao remover o documento '{filename}'.")
