from pydantic import BaseModel, Field
from typing import List, Optional

class QueryRequest(BaseModel):
    question: str = Field(..., description="A pergunta a ser feita para o RAG.")
    session_id: str = Field("default_session", description="O ID da sessão no Redis para resgatar a memória da conversa.")
    
class QueryResponse(BaseModel):
    answer: str = Field(..., description="A resposta gerada pelo LLM.")
    sources: Optional[List[str]] = Field(default=[], description="Fontes utilizadas para a resposta.")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Status da API, ex: online")
    llm: str = Field(..., description="Status do motor LLM")
    vectordb: str = Field(..., description="Status do motor de banco de dados vetorial")
