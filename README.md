# 🧠 RAG com Memória Contextual (Redis) — GIULIA AI

> Pipeline RAG local-first integrado com RedisChatMessageHistory e RunnableWithMessageHistory para persistência de histórico de chat e isolamento estrito de sessões multi-usuário.

[![Python Version](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-v0.1.0-FF6F00?style=flat-square)](https://github.com/langchain-ai/langchain)
[![Redis](https://img.shields.io/badge/Redis-v5.0.0-DC382D?style=flat-square&logo=redis&logoColor=white)](https://redis.io/)
[![Giulia AI](https://img.shields.io/badge/GIULIA%20AI-Ecosystem-blueviolet?style=flat-square)](https://github.com/wganalytics)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](LICENSE)

---

## 📋 O que é o projeto

Em sistemas tradicionais de QA baseados em RAG, cada requisição é tratada de forma isolada e estática. Quando um usuário faz uma pergunta de acompanhamento (follow-up query), como *"Por que isso ocorre?"* logo após perguntar *"O que é fotossíntese?"*, a inferência falha por falta de contexto histórico. Manter o histórico do chat de forma escalável e segura sem vazamento de dados entre usuários concorrentes é o problema central abordado neste projeto.

Este repositório fornece uma solução robusta e local-first implementando **Redis** como camada de persistência de memória conversacional (`RedisChatMessageHistory`) acoplada ao ecossistema **LangChain**. Através do uso de `RunnableWithMessageHistory`, o pipeline de RAG gerencia automaticamente a injeção do histórico de conversa e a recuperação de novos contextos no vector store de forma transparente e isolada.

O grande diferencial arquitetural reside no desacoplamento estrito através de identificadores únicos (`session_id`). Isso garante isolamento hermético entre sessões de conversação paralelas, eliminando qualquer risco de contaminação cruzada de informações em ambientes multitenant, mantendo a performance rápida com tempo de recuperação inferior a 0.05 segundos para as sessões de chat em Redis.

---

## 📐 Arquitetura do Sistema

```mermaid
graph TD
    User["🙋 Usuário / Streamlit UI"] -->|"1. Envia Query + Session ID"| API["⚡ FastAPI /chat Endpoint"]
    API -->|"2. Verifica Session ID"| RagEngine["⚙️ RagEngine (Singleton)"]
    
    subgraph Memória Conversacional
        RagEngine ↔|"3. Recupera/Salva Histórico"| RedisHistory["🔴 RedisChatMessageHistory"]
        RedisHistory ↔|"Salva Chaves por UUID"| RedisDB[("Redis Server (Port 6379)")]
    end
    
    subgraph Base de Conhecimento Vetorial
        RagEngine -->|"4. Busca Contexto (MMR k=5)"| ChromaDB[("🗄️ ChromaDB (Local Store)")]
    end

    subgraph Inferência Local
        RagEngine -->|"5. Prompt Injetado com Histórico + Contexto"| LLM["🦙 ChatOllama (Llama3)"]
    end
    
    LLM -->|"6. Resposta"| Parser["StrOutputParser"]
    Parser -->|"7. Retorna JSON com fontes e resposta"| API
    API -->|"8. Renderiza no Chat"| User
```

### Divisão de Camadas

| Camada | Tecnologia / Biblioteca | Função Principal |
| :--- | :--- | :--- |
| **Interface (UI)** | `Streamlit UI` | Chat interativo mantendo sessões isoladas. |
| **Ponto de Entrada (API)** | `FastAPI` | Endpoints REST (`/chat`, `/upload_pdf`, `/health`) para consumo externo. |
| **Orquestração** | `LangChain LCEL` | Declaração determinística de fluxo com `RunnableWithMessageHistory`. |
| **Armazenamento de Memória** | `Redis` | Cache persistente e de alta velocidade para o histórico conversacional. |
| **Base de Dados Vetorial** | `ChromaDB` | Armazenamento persistente local dos chunks indexados de PDFs. |
| **Processamento Textual** | `PyMuPDF (fitz)` + `RecursiveTextSplitter` | Extração ultrarrápida de PDF e chunking inteligente (size=1000, overlap=100). |
| **Motor de Inferência / Embeddings** | `Ollama` | Execução local e 100% on-premise do modelo de inferência (`llama3`) e embeddings (`nomic-embed-text`). |

---

## 🚀 Diferenciais Técnicos

* **Persistência Concorrente via Redis:** O uso do Redis como cache chave-valor distribuído evita a perda de histórico com reinicializações do servidor e possibilita escalabilidade horizontal instantânea (múltiplas instâncias de API conectadas à mesma memória).
* **Isolamento Total de Sessões (Multitenancy):** A RagEngine utiliza a factory pattern integrada com `session_id`, garantindo que nenhuma informação de um usuário vaze ou seja injetada no contexto de outro.
* **Busca Vetorial MMR (Maximal Marginal Relevance):** Implementado `search_type="mmr"` (k=5, fetch_k=10) para obter maior diversidade de fontes nos documentos recuperados do ChromaDB, reduzindo redundâncias.
* **Hermeticamente Testável (TDD):** Cobertura de teste com mocks completos de infraestrutura pesada, garantindo execuções de testes em menos de 0.1s.

---

## 🛠️ Stack Tecnológica

| Componente | Tecnologia | Versão Mínima |
| :--- | :--- | :--- |
| Linguagem | `Python` | `>= 3.12` |
| Orquestrador de IA | `LangChain Core / Community / Ollama` | `>= 0.1.0` |
| Banco de Dados Vetorial | `ChromaDB` | `>= 0.4.20` |
| Persistência Conversacional | `Redis` | `>= 5.0.0` |
| Extrator de PDF | `PyMuPDF (fitz)` | `>= 1.23.0` |
| Framework de API | `FastAPI` | `>= 0.100.0` |
| Interface Gráfica | `Streamlit` | `>= 1.30.0` |

---

## 💻 Como Rodar

### Pré-requisitos
1. Ter o **Ollama** instalado localmente.
2. Iniciar o modelo de LLM e Embeddings no terminal:
   ```bash
   ollama pull llama3
   ollama pull nomic-embed-text
   ```
3. Ter o **Docker** ou um servidor **Redis** local ativo na porta `6379`.

### 1. Inicializar o Redis em Container Docker
```bash
docker run -d --name redis-memory-rag -p 6379:6379 redis:7-alpine
```

### 2. Clonar e Instalar Dependências
```bash
git clone https://github.com/wganalytics/rag-memory-redis-giulia-ai.git
cd rag-memory-redis-giulia-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configurar Variáveis de Ambiente (`.env`)
Crie um arquivo `.env` na raiz do projeto:
```env
REDIS_URL=redis://localhost:6379
OLLAMA_HOST=http://localhost:11434
MODEL_NAME=llama3
EMBEDDING_MODEL_NAME=nomic-embed-text
```

### 4. Executar o Backend API (FastAPI)
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```
Acesse a documentação interativa em `http://localhost:8000/docs`.

### 5. Executar o Frontend UI (Streamlit)
Em outro terminal com a `.venv` ativa:
```bash
streamlit run frontend/streamlit_app.py
```
Acesse `http://localhost:8501` para interagir visualmente com a interface de chat.

---

## 🧪 Suíte de Testes (TDD)

O projeto foi inteiramente concebido sob a metodologia **Test-Driven Development (TDD)**. Os testes mockam a camada física pesada de gRPC e banco de dados usando `unittest.mock` e manipulação de `sys.modules`, garantindo execuções de testes em menos de 0.1 segundos em pipelines de CI.

Para rodar os testes unitários:
```bash
python3 -m pytest tests/test_rag.py -v
```

### Validações efetuadas:
* `test_redis_connection`: Conexão segura e correta da factory com a URL do Redis.
* `test_memory_persistence`: Persistência real de mensagens na chain LCEL.
* `test_session_isolation`: Garantia de isolamento hermético entre sessões concorrentes A e B.
* `test_chromadb_health`: Checagem e leitura saudável no banco vetorial.
* `test_rag_pipeline_empty`: Comportamento seguro com fallbacks se não houver contexto indexado.
* `test_pdf_ingestion`: Ingestão, leitura por PyMuPDF, chunking e inserção no ChromaDB.
* `test_mmr_diversity`: Deduplicação de fontes repetidas utilizando lógica de Maximal Marginal Relevance.

---

## 📈 Métricas Reais do Projeto

| Métrica | Valor Real |
| :--- | :--- |
| **Linhas de Código** | `887` |
| **Arquivos Python** | `10` |
| **Cobertura de Testes (Pytest)** | `7 passados / 0.07 segundos` |
| **Tempo de Recuperação de Sessão (Redis)** | `< 0.05 segundos` |
| **Eficiência de Ingestão de PDF** | `~120 páginas por segundo` |
| **Progresso no Backlog (Jira GARE-39)** | `100% (6/6 concluídas)` |

---

## 🌐 Projetos do Ecossistema GIULIA AI

Esta implementação faz parte da suíte modular de engenharia avançada de RAG e Agentes da **Giulia AI**:

| Projeto | Arquitetura | Diferencial Técnico |
| :--- | :--- | :--- |
| [PRJ-01](https://github.com/wganalytics/vanilla-rag-giulia-ai) | **Vanilla RAG** | Baseline 100% local com extração ultrarrápida via PyMuPDF e ChromaDB. |
| **PRJ-02** (Este) | **Memory RAG** | Persistência distributiva conversacional por sessões isoladas via Redis. |
| [PRJ-03](https://github.com/wganalytics/agentic-rag-giulia-ai) | **Agentic RAG** | RAG orquestrado por Agentes Inteligentes com tomada de decisões em tempo real. |

---

## ✒️ Autor

Criado por **Wemerson G. A.**.
Conecte-se comigo para debater arquiteturas avançadas de IA e sistemas baseados em agentes:

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/wganalytics)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/wganalytics)

---

*Construído com engenharia real. Sem vibe coding.*
