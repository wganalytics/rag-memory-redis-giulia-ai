import requests
import time
import json

API_URL = "http://127.0.0.1:8000"
SESSION_ID = "sessao-teste-001"

def print_separator():
    print("\n" + "="*50 + "\n")

def check_health():
    print("Verificando Health Check da API...")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print(f"Erro: Não foi possível conectar a {API_URL}. A API está rodando?")
        return False

def execute_query(question: str, session_id: str):
    print(f"Pergunta: '{question}'")
    payload = {
        "question": question,
        "session_id": session_id
    }
    
    start_time = time.time()
    response = requests.post(f"{API_URL}/chat", json=payload)
    end_time = time.time()
    
    print(f"Tempo de resposta: {end_time - start_time:.2f} segundos")
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Resposta da IA: {data.get('answer')}")
    else:
        print(f"❌ Erro na consulta (Status {response.status_code}): {response.text}")

def main():
    print("🚀 Iniciando Teste de Validação de Memória RAG (PRJ-02)")
    
    if not check_health():
        return
        
    print_separator()
    sid1 = "sessao-teste-001"
    print(f"Sessão ID de Teste: {sid1}")
    
    print_separator()
    print("➤ PASSO 1: Enviando primeira mensagem alimentando contexto pessoal.")
    execute_query("Olá! Meu nome é TesteBot e meu animal favorito é um dragão.", sid1)
    
    print_separator()
    print("➤ PASSO 2: Validando a Memória (Redis). Fazendo uma pergunta sobre o contexto anterior.")
    execute_query("Lembra da nossa conversa anterior? Qual é o meu nome e o meu animal favorito?", sid1)
    
    print_separator()
    print("➤ PASSO 3: Enviando mensagem em uma SESSÃO DIFERENTE para provar o isolamento.")
    sid2 = "sessao-teste-isolado-002"
    print(f"Nova Sessão ID: {sid2}")
    execute_query("Qual é o meu nome e o meu animal favorito?", sid2)
    
    print_separator()
    print("🎉 Teste concluído.")

if __name__ == "__main__":
    main()
