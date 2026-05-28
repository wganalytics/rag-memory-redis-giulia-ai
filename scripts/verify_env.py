import os
import sys

# Definir as cores para output visual no terminal
VERDE = '\033[92m'
CORTA = '\033[0m'
VERMELHO = '\033[91m'

print("="*50)
print(f"🤖 INICIANDO CHECAGEM DO AMBIENTE {VERDE}RAG VANILLA{CORTA}")
print("="*50)

def checar_importacao(modulo_nome, biblioteca_relacionada):
    try:
        __import__(modulo_nome)
        print(f"[{VERDE}OK{CORTA}] {biblioteca_relacionada} inicializada perfeitamente.")
        return True
    except ImportError as e:
        print(f"[{VERMELHO}FALHA{CORTA}] Erro ao importar {biblioteca_relacionada}: {e}")
        return False

# Checando LangChain Core
sucesso = checar_importacao("langchain_core", "LangChain Framework")

# Checando VectorDB Chroma
sucesso = checar_importacao("chromadb", "Chroma VectorDB") and sucesso

# Checando Ingestor de PDF
sucesso = checar_importacao("fitz", "PyMuPDF (Leitor Visual de Manuais)") and sucesso

# Checando Conector Ollama
sucesso = checar_importacao("langchain_ollama", "Adaptador Nativo Ollama") and sucesso

# Checando Leitor de Variaveis de Ambiente
sucesso = checar_importacao("dotenv", "Leitor de Variáveis de Ambiente (.env)") and sucesso

print("-" * 50)
if sucesso:
    print(f"🎉 SUCCESSO TOTAL: Seu Mac M1 está pronto para os RAGs locais!")
    print(f"Lembre-se de checar se o App do Ollama está aberto quando formos rodar os modelos!")
else:
    print(f"⚠️ PROBLEMAS DETECTADOS: Verifique as mensagens de 'FALHA' acima.")
    sys.exit(1)
print("=" * 50)
