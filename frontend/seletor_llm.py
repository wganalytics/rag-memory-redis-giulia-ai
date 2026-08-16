"""Seletor de motor de LLM para telas que falam com a API do projeto."""

import streamlit as _st
import requests as _requests

_SUGESTOES = {
    "sem_credito": "Adicione crédito no painel do provider — a chave está correta.",
    "chave_invalida": "Gere uma nova chave e atualize o .env.",
    "limite_taxa": "Aguarde um pouco antes de tentar de novo.",
    "modelo_inexistente": "Confira o nome do modelo no .env.",
}


def _icone(p):
    if not p.get("available"):
        return "⛔"
    return {True: "✅", False: "⛔"}.get(p.get("verified"), "⚙️")


@_st.cache_resource(show_spinner="Testando os motores de LLM...")
def _motores(api_url: str):
    """Pede à API um teste real de cada motor, uma vez por processo."""
    try:
        r = _requests.get(f"{api_url}/providers", params={"probe": "true"}, timeout=180)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def selecionar_motor(api_url: str, sidebar: bool = True):
    """Desenha o seletor e devolve (provider, modelo). (None, None) se a API cair."""
    alvo = _st.sidebar if sidebar else _st
    info = _motores(api_url)
    if not info:
        alvo.warning("API offline: não foi possível listar os motores.")
        return None, None

    todos = info["providers"]
    rotulos = {p["provider"]: p["label"] for p in todos}
    por_nome = {p["provider"]: p for p in todos}
    funcionais = [p["provider"] for p in todos if p.get("verified") is True]
    quebrados = [p["provider"] for p in todos if p.get("verified") is not True]

    mostrar_todos = alvo.checkbox("Mostrar motores indisponíveis", value=not funcionais)
    ofertados = list(rotulos) if mostrar_todos else funcionais
    if not ofertados:
        alvo.error("Nenhum motor de LLM está funcionando.")
        for p in todos:
            alvo.write(f"{_icone(p)} **{p['provider']}** — {p['reason']}")
        _st.stop()

    padrao = info.get("default", ofertados[0])
    provider = alvo.selectbox(
        "🧠 Motor de LLM", options=ofertados,
        index=ofertados.index(padrao) if padrao in ofertados else 0,
        format_func=lambda p: rotulos[p],
    )

    # No Ollama a máquina sabe quais modelos existem: oferecer a lista evita
    # errar o nome e só descobrir na primeira pergunta.
    escolhido = por_nome[provider]
    locais = info.get("ollama_models", []) if provider == "ollama" else []
    padrao_modelo = escolhido["model"]
    if locais:
        if padrao_modelo not in locais:
            locais = [padrao_modelo] + locais
        modelo = alvo.selectbox("Modelo", options=locais, index=locais.index(padrao_modelo),
                                help="Modelos instalados no Ollama desta máquina.")
    else:
        modelo = alvo.text_input("Modelo", value=padrao_modelo)

    if escolhido.get("verified") is True:
        alvo.success("Respondeu ao teste")
    else:
        alvo.error(escolhido["reason"])
        if escolhido.get("categoria") in _SUGESTOES:
            alvo.caption(_SUGESTOES[escolhido["categoria"]])
    if quebrados:
        alvo.caption(f"Fora do seletor: {', '.join(quebrados)}")
    if alvo.button("🔄 Testar motores de novo", use_container_width=True):
        _motores.clear()
        _st.rerun()

    return provider, modelo
