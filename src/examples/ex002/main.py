"""
Exemplo 002: Agente de chat com LangGraph + checkpoint em memória.

Este script implementa um fluxo de conversa (human -> LLM -> resposta)
com persistência de estado por thread (InMemorySaver) e salvamento das
interações em um arquivo JSON.
"""

import json
import re
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages
from rich import print
from rich.markdown import Markdown

# -----------------------------------------------------------------------------
# Configuração do ambiente e do modelo (LLM)
# -----------------------------------------------------------------------------
# Carrega variáveis do .env na raiz do projeto (ex.: GOOGLE_API_KEY).
# Path(__file__).parents[3] sobe de ex002/main.py até a raiz do projeto.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Inicializa o modelo de linguagem. Formato: "provedor:nome_do_modelo".
# Ollama roda localmente; para Google: "google_genai:gemini-2.5-flash" + api_key.
llm = init_chat_model("ollama:deepseek-r1:8b")


# -----------------------------------------------------------------------------
# 1. Estado do agente (State)
# -----------------------------------------------------------------------------
# O agente mantém apenas uma lista de mensagens. Annotated com add_messages
# faz com que, ao retornar {"messages": [nova_msg]}, o LangGraph faça merge
# (append) em vez de substituir a lista inteira. Assim o histórico cresce
# a cada turno (human + assistant).
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# -----------------------------------------------------------------------------
# 2. Nó do grafo: chamar o LLM
# -----------------------------------------------------------------------------
# Recebe o estado atual (com todas as mensagens), envia para o LLM e retorna
# o estado atualizado com a resposta (AIMessage). O merge com add_messages
# adiciona essa nova mensagem à sequência existente.
def call_llm(state: AgentState):
    llm_result = llm.invoke(state["messages"])
    return {"messages": [llm_result]}


# -----------------------------------------------------------------------------
# 3. Construção do grafo (StateGraph)
# -----------------------------------------------------------------------------
# StateGraph define o fluxo: tipo do estado, nós (funções) e arestas (transições).
# - START -> call_llm -> END: fluxo linear (uma pergunta -> uma resposta).
builder = StateGraph(
    AgentState, context_schema=None, input_schema=AgentState, output_schema=AgentState
)
builder.add_node("call_llm", call_llm)
builder.add_edge(START, "call_llm")
builder.add_edge("call_llm", END)


# -----------------------------------------------------------------------------
# 4. Compilação do grafo com checkpoint (memória por thread)
# -----------------------------------------------------------------------------
# InMemorySaver guarda o estado da conversa em memória, indexado por thread_id.
# Assim, cada thread (ou execução com mesmo thread_id) mantém seu próprio
# histórico. RunnableConfig passa o thread_id para o checkpointer.
# Sem checkpoint, cada invoke começaria do zero (sem histórico de mensagens).
checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
config = RunnableConfig(
    configurable={"thread_id": threading.get_ident()}
)


# -----------------------------------------------------------------------------
# 5. Persistência das interações em JSON
# -----------------------------------------------------------------------------
# Caminho do arquivo único que armazena todas as execuções (lista de blocos).
ARQUIVO_INTERACOES = Path(__file__).resolve().parents[3] / "interactions.json"


def _limpar_conteudo(content) -> str:
    """
    Normaliza o texto para exibição/serialização: remove quebras de linha (\\n),
    retorno de carro (\\r), tabulação (\\t) e colapsa espaços múltiplos em um só.
    Evita que o JSON fique cheio de \\n e caracteres de controle.
    """
    if isinstance(content, str):
        texto = content.replace("\n", " ").replace("\r", " ").replace("\t", " ")
        return re.sub(r" +", " ", texto).strip()
    return str(content)


def messages_para_json(messages: Sequence[BaseMessage]) -> list[dict]:
    """
    Converte a lista de mensagens (HumanMessage, AIMessage, etc.) em uma lista
    de dicionários serializáveis: tipo da mensagem e conteúdo (já normalizado).
    Usado ao salvar no interactions.json.
    """
    return [
        {
            "type": type(msg).__name__,
            "content": _limpar_conteudo(getattr(msg, "content", str(msg))),
        }
        for msg in messages
    ]


def _nome_do_modelo(llm) -> str:
    """
    Obtém o nome do modelo de forma compatível com diferentes backends:
    ChatGoogleGenerativeAI usa model_name; ChatOllama usa model.
    """
    return getattr(llm, "model_name", None) or getattr(llm, "model", str(llm))


def salvar_interacoes(
    messages: Sequence[BaseMessage], path: Path = ARQUIVO_INTERACOES
) -> None:
    """
    Adiciona uma nova execução ao interactions.json. O arquivo é sempre um
    array JSON válido: lê o array existente, appenda o novo bloco (data_hora,
    model, mensagens, etc.) e reescreve o arquivo. Se o arquivo não existir
    ou estiver inválido, começa com uma lista vazia.
    """
    data_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    bloco = {
        "data_hora": data_hora,
        "model": _nome_do_modelo(llm),
        "temperature": getattr(llm, "temperature", None),
        "max_tokens": getattr(llm, "max_tokens", None),
        "top_p": getattr(llm, "top_p", None),
        "messages": messages_para_json(messages),
    }
    if path.exists() and path.stat().st_size > 0:
        try:
            lista = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(lista, list):
                lista = [lista]
        except json.JSONDecodeError:
            lista = []
    else:
        lista = []
    lista.append(bloco)
    path.write_text(json.dumps(lista, ensure_ascii=False, indent=2), encoding="utf-8")


# -----------------------------------------------------------------------------
# Loop principal: chat interativo
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # current_messages guarda o histórico da sessão atual (para exibição e,
    # ao sair, para salvar no JSON). O histórico “real” do grafo está no
    # checkpointer (por thread_id).
    current_messages: Sequence[BaseMessage] = []

    while True:
        human_message = input("Digite uma mensagem:")
        print(Markdown("---"))

        # Comando para encerrar: salva as interações no JSON e sai.
        if human_message.lower() in ["sair", "exit", "quit", "q"]:
            print("Saindo...")
            if current_messages:
                salvar_interacoes(current_messages)
                print(f"Interações salvas em: {ARQUIVO_INTERACOES}")
            break

        # Envia apenas a nova mensagem; o checkpointer fornece o histórico
        # anterior ao grafo, então o LLM vê toda a conversa.
        human_message = HumanMessage(content=human_message)
        result = graph.invoke({"messages": [human_message]}, config=config)
        current_messages = result["messages"]

        # Formata a resposta em Markdown para exibição no terminal (Rich).
        def messages_para_markdown(messages):
            blocos = []
            for msg in messages:
                tipo = type(msg).__name__
                content = getattr(msg, "content", str(msg))
                if isinstance(content, str):
                    blocos.append(f"**{tipo}**\n\n{content}")
                else:
                    blocos.append(f"**{tipo}**\n\n{content!r}")
            return "\n\n---\n\n".join(blocos)

        markdown_saida = messages_para_markdown(result["messages"])
        print(Markdown(markdown_saida))
        print(Markdown("---"))
