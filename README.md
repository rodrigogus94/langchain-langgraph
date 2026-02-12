# LangChain-Langgraph-Agent

Projeto de estudo com **LangChain** e **LangGraph**: agentes de chat que conversam com um LLM (modelo de linguagem) em fluxo interativo. Inclui exemplos com e sem checkpoint (memória entre chamadas) e salvamento das interações em JSON.

---

## Índice

- [O que é este projeto?](#o-que-é-este-projeto)
- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Estrutura do projeto](#estrutura-do-projeto)
- [Exemplos explicados](#exemplos-explicados)
- [Como executar](#como-executar)
- [Arquivo de interações (JSON)](#arquivo-de-interações-json)
- [Modelos suportados](#modelos-suportados)
- [Licença](#licença)

---

## O que é este projeto?

Este repositório contém **dois exemplos** de agente de chat construídos com:

- **LangChain**: integração com modelos de linguagem (Ollama, Google Gemini, etc.).
- **LangGraph**: grafo de estados que define o fluxo da conversa (mensagem do usuário → LLM → resposta).

Em ambos os exemplos o fluxo é o mesmo: você digita uma mensagem, o agente envia o histórico (ou a nova mensagem + estado salvo) ao LLM e exibe a resposta em Markdown no terminal. A diferença está em **como o histórico é mantido** e em **como as interações são salvas** ao sair.

---

## Pré-requisitos

- **Python 3.13+** (indicado no `.python-version`).
- **uv** (gerenciador de pacotes e ambientes): [instalação](https://docs.astral.sh/uv/getting-started/installation/).
- Para usar **Ollama**: [Ollama instalado](https://ollama.com) e um modelo baixado (ex.: `ollama pull deepseek-r1:8b`).
- Para usar **Google Gemini**: chave de API no [Google AI Studio](https://aistudio.google.com/apikey).

---

## Instalação

Na raiz do projeto:

```bash
uv sync
```

Isso cria o ambiente virtual (`.venv`) e instala as dependências do `pyproject.toml` (LangChain, LangGraph, Rich, python-dotenv, etc.).

---

## Configuração

### Variáveis de ambiente

Copie o arquivo de exemplo e preencha conforme o modelo que for usar:

```bash
cp .env-examples .env
```

- **Ollama**: não é obrigatório colocar nada no `.env`; o código usa `ollama:deepseek-r1:8b` por padrão.
- **Google Gemini**: no `.env`, defina:
  - `GOOGLE_API_KEY="sua-chave-aqui"`

O script carrega o `.env` da raiz do projeto automaticamente (via `python-dotenv`). O arquivo `.env` não deve ser commitado (já está no `.gitignore`).

---

## Estrutura do projeto

```
Projeto-IA-MK02/
├── .env                    # Variáveis de ambiente (não versionado)
├── .env-examples           # Exemplo de variáveis (GOOGLE_API_KEY, etc.)
├── .python-version         # Python 3.13
├── pyproject.toml          # Nome do projeto, dependências, ferramentas (Ruff, pytest, etc.)
├── uv.lock                 # Lock das dependências (uv)
├── README.md               # Este arquivo
├── interactions.json       # Gerado ao sair do chat: histórico das execuções (não versionado)
└── src/
    └── examples/
        ├── ex001/
        │   └── main.py     # Agente sem checkpoint; histórico em current_messages
        └── ex002/
            └── main.py     # Agente com InMemorySaver (checkpoint por thread)
```

- **ex001** e **ex002**: cada um é um script executável que sobe um chat no terminal.
- **interactions.json**: criado/atualizado quando você digita "sair" em qualquer exemplo; guarda data/hora, modelo e mensagens da sessão.

---

## Exemplos explicados

### ex001 – Agente sem checkpoint

- **O que faz:** fluxo linear: sua mensagem → nó `call_llm` → resposta do LLM.
- **Histórico:** o próprio script mantém a lista de mensagens em `current_messages`. A cada nova mensagem, ele **acrescenta** à lista e passa **todo o histórico** para `graph.invoke({"messages": current_messages})`.
- **Quando usar:** quando não precisa de checkpoint (memória entre sessões ou threads). Tudo roda em uma única sessão e o histórico existe só em memória até você sair.

**Resumo:** o grafo não “lembra” de nada entre uma chamada e outra; quem lembra é a variável `current_messages` no loop.

---

### ex002 – Agente com checkpoint (InMemorySaver)

- **O que faz:** mesmo fluxo (mensagem → `call_llm` → resposta), mas o grafo é compilado com **checkpointer** (`InMemorySaver`).
- **Histórico:** o estado da conversa é salvo pelo LangGraph e identificado por `thread_id` (aqui, o id da thread atual). Você envia **apenas a nova mensagem** no `invoke`; o checkpointer entrega o histórico anterior ao grafo.
- **Quando usar:** quando quer memória por “sessão” (por thread) sem precisar montar manualmente a lista de mensagens. Útil para evoluir para múltiplos usuários ou múltiplas conversas (cada um com seu `thread_id`).

**Resumo:** o grafo “lembra” do estado; você só passa a última mensagem e o `config` com `thread_id`.

---

### Diferença prática

| Aspecto              | ex001                         | ex002                              |
|----------------------|-------------------------------|------------------------------------|
| Quem guarda histórico| Variável `current_messages`   | Checkpointer (InMemorySaver)        |
| O que vai no invoke   | Todo o histórico              | Só a nova mensagem + `config`      |
| Persistência do estado| Só na sessão atual            | Por thread (em memória)            |

Em ambos, ao digitar **sair**, as mensagens da sessão são escritas no **interactions.json** (lista de blocos por execução).

---

## Como executar

Na raiz do projeto (com `uv`):

```bash
# Exemplo 001 – sem checkpoint
uv run .\src\examples\ex001\main.py

# Exemplo 002 – com checkpoint
uv run .\src\examples\ex002\main.py
```

No Windows use `.\`; em Linux/macOS use `./`.

- Digite uma mensagem e pressione Enter para receber a resposta (renderizada em Markdown no terminal).
- Para encerrar e salvar as interações no JSON: digite `sair`, `exit`, `quit` ou `q`.

---

## Arquivo de interações (JSON)

- **Nome:** `interactions.json` (na raiz do projeto).
- **Quando é escrito:** ao sair do chat (comando "sair" / "exit" / "quit" / "q"), se houver pelo menos uma mensagem na sessão.
- **Formato:** um único array JSON. Cada elemento é um **bloco** de uma execução, por exemplo:

```json
{
  "data_hora": "2026-02-12 16:00:00",
  "model": "deepseek-r1:8b",
  "temperature": null,
  "max_tokens": null,
  "top_p": null,
  "messages": [
    { "type": "HumanMessage", "content": "Olá" },
    { "type": "AIMessage", "content": "Olá! Como posso ajudar?" }
  ]
}
```

- O conteúdo das mensagens é normalizado (quebras de linha e tabs viram espaço) para o JSON ficar legível.
- O arquivo está no `.gitignore`; não é versionado.

---

## Modelos suportados

Os exemplos usam `init_chat_model` com o formato `"provedor:modelo"`:

- **Ollama (local):**  
  `init_chat_model("ollama:deepseek-r1:8b")`  
  Ou outro modelo: `ollama:llama3`, `ollama:mistral`, etc.

- **Google Gemini (API):**  
  No código: `init_chat_model("google_genai:gemini-2.5-flash", api_key=os.getenv("GOOGLE_API_KEY"))`  
  E no `.env`: `GOOGLE_API_KEY="sua-chave"`.

Os comentários dentro de `ex001/main.py` e `ex002/main.py` indicam onde trocar o modelo.

---

## Licença

MIT.
