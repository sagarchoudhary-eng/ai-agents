from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, ToolMessage

import math

# ── 1. Define your tools ──────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Example: '347 * 28' or '9716 / 4'"""
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def search_web(query: str) -> str:
    """Search the internet for current information."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
        if not results:
            return "No results found."
        return "\n".join(f"- {r['title']}: {r['body']}" for r in results)
    except Exception as e:
        return f"Search failed: {e}"

@tool
def read_file(filename: str) -> str:
    """Read contents of a local text file."""
    try:
        with open(filename, "r") as f:
            return f.read()
    except FileNotFoundError:
        return f"File '{filename}' not found."

# ── 2. Build the agent ────────────────────────────────────────────────

llm = ChatOllama(model="llama3.2:3b", temperature=0)

tools       = [calculator, search_web, read_file]
tool_map    = {t.name: t for t in tools}
llm_tools   = llm.bind_tools(tools)

SYSTEM = SystemMessage(content="""You are a helpful assistant with access to tools.

For every task, follow this loop:
  Thought: reason about what to do next
  Action: call a tool if needed
  Observation: read the result
  ...repeat until you have enough info...
  Final Answer: give the user a clear answer

Always think before acting. Use tools when you need current data or calculations.
Never guess — use a tool instead.""")

# ── 3. The ReAct loop ─────────────────────────────────────────────────

def run_agent(question: str, max_steps: int = 8):
    print(f"\n{'='*55}")
    print(f"Question: {question}")
    print('='*55)

    messages = [SYSTEM, HumanMessage(content=question)]
    step = 0

    while step < max_steps:
        step += 1
        response = llm_tools.invoke(messages)
        messages.append(response)

        # No tool calls → final answer reached
        if not response.tool_calls:
            print(f"\nFinal Answer: {response.content}")
            break

        # Run each tool the model requested
        for tc in response.tool_calls:
            name = tc["name"]
            args = tc["args"]
            print(f"\n  [Thought] Calling {name}({args})")

            if name in tool_map:
                result = tool_map[name].invoke(args)
                print(f"  [Observation] {result}")
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tc["id"]
                ))
            else:
                messages.append(ToolMessage(
                    content=f"Tool '{name}' not found.",
                    tool_call_id=tc["id"]
                ))

    else:
        print(f"\n[Agent stopped after {max_steps} steps]")

# ── 4. Test it ────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_agent("What is square of 333 and cube of 221?")
    run_agent("What is the square root of 1764?")
    run_agent("What is the react agent ?")