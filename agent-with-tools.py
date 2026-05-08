from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

# ── Step 1: Define your tools using @tool decorator ──────────────────
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression. Input: a string like '25 * 4 + 10'"""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool
def get_word_count(text: str) -> str:
    """Count the number of words in a piece of text."""
    count = len(text.split())
    return f"Word count: {count}"

# ── Step 2: Bind tools to the LLM ────────────────────────────────────
llm = ChatOllama(model="llama3.2:3b", temperature=0)
tools = [calculator, get_word_count]
llm_with_tools = llm.bind_tools(tools)

# ── Step 3: Build the agent loop ─────────────────────────────────────
def run_agent(user_question: str):
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed."),
        HumanMessage(content=user_question)
    ]

    print(f"\nUser: {user_question}")

    while True:
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # No tool calls → model gave a final answer
        if not response.tool_calls:
            print(f"Assistant: {response.content}")
            break

        # Tool calls found → run each one
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"  [Tool call] {tool_name}({tool_args})")

            # Find and run the right tool
            tool_map = {t.name: t for t in tools}
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                print(f"  [Tool result] {result}")

                # Add tool result back to conversation
                from langchain_core.messages import ToolMessage
                messages.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))

# ── Step 4: Test it ───────────────────────────────────────────────────
run_agent("What is 347 multiplied by 28?")
run_agent("How many words are in: 'The quick brown fox jumps over the lazy dog'?")