from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOllama(model="llama3.2:3b", temperature=0.2)  # low temp = more consistent

# 1. System prompt — define the role clearly
system = SystemMessage(content="""You are a helpful coding assistant specializing in Python.
When answering:
- Be concise and direct
- Always include a working code example
- If you don't know something, say so — never guess
- Format code in plain text, no markdown""")

# 2. Few-shot example — show the pattern once
few_shot_user = HumanMessage(content="How do I read a file in Python?")
few_shot_assistant = {"role": "assistant", "content": """Use the built-in open() function.

with open('myfile.txt', 'r') as f:
    content = f.read()
print(content)"""}

# 3. Actual user question
user_question = HumanMessage(content="How can I train an llm in local laptop?")

# Build message history with examples included
from langchain_core.messages import AIMessage

messages = [
    system,
    few_shot_user,
    AIMessage(content=few_shot_assistant["content"]),  # inject the example answer
    user_question
]

response = llm.invoke(messages)
print(response.content)