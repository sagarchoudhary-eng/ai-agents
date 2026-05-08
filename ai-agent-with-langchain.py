from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

llm = ChatOllama(model="llama3.2:3b")

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain machine learning in 2 sentences.")
]

response = llm.invoke(messages)
print(response.content)