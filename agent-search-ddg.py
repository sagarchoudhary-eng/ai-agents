from duckduckgo_search import DDGS
from langchain_core.tools import tool

@tool
def search_web(query: str) -> str:
    """Search the web for current information on a topic."""
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=3))
    if not results:
        return "No results found."
    # Return top 3 results as a summary
    summary = ""
    for r in results:
        summary += f"- {r['title']}: {r['body']}\n"
    return summary