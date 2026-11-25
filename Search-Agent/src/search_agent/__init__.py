from dotenv import load_dotenv

load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

"""
from tavily import TavilyClient
tavily = TavilyClient()
"""

from langchain_tavily import TavilySearch

@tool
def search_agent(query: str) -> str:
    """
    Tool that searches over internet
    Args:
        query: The query to search for
    Returns:
        The search results
    """
    print(f"Search results for: {query}")
    return tavily.search(query=query)

#llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0.0)
#llm = ChatGroq(model="qwen/qwen3.8-27b", temperature=0.0)
llm = ChatOllama(model="gemma4:12b", temperature=0.0)

#tools = [search_agent]
tools = [TavilySearch()] #Herramienta de búsqueda de Tavily implementada en langchain

agent =create_agent(model=llm, tools=tools)

def main() -> None:
    print("Hello from search-agent!")
    #result = agent.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo?")]})
    result = agent.invoke({"messages": [HumanMessage(content="search for 3 job postings for an ai engineer using langchain in madrid area on linkedin and list their details")]} )

    print(result)

if __name__ == "__main__":
    main()

