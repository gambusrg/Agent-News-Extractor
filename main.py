import os
import logging
from pydantic import BaseModel, Field
from typing import Literal
from langchain.tools import Tool, StructuredTool
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain.agents import tool
from langchain.schema import SystemMessage, HumanMessage
from langchain.agents import initialize_agent
from langgraph.graph import MessagesState
from langgraph.types import Command
from langgraph.graph import StateGraph, START, END
import re
import json
import urllib.request
from IPython.display import Image, display
from langchain.output_parsers import PydanticOutputParser


load_dotenv(".env")

# Configurar el logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

_logger = logging.getLogger(__name__)

class Router(BaseModel):
    """Router class to route messages to different agents. If do not need agents, go FINISH."""
    next: Literal["extractor", "preprocessor", "FINISH"]
    reason: str = Field(description="Reason for the next action")


@tool
def fetch_news(query: str, max_results: int) -> dict:
    """
    Fetches news articles based on a query string using the GNews API.

    Args:
        query (str): The search query to fetch news articles.
        api_key (str): Your GNews API key.
        language (str): The language of the news articles (default is 'en').
        max_results (int): The maximum number of articles to return (default is 10).
        country (str): The country to filter news articles (optional).
        sort (str): The sorting order of the articles (optional).
        from_date (str): The start date for the articles in ISO 8601 format (optional).
        to_date (str): The end date for the articles in ISO 8601 format (optional).

    Returns:
        dict: A dictionary containing the fetched news articles.
    """

     # URL encode the query to handle spaces and special characters
    encoded_query = urllib.parse.quote(query)
    url = f"https://gnews.io/api/v4/search?q={encoded_query}&max={max_results}&apikey={os.environ.get('API_KEY')}"

    news = {}
    try:
        response = urllib.request.urlopen(url)
        _logger.info(f"Fetching news articles for query: {query}")
        news_data = json.loads(response.read())
        news["title"] = [article.get('title') for article in news_data['articles']]
        news['articles'] = [article.get('content') for article in news_data['articles']]
    except Exception as e:
        _logger.error(f"Error fetching news articles: {e}")
        news = {}
    return news

@tool
def preprocess_text(text: dict) -> dict:
    """
    Preprocesses the text by removing HTML tags and normalizing the text.

    Args: 
        text (dict): A dictionary containing the following keys:
            - "title" (str): The title to preprocess.
            - "articles" (str): The articles to preprocess.

    Returns: 
        dict: A dictionary with the preprocessed "title" and "articles", both in lowercase and without HTML tags.
    """
       
    _logger.info(f"Preprocessing text: {text}")
    # Remove html tags
    text["title"] = re.sub(r"<.*?>", "", text["title"])
    text["articles"] = re.sub(r"<.*?>", "", text["articles"])
    # Normalize text
    text["title"] = text["title"].lower()
    text["articles"] = text["articles"].lower()
    return text

tools = [fetch_news, preprocess_text]

# System prompts
system_prompt = f"""
You are a Supervisor Agent. Your principal task is to supervise the other agents.
You must follow these instructions:
 1. ** Read the instruction and send the message to the API Agent.**
 2. ** If you recieve a message from the API Agent, send the information to the Preprocessor agent and ask him to preprocess the information.**
 3. ** If you recive the preprocessed information from the Preprocessor agent, determine if the preprocessed information is correct.**
 4. ** If not correct, ask the Preprocessor Agent to preprocess the information again.**
 5. ** If the information is correct FINISH the process.**
 6. ** Repeat the process until the information is correct.**

You must decide where to send the message next. Your options are:
- "extractor": Send to the API Agent to extract information
- "preprocessor": Send to the Preprocessor Agent to preprocess information
- "FINISH": End the process when all steps are complete

Important note: 
- if you recive the message from the preprocessor agent and it is normalized, you can FINISH the process.
Respond with a Dictionary in the following format:
{{"next": "extractor"}} or {{"next": "preprocessor"}} or {{"next": "FINISH"}} 
{{"reason": "Your reason here"}}
"""

system_prompt_api_agent = """
You are an Agent that extracts news through an API. Your principal task is do what the Supervisor Agent will tell you. 
You must follow these instructions:
 1. ** Do what you are told by the Supervisor Agent and just the tools you have**
 2. ** Once you have finished, send the information to the suppervisor Agent.**
 3. ** Wait for more instructions from the Supervisor Agent.**
""" 

system_prompt_preprocessor_agent = """
You are a preprocessor agent. Your principal task is to preprocess the data extracted by the API agent and provided by the Supervisor Agent.
You must follow these instructions:
 1. ** Do what you are told by the Supervisor Agent and just the tools you have**
 2. ** Preprocess the data extracted by the API agent.**
 3. ** Delete the html tags from the text.**
 4. ** Normalize the text.**
 5. ** Return the preprocessed text in json format.**
 6. ** Send the preprocessed text to the Supervisor Agent.**
 7. ** Wait for more instructions from the Supervisor Agent.**

"""

# Initialize the OllamaLLM models
llm = OllamaLLM(
    model="mistral",
    temperature=0.5,
    num_predict=4000
)

llm_supervisor = OllamaLLM(
    model="mistral",
    temperature=0.1,
    num_predict=1024
)

# Create output parser for the supervisor
router_parser = PydanticOutputParser(pydantic_object=Router)

# Create Agents
agent_supervisor = initialize_agent(
    llm=llm_supervisor,
    tools=[],
    agent="structured-chat-zero-shot-react-description",
    state_modifier=SystemMessage(content=system_prompt)
)

agent_api = initialize_agent(
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    tools=tools,
    state_modifier=SystemMessage(content=system_prompt_api_agent)
)

agent_preprocessor = initialize_agent(
    llm=llm,
    agent="structured-chat-zero-shot-react-description",
    tools=tools,
    state_modifier=SystemMessage(content=system_prompt_preprocessor_agent)
)

def supervisor_node(state: MessagesState) -> Command[Literal["extractor", "preprocessor", "__end__"]]:
    """Supervisor node to route messages to different agents.
    Args:
        state (MessagesState): The current state of the conversation.
    Returns:
    Command[Literal["extractor", "preprocessor", "__end__"]]: The command to execute"""

    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    
    # Invoke the LLM
    _logger.info(f"=====ENTRADA SUPERVISOR=====Invoking the LLM model {state['messages'][-1].content}")
    result = llm_supervisor.invoke(messages)
    if isinstance(result, str):
        response = json.loads(result)
    else:
        response = result
    try:
        next_action = response.get('next')  # Access the 'next' field
        reason = response.get('reason', 'No reason provided')  # Access the 'reason' field
        
        _logger.info(f"=====SALIDA SUPERVISOR=====LLM raw response: {next_action} Reason: {reason}")
        
        return Command(goto=next_action, update={"reason": reason})  # Proceed with the next command
    except json.JSONDecodeError as e:
        _logger.error(f"Error parsing LLM response: {e}")
        return Command(goto="extractor")  

   
def api_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Connects to an API and extracts information about news."""
    _logger.info(f"=====ENTRADA API=====Invoking the LLM model {state['messages'][-1].content}")
    response = agent_api.invoke({"input": state})
    _logger.info(f"=====SALIDA API=====LLM response: {response}")  
    return Command(
        update = {"messages": [
            HumanMessage(content=response["output"], name="API Agent")
        ]},
        goto="supervisor"
    )

def preprocessor_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Preprocesses the information extracted by the API agent.
    Args:
        state (MessagesState): The current state of the conversation.
    Returns:
        Command[Literal["supervisor"]]: The command to execute
    """
    _logger.info(f'=====ENTRADA PREPROCESSOR=====Invoking the LLM model {state["messages"][-1].content}')
    response = agent_preprocessor.invoke({"input": state})
    _logger.info(f"=====SALIDA PREPROCESSOR=====LLM response: {response}")  
    return Command(
        update = {"messages": [
            HumanMessage(content=response["output"], name="Preprocessor Agent")
        ]},
        goto = "supervisor"
    )

builder = StateGraph(MessagesState)
builder.add_edge(START, "supervisor")

builder.add_node("supervisor", supervisor_node)
builder.add_node("extractor", api_node)
builder.add_node("preprocessor", preprocessor_node)

graph = builder.compile()

# Display the graph visualization
display(Image(graph.get_graph().draw_mermaid_png()))

def main():
    list_messages = []
    user_question = "Extract the last new published"
    try:
        for s in graph.stream(
            {"messages": [HumanMessage(content=user_question)]},
            subgraphs=True,
            config={"recursion_limit": 10}
        ):
            list_messages.append(s)
    except Exception as e:
        _logger.exception(e)
        return
    
if __name__ == "__main__":
    main()