"""
Multi-turn web search with LangChain and Bedrock models with history.
"""

from langchain_aws import ChatBedrockConverse
from langchain.agents import create_agent
from dotenv import load_dotenv
from termcolor import cprint
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.tools import StructuredTool

from tools import TavilySearch, RAGTool

import os
import boto3

load_dotenv()

def load_environment():
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')
    MODEL_ID = os.getenv('MODEL_ID')

    bedrock_client = boto3.client(
        service_name='bedrock-runtime', region_name='us-east-1'
    )

    return MODEL_ID, bedrock_client

def create_model(model_id):
    llm = ChatBedrockConverse(
        model_id=model_id,
        region_name="us-east-1",
        streaming=True
    )
    return llm

def create_tools(search_tool, rag_tool):
    retrieve_tool = StructuredTool.from_function(
        name="RAGTool",
        description="A tool that retrieves relevant information from a knowledge base using embeddings.",
        func=rag_tool.retrieve
    )

    tools = [search_tool, retrieve_tool]
    return tools

def create_langgraph_agent(llm, tools):
    agent = create_agent(
        llm, 
        tools,
        checkpointer=MemorySaver() # Enable automatic memory persitence.
    )
    return agent

def create_config():
    config = {"configurable": {"thread_id": "conversation-123"}}
    return config

def chat(agent, config, user_input):
    stream = agent.stream_events(
        {"messages": [("user", user_input)]},
        version="v3",
        config=config
    )

    for kind, item in stream.interleave("messages", "tool_calls"):
        if kind == "messages":
            cprint("Assistant: ", "magenta", attrs=["bold"], end="")
            for token in item.text:
                cprint(token, "yellow", end="", flush=True)
        elif kind == "tool_calls":
            cprint(f"\nTool call: {item.tool_name}({item.input})", "green", attrs=["bold"])
            for delta in item.output_deltas:
                cprint(delta, "cyan", end="", flush=True)
            cprint(f"\nTool result: {item.output}", "green", attrs=["bold"])
        print()

    final_state = stream.output
    print()


if __name__ == "__main__":
    MODEL_ID, bedrock_client = load_environment()

    llm = create_model(MODEL_ID)
    search_tool = TavilySearch(max_results=3)
    rag_tool = RAGTool(
        embedding_model_id=os.getenv('EMBEDDING_MODEL_ID'), bedrock_client=bedrock_client
    )
    tools = create_tools(search_tool, rag_tool)
    config = create_config()
    agent = create_langgraph_agent(llm, tools)

    cprint("Welcome to the multi-turn web search with LangChain and Bedrock models!", "blue", attrs=["bold"])
    cprint("Type 'quit' or 'exit' to end the conversation.", "blue", attrs=["bold"])
    print()

    # Ask the user to pass PDF, text, or folder path to read documents and create ChromaDB
    folder_path = input("Enter the folder path to read documents (or press Enter to skip): ")
    file_path_pdf = input("Enter the PDF file path to read document (or press Enter to skip): ")
    file_path_text = input("Enter the text file path to read document (or press Enter to skip): ")
    
    # If either of the paths is empty, set it to "None"
    if file_path_pdf == "":
        file_path_pdf = "None"
    if file_path_text == "":
        file_path_text = "None"
    if folder_path == "":
        folder_path = "None"

    if folder_path != "None":
        # Read documents from folder
        docs_from_folder = rag_tool.read_directory(folder_path)
    else:
        docs_from_folder = []
    if file_path_pdf != "None":
        # Read PDF document
        docs_from_pdf = rag_tool.read_pdf(file_path_pdf)
    else:
        docs_from_pdf = []
    if file_path_text !="None":
        # Read text document
        docs_from_text = rag_tool.read_text(file_path_text)
    else:
        docs_from_text = []

    # Combine all documents
    all_docs = docs_from_folder + docs_from_pdf + docs_from_text

    chunks = rag_tool.get_chunks(all_docs)
    
    # Generate embeddings and store in ChromaDB
    db = rag_tool.embed_and_store(chunks)


    while True:
        cprint("USER: ", "blue", attrs=["bold"], end="")
        user_input = input()
        if user_input.lower() in ["quit", "exit"]:
            break

        chat(agent, config, user_input)