"""
Multi-turn web search with LangChain and Bedrock models with history.
"""

from langchain_aws import ChatBedrockConverse
from langchain_aws import ChatBedrock
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv
from termcolor import cprint
from langgraph.checkpoint.memory import MemorySaver

import os

load_dotenv()

def load_environment():
    os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')
    MODEL_ID = os.getenv('MODEL_ID')
    return MODEL_ID

# 1. Initialize the Bedrock model
# llm = ChatBedrock(
#     model_id=MODEL_ID,
#     region_name="us-east-1"
# )
def create_model(model_id):
    llm = ChatBedrockConverse(
        model_id=model_id,
        region_name="us-east-1",
        streaming=True
    )
    return llm

# 2. Initialize a web search tool
def create_tools():
    search_tool = TavilySearch(max_results=3)
    tools = [search_tool]
    return tools

# 3. Create the agent with LangGraph
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


# 4. Invoke the agent stream until user types "quit" or "exit".
if __name__ == "__main__":
    MODEL_ID = load_environment()
    llm = create_model(MODEL_ID)
    tools = create_tools()
    config = create_config()
    agent = create_langgraph_agent(llm, tools)

    cprint("Welcome to the multi-turn web search with LangChain and Bedrock models!", "blue", attrs=["bold"])
    cprint("Type 'quit' or 'exit' to end the conversation.", "blue", attrs=["bold"])
    print()

    while True:
        cprint("USER: ", "blue", attrs=["bold"], end="")
        user_input = input()
        if user_input.lower() in ["quit", "exit"]:
            break

        chat(agent, config, user_input)