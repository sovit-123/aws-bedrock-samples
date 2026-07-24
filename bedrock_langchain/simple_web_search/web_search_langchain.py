"""
Simple web search with LangChain and Bedrock models.
"""

from langchain_aws import ChatBedrock
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv
from termcolor import cprint

import os

load_dotenv()

os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')

MODEL_ID = os.getenv('MODEL_ID')

# 1. Initialize the Bedrock model
llm = ChatBedrock(
    model_id=MODEL_ID,
    region_name="us-east-1"
)

# 2. Initialize a web search tool
search_tool = TavilySearch(max_results=3)
tools = [search_tool]

# 3. Create the agent with LangGraph
agent = create_agent(llm, tools)

# 4. Invoke the agent stream until.
user_input = input("Enter your query: ")

stream = agent.stream_events(
    {"messages": [("user", user_input)]},
    version="v3"
)

for kind, item in stream.interleave("messages", "tool_calls"):
    if kind == "messages":
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