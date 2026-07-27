# Multiple Tool use with LangChain, LangGraph, and Bedrock

This folder contains the code to create a simple agent that can invoke the following tools based on the user's prompt:

* Web search
* RAG
* URL search - whenever the user prompt something like "Find the information from this URL - https://en.wikipedia.org/wiki/Laptop"

**How to execute?**

```
python run_chat.py
```

**Requiremets**: Add the the `TAVILY_API_KEY` API key to your `.env` file.

```
TAVILY_API_KEY=YOUR_API_KEY
```

