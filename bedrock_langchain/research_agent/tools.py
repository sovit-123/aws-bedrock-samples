from langchain_tavily import TavilySearch
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter  
from langchain_aws import BedrockEmbeddings
from dotenv import load_dotenv
from langchain.tools import tool

import os

load_dotenv()

# Set the API key as an environment variable
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')

class TavilySearchTool:
    """
    A wrapper class for the TavilySearch tool.
    """

    def __init__(self, max_results=3):
        self.max_results = max_results
       
    def create_search_tool(self):
        """
        Performs a search using the TavilySearch tool.
        """
        tool = TavilySearch(max_results=self.max_results)
        return tool

class RAGTool:
    """
    A wrapper class for RAG (Retrieval-Augmented Generation) tool.
    """

    def __init__(self, embedding_model_id, bedrock_client):
        self.embedding_model_id = embedding_model_id
        self.bedrock_client = bedrock_client
        self.bedrock_embeddings = BedrockEmbeddings(
            model_id=self.embedding_model_id, client=self.bedrock_client
        )
        self.db = None

    def read_directory(self, folder_path):
        """
        Reads all documents from the folder path.
        """
        loader = DirectoryLoader(folder_path)
        docs = loader.load()
        print(f"Number of documents read from folder: {len(docs)}")
        
        return docs

    def read_pdf(self, file_path):
        """
        Reads PDF documents.
        """
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        return docs

    def read_text(self, file_path):
        """
        Reads text documents.
        """
        loader = TextLoader(file_path)
        docs = loader.load()
        
        return docs

    def get_chunks(self, docs):
        """
        Gets chunks of the documents.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        
        return chunks

    def embed_and_store(self, chunks):
        """
        Generates embeddings and stores in ChromaDB.
        """
        self.db = Chroma.from_documents(
            documents=chunks, 
            embedding=self.bedrock_embeddings
        )

    def retrieve(self, query, k=5, show_chunks=False):
        """
        Retrieves relevant chunks from ChromaDB based on query and by appending by new line.
        """
        if not self.db:
            raise ValueError("ChromaDB is not initialized. Please call embed_and_store() first.")

        results = self.db.similarity_search(query, k=k)
        
        # Append the results by new line
        retrieved_text = "\n".join([result.page_content for result in results])
        
        if show_chunks:
            for i, result in enumerate(results):
                print('\n\n')
                print(f"Chunk {i+1}: {result.page_content}")
                print('#' * 100)
        
        return retrieved_text

if __name__ == "__main__":
    tavily_tool = TavilySearchTool(max_results=3)
    search_tool = tavily_tool.create_search_tool()

    search_results = search_tool.run(tool_input="What is the capital of France?")
    print("Search Results:", search_results)