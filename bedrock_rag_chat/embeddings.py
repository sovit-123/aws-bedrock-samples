"""
Script to generate embeddings by taking file path or folder path.
The folder can contain PDFs and text files.
Same extension goes for the indivual files paths as well.
Using ChromaDB and LangChain.

Functions present.
`read_directory()` => gets all documents from the folder path.
`read_pdf()` => gets PDF documents.
`read_text()` => gets text documents.
`get_chunks()` => gets chunks of the documents.
`embed_and_store()` => generates embeddings and stores in ChromaDB.
`retrieve()` => retrieves relevant chunks from ChromaDB based on query and by appending by new line.
"""

import langchain
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter  

def read_directory(folder_path):
    """
    Reads all documents from the folder path.
    """
    loader = DirectoryLoader(folder_path)
    docs = loader.load()
    print(f"Number of documents read from folder: {len(docs)}")
    
    return docs

def read_pdf(file_path):
    """
    Reads PDF documents.
    """
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    return docs

def read_text(file_path):
    """
    Reads text documents.
    """
    loader = TextLoader(file_path)
    docs = loader.load()
    
    return docs

def get_chunks(docs):
    """
    Gets chunks of the documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    return chunks

def embed_and_store(chunks):
    """
    Generates embeddings and stores in ChromaDB.
    """

    # Create embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


    # Create ChromaDB instance
    db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings, 
        # persist_directory='chroma_db'
    )
    
    return db

def retrieve(query, db, k=5, show_chunks=False):
    """
    Retrieves relevant chunks from ChromaDB based on query and by appending by new line.
    """
    results = db.similarity_search(query, k=k)
    
    # Append the results by new line
    retrieved_text = "\n".join([result.page_content for result in results])
    
    if show_chunks:
        for i, result in enumerate(results):
            print(f"Chunk {i+1}: {result.page_content}")
    
    return retrieved_text

def __main__():
    """
    Main function to test the above functions.
    """
    # Example usage
    folder_path = "input"
    file_path_pdf = "None"
    file_path_text = "None"
    
    # Read documents from folder
    docs_from_folder = read_directory(folder_path)
    
    # Read PDF document
    if file_path_pdf != "None":
        docs_from_pdf = read_pdf(file_path_pdf)
    else:
        docs_from_pdf = []
    
    # Read text document
    if file_path_text != "None":
        docs_from_text = read_text(file_path_text)
    else:
        docs_from_text = []
    
    # Combine all documents
    all_docs = docs_from_folder + docs_from_pdf + docs_from_text

    chunks = get_chunks(all_docs)
    
    # Generate embeddings and store in ChromaDB
    db = embed_and_store(chunks)
    
    # Retrieve relevant chunks based on query
    query = "Elon Musk"
    retrieved_text = retrieve(query, db, k=5, show_chunks=True)
    
    print(retrieved_text)

if __name__ == "__main__":
    __main__()