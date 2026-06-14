from embeddings import (
    read_directory,
    read_pdf,
    read_text, 
    get_chunks, 
    embed_and_store, 
    retrieve
)
from dotenv import load_dotenv

import boto3
import os

load_dotenv()

# Set the API key as an environment variable
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')

# Create the Bedrock client
client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# Define the model and message
model_id = os.getenv('MODEL_ID')

messages = []

def handle_message(user_input, db):
    """
    Handles the user input and generates a response using Bedrock.
    """
    global messages

    # Create the messages for Bedrock API call
    messages = [
        {
            'role': 'system', 
            'content': [{'text': 'You are a helpful assistant. You are give context along with user question. Use that context to answer the question. If you do not know the answer, say you do not know. Always use all the relevant information from the context to answer the question.'}]
        },
    ]

    return messages
    

def main():
    # Ask the user to pass PDF, text, or folder path to read documents and create ChromaDB
    folder_path = input("Enter the folder path to read documents: ")
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
        docs_from_folder = read_directory(folder_path)
    else:
        docs_from_folder = []
    if file_path_pdf != "None":
        # Read PDF document
        docs_from_pdf = read_pdf(file_path_pdf)
    else:
        docs_from_pdf = []
    if file_path_text !="None":
        # Read text document
        docs_from_text = read_text(file_path_text)
    else:
        docs_from_text = []

    # Combine all documents
    all_docs = docs_from_folder + docs_from_pdf + docs_from_text

    chunks = get_chunks(all_docs)
    
    # Generate embeddings and store in ChromaDB
    db = embed_and_store(chunks)

    while True:
        user_input = input("User: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the chat.")
            break

        retrieved_text = retrieve(user_input, db, k=5, show_chunks=False)
        
        # Append the user's message to the messages list.
        user_message = f"Context: {retrieved_text}\n\nQuestion: {user_input}"
        messages.append({
            'role': 'user',
            'content': [{'text': user_message}]
        })

        assistant_response = ''

        print('Assistant: ', end='')

        # Make the API call with streaming
        response = client.converse_stream(
            modelId=model_id,
            messages=messages
        )
    
        response_stream = response.get('stream')
        
        # Print the streamed response   
        for event in response_stream:
            if 'contentBlockDelta' in event:
                stream_out = event['contentBlockDelta']['delta']['text']
                print(stream_out, end='')
                assistant_response += stream_out

        print('\n')

        # Append the assistant's response to the messages list to maintain conversation history.
        messages.append({
            'role': 'assistant',
            'content': [{'text': assistant_response}]
        })

if __name__ == '__main__':
    main()