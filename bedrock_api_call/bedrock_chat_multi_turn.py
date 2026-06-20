"""
Simple multi-turn API call to Amazon Bedrock using the Boto3 SDK. 
This script demonstrates how to send a message to a Bedrock model and receive a response, 
with optional streaming support.

Support for image, video, and document inputs.
"""

import boto3
import os
import argparse

from dotenv import load_dotenv
from termcolor import cprint

# Argument parsers.
parser = argparse.ArgumentParser()
parser.add_argument('--stream', action='store_true', help='Enable streaming mode')
parser.add_argument(
    '--max_tokens', 
    type=int, 
    default=1024, 
    help='Maximum number of tokens to generate in the response'
)

args = parser.parse_args()

load_dotenv()

STREAM = args.stream
MAX_TOKENS = args.max_tokens
INFERENCE_CONFIG = {
    'maxTokens': MAX_TOKENS,
}

# Set the API key as an environment variable
os.environ['AWS_BEARER_TOKEN_BEDROCK'] = os.getenv('AWS_BEDROCK_API_KEY')

# Create the Bedrock client
client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

# Define the model and message
model_id = os.getenv('MODEL_ID')
# Empty messages list to hold the conversation history.
messages = []

def chat(user_input, model_id, chat_iter=0, stream=False):
    global messages
    
    # Deternine the type of input based on user input and append to messages list.
    # We take a look at the extensions for now. The file names should have a space before
    # the message with underscores and no spaces in the file name. The file name should be the last word in the message.
    # jpeg, jpg, png, mp4, avi, pdf, docx, txt, etc.

    # Extract file path and user message from the input. We assume the file path is the last word in the input.
    user_input_parts = user_input.split()
    if len(user_input_parts) > 1:
        file_path = user_input_parts[-1]
        user_message = ' '.join(user_input_parts[:-1])
    else:
        file_path = ''
        user_message = user_input

    if file_path.lower().endswith(('.jpeg', '.jpg', '.png')):
        # Read image and convert to bytes.
        with open(file_path, 'rb') as image_file:
            image_bytes = image_file.read()
        
        messages.append({
            'role': 'user',
            'content': [
                {
                    'image': {
                        'format': 'jpeg',
                        'source': {
                            'bytes': image_bytes
                        }
                    }
                },
                {'text': user_message}
            ]
        })
    elif file_path.lower().endswith(('.mp4', '.avi')):
        # Read video file and convert to bytes.
        with open(file_path, 'rb') as video_file:
            video_bytes = video_file.read()
        
        messages.append({
            'role': 'user',
            'content': [
                {
                    'video': {
                        'format': 'mp4',
                        'source': {
                            'bytes': video_bytes
                        }
                    }
                },
                {'text': user_message}
            ]
        })
    elif file_path.lower().endswith(('.pdf', '.docx', '.txt')):
        # Read document and convert to bytes.
        with open(file_path, 'rb') as document_file:
            document_bytes = document_file.read()
        
        messages.append({
            'role': 'user',
            'content': [
                {
                    'document': {
                        'format': 'pdf',
                        'name': f"User document {chat_iter}",
                        'source': {
                            'bytes': document_bytes
                        }
                    }
                },
                {'text': user_message}
            ]
        })
    

    # Append the user's message to the messages list if text only.
    else:
        messages.append({
            'role': 'user',
            'content': [{'text': user_input}]
        })

    assistant_response = ''

    cprint('Assistant: ', 'magenta', attrs=['bold'], end='')

    if stream:
        # print(f"CURRENT MESSGAGES SENT TO MODEL: {messages}\n")  # Debugging line to check the messages being sent to the model.
        # Make the API call with streaming
        response = client.converse_stream(
            modelId=model_id,
            messages=messages,
            inferenceConfig=INFERENCE_CONFIG
        )
    
        response_stream = response.get('stream')
        
        # Print the streamed response   
        for event in response_stream:
            if 'contentBlockDelta' in event:
                stream_out = event['contentBlockDelta']['delta']['text']
                cprint(stream_out, 'yellow', end='')
                assistant_response += stream_out

        print('\n')
    else:
        # Make the API call
        response = client.converse(
            modelId=model_id,
            messages=messages,
            inferenceConfig=INFERENCE_CONFIG
        )
        
        # Print the response
        assistant_response = response['output']['message']['content'][0]['text']
        cprint(assistant_response, 'yellow')

        print()

    # Append the assistant's response to the messages list to maintain conversation history.
    messages.append({
        'role': 'assistant',
        'content': [{'text': assistant_response}]
    })


if __name__ == '__main__':
    iter = 0
    while True:
        cprint('USER: ', 'blue', attrs=['bold'], end='')
        user_input = input()
        if user_input.lower() in ['exit', 'quit']:
            print("Exiting the chat.")
            break

        chat(user_input, model_id, chat_iter=iter, stream=STREAM)

        iter += 1