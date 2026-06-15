"""
Simple multi-turn API call to Amazon Bedrock using the Boto3 SDK. 
This script demonstrates how to send a message to a Bedrock model and receive a response, 
with optional streaming support.
"""

import boto3
import os
import argparse

from dotenv import load_dotenv
from termcolor import cprint

# Argument parsers.
parser = argparse.ArgumentParser()
parser.add_argument('--stream', action='store_true', help='Enable streaming mode')

args = parser.parse_args()

load_dotenv()

STREAM = args.stream

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

cprint("You can type 'exit' or 'quit' to end the chat.", 'green')
while True:
    cprint('USER: ', 'blue', attrs=['bold'], end='')
    user_input = input()
    if user_input.lower() in ['exit', 'quit']:
        print("Exiting the chat.")
        break
    
    # Append the user's message to the messages list.
    messages.append({
        'role': 'user',
        'content': [{'text': user_input}]
    })

    assistant_response = ''

    cprint('Assistant: ', 'magenta', attrs=['bold'], end='')

    if STREAM:
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
                cprint(stream_out, 'yellow', end='')
                assistant_response += stream_out

        print('\n')

    else:
        # Make the API call
        response = client.converse(
            modelId=model_id,
            messages=messages,
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