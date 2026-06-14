"""
Simple single-turn API call to Amazon Bedrock using the Boto3 SDK. 
This script demonstrates how to send a message to a Bedrock model and receive a response, 
with optional streaming support.
"""

import boto3
import os
import argparse

from dotenv import load_dotenv

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
messages = [
    {
        'role': 'user', 
        'content': [{'text': 'Hello! Who are you?'}]
    }
]

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
            print(event['contentBlockDelta']['delta']['text'], end="")

else:
    # Make the API call
    response = client.converse(
        modelId=model_id,
        messages=messages,
    )
    
    # Print the response
    print(response['output']['message']['content'][0]['text'])