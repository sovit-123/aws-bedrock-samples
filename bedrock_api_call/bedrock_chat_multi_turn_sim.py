"""
Simple simulated multi-turn API call to Amazon Bedrock using the Boto3 SDK. 
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
# Simulated multi-turn example.
messages = [
    {
        'role': 'user',
        'content': [{'text': 'What is a neural network?'}]
    },
    {
        'role': 'assistant',
        'content': [{'text': 'A neural network is a machine learning model inspired by the structure of the brain. It consists of layers of interconnected nodes (neurons) that learn patterns from data.'}]
    },
    {
        'role': 'user',
        'content': [{'text': 'How does a neural network learn from data?'}]
    },
    {
        'role': 'assistant',
        'content': [{'text': 'Neural networks learn by adjusting their weights during training. They make predictions, measure the error using a loss function, and update the weights through backpropagation and gradient descent.'}]
    },
    {
        'role': 'user',
        'content': [{'text': 'What is backpropagation?'}]
    },
    {
        'role': 'assistant',
        'content': [{'text': 'Backpropagation is the process of computing how much each weight contributed to the prediction error. These gradients are then used to update the weights and improve future predictions.'}]
    },
    {
        'role': 'user',
        'content': [{'text': 'What are some common neural network architectures?'}]
    },
    {
        'role': 'assistant',
        'content': [{'text': 'Common architectures include feedforward neural networks, convolutional neural networks (CNNs) for images, recurrent neural networks (RNNs) for sequences, and transformers for language and multimodal tasks.'}]
    },
    {
        'role': 'user',
        'content': [{'text': 'What have we been talking about?'}]
    },
]

def chat(model_id, messages, stream=False):
    if stream:
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

    return response

if __name__ == '__main__':
    response = chat(model_id, messages, STREAM)