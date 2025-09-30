import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List

import boto3
from aws_lambda_powertools import Logger, Tracer, Metrics
from aws_lambda_powertools.metrics import MetricUnit

logger = Logger()
tracer = Tracer()
metrics = Metrics()

# Initialize AWS clients
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get('BEDROCK_REGION', 'eu-central-1'))
s3_vectors_client = boto3.client('s3vectors', region_name=os.environ.get('S3_VECTOR_REGION', 'eu-central-1'))


@tracer.capture_lambda_handler
@logger.inject_lambda_context
@metrics.log_metrics
def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Lambda handler that processes text and converts it to vector embeddings using Titan model.
    Stores the result in S3 Vector.

    Expected input:
    {
        "text": "some text to embed",
        "date": "2023-12-01",
        "price": 100.50,
        "private": true
    }
    """
    try:
        # Parse input
        body = event.get('body', '{}')
        if isinstance(body, str):
            data = json.loads(body)
        else:
            data = body

        text = data.get('text', '')
        date = data.get('date', datetime.now().strftime('%d-%m-%Y'))
        price = data.get('cena', 0.0)
        miasto = data.get('miasto', 'Wrocław')

        if not text:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Text parameter is required'})}

        # Generate embedding using Titan model
        embedding = generate_embedding(text)

        # Create metadata
        metadata = {
            'id': str(uuid.uuid4()),
            'text': text,
            'date': date,
            'price': float(price),
            'city': miasto,
            'embedding_model': os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-embed-text-v2:0'),
            'embedding_dimension': len(embedding),
            'created_at': datetime.now().isoformat(),
        }

        # Store in S3 Vector (simulated - replace with actual S3 Vector implementation)
        result = store_vector_in_s3(embedding, metadata)

        metrics.add_metric(name='EmbeddingGenerated', unit=MetricUnit.Count, value=1)
        logger.info('Successfully processed text embedding', extra={'metadata': metadata})

        return {
            'statusCode': 200,
            'body': json.dumps(
                {
                    'message': 'Text successfully converted to vector and stored',
                    'id': metadata['id'],
                    'embedding_dimension': metadata['embedding_dimension'],
                    'result': result,
                }
            ),
        }

    except Exception as e:
        logger.error(f'Error processing request: {str(e)}', exc_info=True)
        metrics.add_metric(name='EmbeddingError', unit=MetricUnit.Count, value=1)
        return {'statusCode': 500, 'body': json.dumps({'error': f'Internal server error: {str(e)}'})}


@tracer.capture_method
def generate_embedding(text: str) -> List[float]:
    """Generate embedding using Amazon Titan model"""
    try:
        model_id = os.environ.get('BEDROCK_MODEL_ID', 'amazon.titan-embed-text-v2:0')

        # Prepare request body for Titan embedding model
        request_body = {'inputText': text, 'dimensions': 1024, 'normalize': True}

        # Invoke Bedrock model
        response = bedrock_client.invoke_model(
            modelId=model_id, body=json.dumps(request_body), contentType='application/json', accept='application/json'
        )

        # Parse response
        response_body = json.loads(response['body'].read())
        embedding = response_body.get('embedding', [])

        if not embedding or len(embedding) != 1024:
            raise ValueError(f'Invalid embedding response: expected 1024 dimensions, got {len(embedding)}')

        logger.info(f'Generated embedding with {len(embedding)} dimensions')
        return embedding

    except Exception as e:
        logger.error(f'Error generating embedding: {str(e)}')
        raise


@tracer.capture_method
def store_vector_in_s3(embedding: List[float], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Store vector and metadata in S3 Vector using put_vectors API"""
    try:
        # Get S3 Vector configuration from environment variables
        vector_bucket_name = os.environ.get('S3_VECTOR_BUCKET', 'your-vector-bucket')
        index_name = os.environ.get('S3_VECTOR_INDEX_NAME', 'default-index')

        # Convert embedding to float32 as required by S3 Vectors
        float32_embedding = [float(x) for x in embedding]

        # Prepare vector for S3 Vectors API
        vector_key = f'vector-{metadata["id"]}'

        # Use S3 Vectors put_vectors API
        s3_vectors_client.put_vectors(
            vectorBucketName=vector_bucket_name,
            indexName=index_name,
            vectors=[
                {
                    'key': vector_key,
                    'data': {'float32': float32_embedding},
                    'metadata': {
                        'text': metadata['text'],
                        'date': metadata['date'],
                        'price': metadata['price'],
                        'private': metadata['private'],
                        'embedding_model': metadata['embedding_model'],
                        'embedding_dimension': metadata['embedding_dimension'],
                        'created_at': metadata['created_at'],
                    },
                }
            ],
        )

        logger.info(f'Stored vector in S3 Vectors: bucket={vector_bucket_name}, index={index_name}, key={vector_key}')

        return {
            'vector_bucket': vector_bucket_name,
            'index_name': index_name,
            'vector_key': vector_key,
            'embedding_dimension': len(float32_embedding),
        }

    except Exception as e:
        logger.error(f'Error storing vector in S3 Vectors: {str(e)}')
        raise
