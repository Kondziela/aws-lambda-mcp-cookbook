import json
import os
from typing import Any, Dict, List, Optional

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
    Lambda handler that queries S3 Vector for similar vectors using text input.

    Expected input:
    {
        "text": "search query text",
        "topK": 5,
        "metadata_filter": {
            "private": false,
            "price": {"$gte": 10.0}
        },
        "return_metadata": true,
        "return_distance": true
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
        top_k = data.get('topK', 5)
        metadata_filter = data.get('metadata_filter')
        return_metadata = data.get('return_metadata', True)
        return_distance = data.get('return_distance', True)

        if not text:
            return {'statusCode': 400, 'body': json.dumps({'error': 'Text parameter is required'})}

        if top_k <= 0 or top_k > 100:
            return {'statusCode': 400, 'body': json.dumps({'error': 'topK must be between 1 and 100'})}

        # Generate embedding for query text using Titan model
        query_embedding = generate_embedding(text)

        # Query S3 Vectors for similar vectors
        results = query_similar_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_metadata=return_metadata,
            return_distance=return_distance
        )

        metrics.add_metric(name='VectorQuery', unit=MetricUnit.Count, value=1)
        logger.info(f'Successfully queried vectors', extra={
            'query_text': text,
            'results_count': len(results.get('vectors', [])),
            'top_k': top_k
        })

        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Vector query completed successfully',
                'query_text': text,
                'results_count': len(results.get('vectors', [])),
                'vectors': results.get('vectors', [])
            })
        }

    except Exception as e:
        logger.error(f'Error processing query request: {str(e)}', exc_info=True)
        metrics.add_metric(name='VectorQueryError', unit=MetricUnit.Count, value=1)
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

        logger.info(f'Generated query embedding with {len(embedding)} dimensions')
        return embedding

    except Exception as e:
        logger.error(f'Error generating embedding: {str(e)}')
        raise


@tracer.capture_method
def query_similar_vectors(
    query_embedding: List[float],
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = True
) -> Dict[str, Any]:
    """Query S3 Vectors for similar vectors using query_vectors API"""
    try:
        # Get S3 Vector configuration from environment variables
        vector_bucket_name = os.environ.get('S3_VECTOR_BUCKET', 'your-vector-bucket')
        index_name = os.environ.get('S3_VECTOR_INDEX_NAME', 'default-index')

        # Convert embedding to float32 as required by S3 Vectors
        float32_embedding = [float(x) for x in query_embedding]

        # Prepare query parameters
        query_params = {
            'vectorBucketName': vector_bucket_name,
            'indexName': index_name,
            'topK': top_k,
            'queryVector': {
                'float32': float32_embedding
            },
            'returnMetadata': return_metadata,
            'returnDistance': return_distance
        }

        # Add metadata filter if provided
        if metadata_filter:
            query_params['filter'] = metadata_filter

        # Use S3 Vectors query_vectors API
        response = s3_vectors_client.query_vectors(**query_params)

        logger.info(f'Queried S3 Vectors: bucket={vector_bucket_name}, index={index_name}, results={len(response.get("vectors", []))}')

        # Process and format results
        formatted_vectors = []
        for vector in response.get('vectors', []):
            formatted_vector = {
                'key': vector.get('key'),
            }

            if return_distance and 'distance' in vector:
                formatted_vector['distance'] = vector['distance']

            if return_metadata and 'metadata' in vector:
                formatted_vector['metadata'] = vector['metadata']

            if 'data' in vector:
                formatted_vector['data'] = vector['data']

            formatted_vectors.append(formatted_vector)

        return {
            'vectors': formatted_vectors,
            'total_results': len(formatted_vectors)
        }

    except Exception as e:
        logger.error(f'Error querying S3 Vectors: {str(e)}')
        raise
