from aws_lambda_env_modeler import init_environment_variables
from aws_lambda_powertools.logging import correlation_paths
from aws_lambda_powertools.metrics import MetricUnit
from aws_lambda_powertools.utilities.typing import LambdaContext

from service.handlers.models.env_vars import McpHandlerEnvVars
from service.handlers.utils.authentication import authenticate
from service.handlers.utils.mcp import mcp
from service.handlers.utils.observability import logger, metrics, tracer
from service.logic.tools.math import add_two_numbers
import json
import os
import math
from typing import Any, Dict, List, Optional

import boto3


# Initialize AWS clients for vector search
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get('BEDROCK_REGION', 'eu-central-1'))

# Try to initialize s3vectors client, fallback to regular S3 if not available
try:
    s3_vectors_client = boto3.client('s3vectors', region_name=os.environ.get('S3_VECTOR_REGION', 'eu-central-1'))
    S3_VECTORS_AVAILABLE = True
    logger.info('S3 Vectors service is available')
except Exception as e:
    logger.warning(f'S3 Vectors service not available, falling back to regular S3: {str(e)}')
    s3_client = boto3.client('s3', region_name=os.environ.get('S3_VECTOR_REGION', 'eu-central-1'))
    S3_VECTORS_AVAILABLE = False


@mcp.tool()
def math(a: int, b: int) -> int:
    """Add two numbers together"""
    # Uncomment the following line if you want to use session data
    # session_data: Optional[SessionData] = mcp.get_session()

    # call logic layer
    result = add_two_numbers(a, b)

    # save session data
    mcp.set_session(data={'result': result})

    metrics.add_metric(name='ValidMcpEvents', unit=MetricUnit.Count, value=1)
    return result


@mcp.tool()
def search_vectors(
    text: str, top_k: int = 5, metadata_filter: Optional[Dict[str, Any]] = None, return_metadata: bool = True, return_distance: bool = True
) -> Dict[str, Any]:
    """Search for similar vectors in S3 Vector database using text input

    Args:
        text: The search query text to find similar vectors
        top_k: Number of results to return (1-100, default: 5)
        metadata_filter: Optional metadata filter (e.g., {"private": false, "price": {"$gte": 10.0}})
        return_metadata: Whether to include metadata in results (default: True)
        return_distance: Whether to include similarity distances (default: True)

    Returns:
        Dictionary containing search results with similar vectors
    """
    try:
        if not text:
            raise ValueError('Text parameter is required')

        if top_k <= 0 or top_k > 100:
            raise ValueError('topK must be between 1 and 100')

        # Generate embedding for query text using Titan model
        query_embedding = _generate_embedding(text)

        # Query S3 Vectors for similar vectors
        results = _query_similar_vectors(
            query_embedding=query_embedding,
            top_k=top_k,
            metadata_filter=metadata_filter,
            return_metadata=return_metadata,
            return_distance=return_distance,
        )

        # Save session data
        mcp.set_session(data={'last_search': {'query': text, 'results_count': len(results.get('vectors', [])), 'top_k': top_k}})

        metrics.add_metric(name='VectorSearchTool', unit=MetricUnit.Count, value=1)
        logger.info(f'Vector search completed', extra={'query_text': text, 'results_count': len(results.get('vectors', [])), 'top_k': top_k})

        return {
            'message': 'Vector search completed successfully',
            'query_text': text,
            'results_count': len(results.get('vectors', [])),
            'vectors': results.get('vectors', []),
            'query_type': results.get('query_type', 'unknown'),
        }

    except Exception as e:
        logger.error(f'Error in vector search tool: {str(e)}', exc_info=True)
        metrics.add_metric(name='VectorSearchError', unit=MetricUnit.Count, value=1)
        raise


@tracer.capture_method
def _generate_embedding(text: str) -> List[float]:
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
def _query_similar_vectors(
    query_embedding: List[float],
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> Dict[str, Any]:
    """Query S3 Vectors for similar vectors using query_vectors API or fallback to S3 simulation"""
    try:
        if S3_VECTORS_AVAILABLE:
            return _query_with_s3vectors(query_embedding, top_k, metadata_filter, return_metadata, return_distance)
        else:
            return _query_with_s3_fallback(query_embedding, top_k, metadata_filter, return_metadata, return_distance)

    except Exception as e:
        logger.error(f'Error querying vectors: {str(e)}')
        raise


@tracer.capture_method
def _query_with_s3vectors(
    query_embedding: List[float],
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> Dict[str, Any]:
    """Query using S3 Vectors service"""
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
        'queryVector': {'float32': float32_embedding},
        'returnMetadata': return_metadata,
        'returnDistance': return_distance,
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
        formatted_vector = {'key': vector.get('key')}

        if return_distance and 'distance' in vector:
            formatted_vector['distance'] = vector['distance']

        if return_metadata and 'metadata' in vector:
            formatted_vector['metadata'] = vector['metadata']

        if 'data' in vector:
            formatted_vector['data'] = vector['data']

        formatted_vectors.append(formatted_vector)

    return {'vectors': formatted_vectors, 'total_results': len(formatted_vectors), 'query_type': 's3vectors'}


@tracer.capture_method
def _query_with_s3_fallback(
    query_embedding: List[float],
    top_k: int,
    metadata_filter: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = True,
) -> Dict[str, Any]:
    """Fallback: Query vectors stored in regular S3 bucket using cosine similarity"""
    bucket_name = os.environ.get('S3_VECTOR_BUCKET', 'your-vector-bucket')

    try:
        # List all vector files in the bucket
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix='vectors/')

        if 'Contents' not in response:
            logger.warning(f'No vectors found in bucket {bucket_name}')
            return {'vectors': [], 'total_results': 0, 'query_type': 's3_fallback'}

        vector_similarities = []

        for obj in response['Contents']:
            try:
                # Get vector data from S3
                vector_obj = s3_client.get_object(Bucket=bucket_name, Key=obj['Key'])
                vector_data = json.loads(vector_obj['Body'].read())

                stored_vector = vector_data.get('vector', [])
                stored_metadata = vector_data.get('metadata', {})

                # Apply metadata filter if provided
                if metadata_filter and not _matches_filter(stored_metadata, metadata_filter):
                    continue

                # Calculate cosine similarity
                similarity = _cosine_similarity(query_embedding, stored_vector)

                result_vector = {'key': obj['Key'].replace('vectors/', '').replace('.json', ''), 'similarity': similarity}

                if return_distance:
                    # Convert similarity to distance (1 - similarity for cosine)
                    result_vector['distance'] = 1.0 - similarity

                if return_metadata:
                    result_vector['metadata'] = stored_metadata

                vector_similarities.append(result_vector)

            except Exception as e:
                logger.warning(f'Error processing vector {obj["Key"]}: {str(e)}')
                continue

        # Sort by similarity (highest first) and take top_k
        vector_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_vectors = vector_similarities[:top_k]

        logger.info(f'Queried S3 (fallback): bucket={bucket_name}, results={len(top_vectors)}')

        return {'vectors': top_vectors, 'total_results': len(top_vectors), 'query_type': 's3_fallback'}

    except Exception as e:
        logger.error(f'Error in S3 fallback query: {str(e)}')
        return {'vectors': [], 'total_results': 0, 'query_type': 's3_fallback', 'error': str(e)}


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if len(vec1) != len(vec2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(a * a for a in vec2))

    if magnitude1 == 0.0 or magnitude2 == 0.0:
        return 0.0

    return dot_product / (magnitude1 * magnitude2)


def _matches_filter(metadata: Dict[str, Any], filter_criteria: Dict[str, Any]) -> bool:
    """Check if metadata matches the filter criteria"""
    for key, criteria in filter_criteria.items():
        if key not in metadata:
            return False

        value = metadata[key]

        if isinstance(criteria, dict):
            # Handle operators like {"$gte": 10.0}
            for operator, target_value in criteria.items():
                if operator == '$gte' and value < target_value:
                    return False
                elif operator == '$lte' and value > target_value:
                    return False
                elif operator == '$gt' and value <= target_value:
                    return False
                elif operator == '$lt' and value >= target_value:
                    return False
                elif operator == '$eq' and value != target_value:
                    return False
                elif operator == '$ne' and value == target_value:
                    return False
        else:
            # Direct value comparison
            if value != criteria:
                return False

    return True


@init_environment_variables(model=McpHandlerEnvVars)
@logger.inject_lambda_context(correlation_id_path=correlation_paths.API_GATEWAY_REST)
@metrics.log_metrics
@tracer.capture_lambda_handler(capture_response=False)
def lambda_handler(event: dict, context: LambdaContext) -> dict:
    authenticate(event, context)
    return mcp.handle_request(event, context)
