from fastmcp import FastMCP

from service.logic.prompts.hld import hld_prompt
from service.logic.resources.profiles import get_profile_by_id
from service.logic.tools.math import add_two_numbers
from service.handlers.utils.observability import logger, metrics
from aws_lambda_powertools.metrics import MetricUnit
from datetime import date, datetime
from typing import Any, Dict, List, Optional
import json
import os

import boto3

mcp: FastMCP = FastMCP(name='mcp-lambda-server')

# Initialize AWS clients for vector search
bedrock_client = boto3.client('bedrock-runtime', region_name=os.environ.get('BEDROCK_REGION', 'eu-central-1'))
s3_vectors_client = boto3.client('s3vectors', region_name=os.environ.get('S3_VECTOR_REGION', 'eu-central-1'))


@mcp.tool
def math(a: int, b: int) -> int:
    """Add two numbers together"""
    logger.info('using math tool', extra={'a': a, 'b': b})
    return add_two_numbers(a, b)


@mcp.tool()
def imprezy_wroclaw(opis: str, date: date = None, miasto: Optional[str] = None, cena: Optional[float] = None, top_k: int = 5) -> Dict[str, Any]:
    """Search for events in Wroclaw based on description, date, city and price

    Args:
        opis: Description of the event to search for (required)
        date: Event date to filter by (optional, defaults to today)
        miasto: City to search in (optional)
        cena: Price filter for events (optional, None=ignore price, 0=free events only)
        top_k: Number of results to return (1-100, default: 5)

    Returns:
        Dictionary containing search results with similar events
    """
    try:
        if not opis:
            raise ValueError('Opis parameter is required')

        # Set default date to today if not provided
        if date is None:
            date = datetime.now().date()

        if top_k <= 0 or top_k > 100:
            raise ValueError('topK must be between 1 and 100')

        # Generate embedding for query text using Titan model
        query_embedding = _generate_embedding(opis)

        # Create metadata filter based on parameters
        metadata_filter = {}

        # Convert date to string format for filtering
        metadata_filter['date'] = date.isoformat()

        if miasto:
            metadata_filter['city'] = miasto

        if cena is not None:
            if cena == 0:
                # Looking for free events (price equals 0)
                metadata_filter['price'] = 0
            else:
                # Looking for events up to specified price
                metadata_filter['price'] = {'$lte': cena}

        # Query S3 Vectors for similar vectors
        results = _query_with_s3vectors(query_embedding, top_k, metadata_filter, True, True)

        # Save session data
        mcp.set_session(data={'last_search': {'query': opis, 'results_count': len(results.get('vectors', [])), 'top_k': top_k}})

        metrics.add_metric(name='VectorSearchTool', unit=MetricUnit.Count, value=1)
        logger.info(
            f'Event search completed',
            extra={
                'query_text': opis,
                'date': date.isoformat(),
                'miasto': miasto,
                'cena': cena,
                'results_count': len(results.get('vectors', [])),
                'top_k': top_k,
            },
        )

        return {
            'message': 'Event search completed successfully',
            'query_text': opis,
            'date': date.isoformat(),
            'miasto': miasto,
            'cena': cena,
            'results_count': len(results.get('vectors', [])),
            'vectors': results.get('vectors', []),
            'query_type': results.get('query_type', 'unknown'),
        }

    except Exception as e:
        logger.error(f'Error in event search tool: {str(e)}', exc_info=True)
        metrics.add_metric(name='VectorSearchError', unit=MetricUnit.Count, value=1)
        raise


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
    formatted_vectors = [
        {
            'key': vector.get('key'),
            **({'distance': vector['distance']} if return_distance and 'distance' in vector else {}),
            **({'metadata': vector['metadata']} if return_metadata and 'metadata' in vector else {}),
            **({'data': vector['data']} if 'data' in vector else {}),
        }
        for vector in response.get('vectors', [])
    ]

    return {'vectors': formatted_vectors, 'total_results': len(formatted_vectors), 'query_type': 's3vectors'}


# Dynamic resource template
@mcp.resource('users://{user_id}/profile')
def get_profile(user_id: int):
    """Fetch user profile by user ID."""
    logger.info('fetching user profile', extra={'user_id': user_id})
    return get_profile_by_id(user_id)


@mcp.prompt()
def generate_serverless_design_prompt(design_requirements: str) -> str:
    """Generate a serverless design prompt based on the provided design requirements."""
    logger.info('generating serverless design prompt', extra={'design_requirements': design_requirements})
    return hld_prompt(design_requirements)


app = mcp.http_app(transport='http', stateless_http=True, json_response=True)
