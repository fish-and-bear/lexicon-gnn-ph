"""
GraphQL view handlers for the Filipino Dictionary API.
"""

from flask import request, jsonify
from graphql import graphql
from .schema import schema
import structlog

# Set up logging
logger = structlog.get_logger(__name__)

def graphql_view():
    """Handle GraphQL requests."""
    try:
        if request.method == 'GET':
            # Handle introspection queries
            return jsonify({
                'data': {
                    '__schema': schema.introspect()
                }
            })

        # Handle POST requests with queries
        data = request.get_json()
        if not data:
            return jsonify({
                'errors': [{'message': 'No query provided'}]
            }), 400

        query = data.get('query')
        variables = data.get('variables')

        if not query:
            return jsonify({
                'errors': [{'message': 'Query is required'}]
            }), 400

        logger.info("Processing GraphQL query", query=query, variables=variables)
        
        result = graphql(
            schema,
            query,
            variable_values=variables
        )

        if result.errors:
            logger.error("GraphQL errors", errors=result.errors)
            return jsonify(result), 200  # GraphQL always returns 200 even with errors

        logger.info("Query executed successfully")
        return jsonify(result)

    except Exception as e:
        logger.error("Error processing GraphQL request", error=str(e))
        return jsonify({
            'errors': [{'message': 'Internal server error'}]
        }), 500 