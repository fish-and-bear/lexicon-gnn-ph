"""
GraphQL view handlers for the Filipino Dictionary API.
"""

from flask import request, jsonify
from graphql import graphql
from .schema import schema
import structlog
import json

# Set up logging
logger = structlog.get_logger(__name__)

def graphql_view():
    """Handle GraphQL requests."""
    try:
        if request.method == 'GET':
            # Handle introspection queries
            introspection_query = '''
                query IntrospectionQuery {
                    __schema {
                        types {
                            name
                            description
                            fields {
                                name
                                description
                                type {
                                    name
                                }
                            }
                        }
                    }
                }
            '''
            result = graphql(schema, introspection_query)
            return jsonify({
                'data': result.data,
                'errors': [str(error) for error in result.errors] if result.errors else None
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
            variable_values=variables,
            context_value={'request': request}
        )

        response_data = {
            'data': result.data,
            'errors': [{'message': str(error)} for error in result.errors] if result.errors else None
        }

        return jsonify(response_data), 200

    except Exception as e:
        logger.error("Error processing GraphQL request", error=str(e))
        return jsonify({
            'errors': [{'message': 'Internal server error', 'details': str(e)}]
        }), 500 