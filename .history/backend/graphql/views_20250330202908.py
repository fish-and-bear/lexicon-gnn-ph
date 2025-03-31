"""
GraphQL view handlers for the Filipino Dictionary API.
"""

from flask import request, jsonify
from graphql import graphql_sync
from .schema import schema

def graphql_view():
    if request.method == 'GET':
        return jsonify({
            'data': {
                '__schema': schema.introspect()
            }
        })

    data = request.get_json()
    query = data.get('query')
    variables = data.get('variables')

    result = graphql_sync(
        schema,
        query,
        variable_values=variables
    )

    return jsonify(result) 