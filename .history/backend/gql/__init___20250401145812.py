"""
GraphQL package for the Filipino Dictionary API.
"""

# Import at function-level to avoid circular dependencies
schema = None
graphql_view = None

def init_graphql():
    """Initialize GraphQL schema and view."""
    global schema, graphql_view
    from .schema import schema as _schema
    from .views import graphql_view as _graphql_view
    schema = _schema
    graphql_view = _graphql_view
    return schema, graphql_view

__all__ = ['schema', 'graphql_view', 'init_graphql'] 