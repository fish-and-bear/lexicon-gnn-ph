"""
GraphQL package for the Filipino Dictionary API.
"""

from .schema import schema
from .views import graphql_view

__all__ = ['schema', 'graphql_view'] 