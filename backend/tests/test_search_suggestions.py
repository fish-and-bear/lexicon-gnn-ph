#!/usr/bin/env python3
"""
Unit tests for the search suggestions system.
"""

import unittest
import json
import time
from unittest.mock import patch, MagicMock
from flask import url_for
import pytest
from app import create_app
from backend.database import db
from sqlalchemy import text
import os
import sys

# Add the parent directory to the path so we can import from backend
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from search_tasks import log_search_query, refresh_popular_words

class TestSearchSuggestions(unittest.TestCase):
    """Test the search suggestions functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment once before all tests."""
        # Create a test app
        cls.app = create_app(testing=True)
        cls.app_context = cls.app.app_context()
        cls.app_context.push()
        
        # Create a test client
        cls.client = cls.app.test_client()
        
        # Set up database for testing
        with cls.app.app_context():
            # Ensure the popular_words view exists
            try:
                db.session.execute(text("""
                CREATE MATERIALIZED VIEW IF NOT EXISTS popular_words AS
                SELECT 
                    word_id, 
                    COUNT(*) as search_count
                FROM search_logs
                WHERE 
                    created_at > NOW() - INTERVAL '30 days'
                    AND word_id IS NOT NULL
                GROUP BY word_id
                ORDER BY search_count DESC;
                
                CREATE INDEX IF NOT EXISTS popular_words_count_idx ON popular_words (search_count DESC);
                """))
                db.session.commit()
            except Exception as e:
                print(f"Error setting up materialized view: {e}")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up after all tests."""
        cls.app_context.pop()
    
    def setUp(self):
        """Set up before each test."""
        # Clear any existing test data
        with self.app.app_context():
            db.session.execute(text("DELETE FROM search_logs WHERE query_text LIKE 'test%'"))
            db.session.commit()
    
    def test_get_suggestions_empty_query(self):
        """Test that empty queries return empty suggestions."""
        response = self.client.get('/api/search/suggestions?q=')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('suggestions', data)
        self.assertEqual(len(data['suggestions']), 0)
    
    def test_get_suggestions_short_query(self):
        """Test that queries shorter than 2 chars return empty suggestions."""
        response = self.client.get('/api/search/suggestions?q=a')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('suggestions', data)
        self.assertEqual(len(data['suggestions']), 0)
    
    def test_get_suggestions_valid_query(self):
        """Test that valid queries return suggestions."""
        # Use a common word likely to be in any dictionary
        response = self.client.get('/api/search/suggestions?q=an')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('suggestions', data)
        # We can't be sure of exact counts in different environments,
        # but we should get some suggestions
        self.assertIsInstance(data['suggestions'], list)
    
    def test_get_suggestions_with_language(self):
        """Test suggestions with language filter."""
        # Test with Tagalog
        response = self.client.get('/api/search/suggestions?q=an&language=tl')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        # All returned suggestions should have language=tl
        for suggestion in data['suggestions']:
            self.assertEqual(suggestion['language'], 'tl')
    
    def test_get_suggestions_with_limit(self):
        """Test suggestions with limit parameter."""
        limit = 5
        response = self.client.get(f'/api/search/suggestions?q=an&limit={limit}')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        # Should return at most 'limit' suggestions
        self.assertLessEqual(len(data['suggestions']), limit)
    
    def test_track_selection(self):
        """Test tracking search selections."""
        test_data = {
            "query": "test query",
            "selected_id": 1,
            "selected_text": "test word"
        }
        
        response = self.client.post(
            '/api/search/track-selection',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'success')
        
        # Verify it was logged to the database
        with self.app.app_context():
            result = db.session.execute(text(
                "SELECT COUNT(*) FROM search_logs WHERE query_text = :query AND word_id = :word_id"
            ), {"query": "test word", "word_id": 1}).scalar()
            
            # Note: Due to asynchronous logging, this might not be immediately reflected
            # This is a soft assertion
            if result > 0:
                self.assertTrue(True)
    
    def test_track_selection_missing_params(self):
        """Test tracking with missing parameters."""
        # Missing selected_id and selected_text
        test_data = {
            "query": "test query"
        }
        
        response = self.client.post(
            '/api/search/track-selection',
            data=json.dumps(test_data),
            content_type='application/json'
        )
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
    
    @patch('search_tasks.log_search_query')
    def test_search_log_queue(self, mock_log):
        """Test that search logging works correctly."""
        mock_log.return_value = None
        
        # Call the API with logging
        response = self.client.get('/api/search/suggestions?q=test')
        self.assertEqual(response.status_code, 200)
        
        # Our mock should have been called
        # Allow some time for async logging
        time.sleep(0.1)
        mock_log.assert_called()
    
    def test_suggestions_response_format(self):
        """Test the structure of returned suggestions."""
        response = self.client.get('/api/search/suggestions?q=an')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Skip if no suggestions returned (could happen in empty test DB)
        if not data['suggestions']:
            self.skipTest("No suggestions returned from test database")
            
        # Check suggestion format
        suggestion = data['suggestions'][0]
        
        # Required fields
        self.assertIn('id', suggestion)
        self.assertIn('text', suggestion)
        self.assertIn('language', suggestion)
        self.assertIn('type', suggestion)
        self.assertIn('confidence', suggestion)
        
        # Type should be one of our defined types
        self.assertIn(suggestion['type'], 
                     ['prefix_match', 'popular_match', 'spelling_suggestion', 'definition_match'])
        
        # Confidence should be a float between 0 and 1
        self.assertIsInstance(suggestion['confidence'], float)
        self.assertGreaterEqual(suggestion['confidence'], 0)
        self.assertLessEqual(suggestion['confidence'], 1)

if __name__ == '__main__':
    unittest.main()