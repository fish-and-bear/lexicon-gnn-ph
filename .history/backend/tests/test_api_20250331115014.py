"""
Comprehensive test suite for the Filipino Dictionary API.
Tests all endpoints, error handling, and edge cases.
"""

import pytest
import json
from flask import url_for
from datetime import datetime, timedelta
from unittest.mock import patch
from models import Word, Definition, Etymology, Relation
from dictionary_manager import RelationshipType, BaybayinRomanizer

class TestAPI:
    def setup_method(self):
        """Setup test data before each test."""
        self.test_word = {
            "lemma": "aso",
            "language_code": "tl",
            "has_baybayin": True,
            "baybayin_form": "ᜀᜐᜓ",
            "definitions": [
                {
                    "definition_text": "domesticated canine",
                    "part_of_speech": "n",
                    "examples": [{"text": "Ang aso ay tumatahol.", "translation": "The dog is barking."}]
                }
            ]
        }
        
        self.test_etymology = {
            "etymology_text": "From Proto-Austronesian *asu",
            "language_codes": ["poz"],
            "confidence_score": 0.9
        }

    def test_get_word(self, client, db_session):
        """Test word retrieval endpoint."""
        # Test successful retrieval
        response = client.get('/api/v2/words/aso')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['lemma'] == 'aso'
        assert 'definitions' in data
        assert 'data_completeness' in data
        
        # Test word not found
        response = client.get('/api/v2/words/nonexistentword')
        assert response.status_code == 404
        assert 'suggestions' in json.loads(response.data)
        
        # Test with special characters
        response = client.get('/api/v2/words/áso')
        assert response.status_code == 200
        
        # Test with Baybayin
        response = client.get('/api/v2/words/ᜀᜐᜓ')
        assert response.status_code == 200

    def test_search(self, client, db_session):
        """Test search functionality."""
        # Test basic search
        response = client.get('/api/v2/search?q=aso')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'words' in data
        assert 'facets' in data
        assert 'suggestions' in data
        
        # Test search modes
        modes = ['exact', 'baybayin', 'phonetic', 'etymology', 'semantic', 'root', 'affixed']
        for mode in modes:
            response = client.get(f'/api/v2/search?q=aso&mode={mode}')
            assert response.status_code == 200
            
        # Test filters
        response = client.get('/api/v2/search?q=aso&language=tl&pos=n&min_quality=0.5')
        assert response.status_code == 200
        
        # Test pagination
        response = client.get('/api/v2/search?q=aso&limit=10&offset=0')
        assert response.status_code == 200
        
        # Test sorting
        sorts = ['relevance', 'alphabetical', 'created', 'updated', 'quality']
        for sort in sorts:
            response = client.get(f'/api/v2/search?q=aso&sort={sort}')
            assert response.status_code == 200

    def test_word_relations(self, client, db_session):
        """Test word relationships endpoint."""
        # Test basic relations
        response = client.get('/api/v2/words/aso/relations')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'categories' in data
        
        # Test with depth parameter
        response = client.get('/api/v2/words/aso/relations?depth=2')
        assert response.status_code == 200
        
        # Test relation types
        for rel_type in RelationshipType:
            response = client.get(f'/api/v2/words/aso/relations?type={rel_type.value}')
            assert response.status_code == 200

    def test_word_etymology(self, client, db_session):
        """Test etymology endpoint."""
        # Test basic etymology
        response = client.get('/api/v2/words/aso/etymology')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'etymologies' in data
        
        # Test etymology tree
        response = client.get('/api/v2/words/1/etymology/tree')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'etymology_tree' in data

    def test_word_affixations(self, client, db_session):
        """Test affixation endpoint."""
        # Test basic affixations
        response = client.get('/api/v2/words/aso/affixations')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'affixations' in data
        
        # Test affixation tree
        response = client.get('/api/v2/words/aso/affixation_tree')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'tree' in data

    def test_word_pronunciation(self, client, db_session):
        """Test pronunciation endpoint."""
        response = client.get('/api/v2/words/aso/pronunciation')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'pronunciations' in data

    def test_baybayin_processing(self, client, db_session):
        """Test Baybayin processing endpoint."""
        # Test conversion
        payload = {"text": "aso"}
        response = client.post('/api/v2/baybayin/process', 
                             data=json.dumps(payload),
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'results' in data
        
        # Test validation
        payload = {"text": "ᜀᜐᜓ"}
        response = client.post('/api/v2/baybayin/process',
                             data=json.dumps(payload),
                             content_type='application/json')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert data['results'][0]['is_valid']

    def test_statistics(self, client, db_session):
        """Test statistics endpoint."""
        response = client.get('/api/v2/statistics')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'total_words' in data
        assert 'words_by_language' in data
        assert 'words_by_pos' in data
        assert 'words_with_baybayin' in data

    def test_semantic_network(self, client, db_session):
        """Test semantic network endpoint."""
        response = client.get('/api/v2/words/aso/semantic_network')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'nodes' in data
        assert 'edges' in data
        assert 'metadata' in data

    def test_relation_graph(self, client, db_session):
        """Test relation graph endpoint."""
        response = client.get('/api/v2/words/1/relations/graph')
        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'nodes' in data
        assert 'edges' in data
        assert 'metadata' in data

    def test_error_handling(self, client, db_session):
        """Test error handling."""
        # Test 400 Bad Request
        response = client.get('/api/v2/search')  # Missing required parameter
        assert response.status_code == 400
        
        # Test 404 Not Found
        response = client.get('/api/v2/words/nonexistentword')
        assert response.status_code == 404
        
        # Test 500 Internal Server Error
        with patch('models.Word.query') as mock_query:
            mock_query.side_effect = Exception("Database error")
            response = client.get('/api/v2/words/aso')
            assert response.status_code == 500

    def test_performance(self, client, db_session):
        """Test API performance."""
        import time
        
        # Test response time for main endpoints
        endpoints = [
            '/api/v2/words/aso',
            '/api/v2/search?q=aso',
            '/api/v2/words/aso/relations',
            '/api/v2/words/aso/etymology',
            '/api/v2/statistics'
        ]
        
        for endpoint in endpoints:
            start_time = time.time()
            response = client.get(endpoint)
            duration = time.time() - start_time
            assert duration < 1.0  # Response should be under 1 second
            assert response.status_code == 200

    def test_caching(self, client, db_session):
        """Test caching functionality."""
        # Test cache hit
        response1 = client.get('/api/v2/words/aso')
        response2 = client.get('/api/v2/words/aso')
        assert response1.data == response2.data
        
        # Test cache invalidation
        client.put('/api/v2/words/aso', json=self.test_word)
        response3 = client.get('/api/v2/words/aso')
        assert response3.data != response1.data

    def test_validation(self, client, db_session):
        """Test input validation."""
        # Test invalid language code
        response = client.get('/api/v2/search?q=aso&language=invalid')
        assert response.status_code == 400
        
        # Test invalid search mode
        response = client.get('/api/v2/search?q=aso&mode=invalid')
        assert response.status_code == 400
        
        # Test invalid pagination
        response = client.get('/api/v2/search?q=aso&limit=1000')  # Exceeds max
        assert response.status_code == 400

    def test_edge_cases(self, client, db_session):
        """Test edge cases."""
        # Test empty string
        response = client.get('/api/v2/words/ ')
        assert response.status_code == 404
        
        # Test very long word
        long_word = 'a' * 1000
        response = client.get(f'/api/v2/words/{long_word}')
        assert response.status_code == 404
        
        # Test special characters
        special_chars = '!@#$%^&*()'
        response = client.get(f'/api/v2/words/{special_chars}')
        assert response.status_code == 404
        
        # Test Unicode characters
        unicode_chars = 'ñáéíóú'
        response = client.get(f'/api/v2/words/{unicode_chars}')
        assert response.status_code == 404

if __name__ == '__main__':
    pytest.main(['-v', 'test_api.py']) 