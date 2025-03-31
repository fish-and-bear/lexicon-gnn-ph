import unittest
import requests
import json
import logging
from typing import Dict, Any
from datetime import datetime, timezone
from flask import Flask
from flask_testing import TestCase
from run_tests import TimedTestCase

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestFilipinoDictionaryAPI(TimedTestCase, TestCase):
    """Comprehensive test suite for the Filipino Dictionary API."""
    
    def create_app(self):
        """Create Flask test app."""
        from app import create_app
        app = create_app()
        app.config['TESTING'] = True
        return app
    
    def setUp(self):
        """Set up test environment."""
        self.client = self.app.test_client()
        self.base_url = 'http://localhost:5000'
        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }
        self.test_words = {
            'bahay': {
                'pos': 'n',
                'expected_definitions': ['house', 'home', 'dwelling'],
                'expected_relations': ['tahanan', 'tirahan']
            },
            'aral': {
                'pos': 'n',
                'expected_definitions': ['lesson', 'study', 'moral'],
                'expected_relations': ['pag-aaral', 'aralin']
            }
        }
    
    def test_health_endpoint(self):
        """Test API health check endpoint."""
        response = self.client.get('/api/health')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['status'], 'healthy')
        
    def test_graphql_introspection(self):
        """Test GraphQL schema introspection."""
        query = """
        {
            __schema {
                types {
                    name
                    description
                }
            }
        }
        """
        response = self.client.post('/graphql', 
                                  json={'query': query},
                                  headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        self.assertIn('__schema', data['data'])
    
    def test_word_lookup(self):
        """Test word lookup functionality."""
        for word, expected in self.test_words.items():
            query = f"""
            {{
                getWord(word: "{word}") {{
                    lemma
                    definitions {{
                        definitionText
                        partOfSpeech
                    }}
                    etymology {{
                        components {{
                            text
                            language
                        }}
                    }}
                }}
            }}
            """
            response = self.client.post('/graphql',
                                      json={'query': query},
                                      headers=self.headers)
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('data', data)
            word_data = data['data']['getWord']
            self.assertEqual(word_data['lemma'], word)
            
            # Check definitions
            definitions = word_data['definitions']
            self.assertTrue(len(definitions) > 0)
            found_definitions = [d['definitionText'].lower() for d in definitions]
            for expected_def in expected['expected_definitions']:
                self.assertTrue(
                    any(expected_def in d for d in found_definitions),
                    f"Expected definition '{expected_def}' not found in {found_definitions}"
                )
    
    def test_word_search(self):
        """Test word search functionality."""
        test_queries = [
            ('bahay', 'exact'),
            ('bah', 'prefix'),
            ('hay', 'contains'),
            ('bahey', 'fuzzy')
        ]
        
        for query, mode in test_queries:
            graphql_query = f"""
            {{
                searchWords(
                    query: "{query}",
                    mode: {mode},
                    limit: 5
                ) {{
                    lemma
                    definitions {{
                        definitionText
                    }}
                }}
            }}
            """
            response = self.client.post('/graphql',
                                      json={'query': graphql_query},
                                      headers=self.headers)
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('data', data)
            search_results = data['data']['searchWords']
            self.assertIsInstance(search_results, list)
            if mode == 'exact':
                self.assertTrue(
                    any(r['lemma'] == query for r in search_results),
                    f"Exact match '{query}' not found in results"
                )
    
    def test_word_network(self):
        """Test word network functionality."""
        for word in self.test_words:
            query = f"""
            {{
                getWordNetwork(
                    word: "{word}",
                    depth: 2,
                    breadth: 5,
                    includeEtymology: true,
                    includeAffixes: true
                ) {{
                    nodes {{
                        id
                        info {{
                            word
                            definition
                        }}
                    }}
                    links {{
                        source
                        target
                        type
                    }}
                }}
            }}
            """
            response = self.client.post('/graphql',
                                      json={'query': query},
                                      headers=self.headers)
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('data', data)
            network = data['data']['getWordNetwork']
            
            # Check network structure
            self.assertIn('nodes', network)
            self.assertIn('links', network)
            self.assertTrue(len(network['nodes']) > 0)
            self.assertTrue(len(network['links']) > 0)
            
            # Check if the original word is in the network
            node_words = [n['info']['word'] for n in network['nodes']]
            self.assertIn(word, node_words)
            
            # Check if expected relations are in the network
            expected_relations = self.test_words[word]['expected_relations']
            for relation in expected_relations:
                self.assertTrue(
                    any(relation == w for w in node_words),
                    f"Expected relation '{relation}' not found in network"
                )
    
    def test_baybayin_processing(self):
        """Test Baybayin text processing."""
        query = """
        {
            baybayinWords(
                limit: 5,
                minQuality: 80,
                hasEtymology: true
            ) {
                lemma
                baybayinForm
                romanizedForm
                hasEtymology
                qualityMetrics {
                    totalScore
                }
            }
        }
        """
        response = self.client.post('/graphql',
                                  json={'query': query},
                                  headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('data', data)
        words = data['data']['baybayinWords']
        self.assertIsInstance(words, list)
        self.assertTrue(len(words) <= 5)
        
        for word in words:
            self.assertIsNotNone(word['baybayinForm'])
            self.assertIsNotNone(word['romanizedForm'])
            self.assertTrue(word['hasEtymology'])
            self.assertGreaterEqual(word['qualityMetrics']['totalScore'], 80)
    
    def test_error_handling(self):
        """Test error handling for various scenarios."""
        test_cases = [
            {
                'query': '{ nonExistentField }',
                'expected_error': 'Cannot query field'
            },
            {
                'query': '{ getWord }',
                'expected_error': 'Field getWord argument "word" of type "String!" is required'
            },
            {
                'query': '{ getWord(word: "") }',
                'expected_error': 'Word cannot be empty'
            },
            {
                'query': '{ searchWords(query: "", limit: -1) }',
                'expected_error': 'Limit must be positive'
            }
        ]
        
        for case in test_cases:
            response = self.client.post('/graphql',
                                      json={'query': case['query']},
                                      headers=self.headers)
            data = json.loads(response.data)
            self.assertIn('errors', data)
            error_message = data['errors'][0]['message']
            self.assertTrue(
                case['expected_error'] in error_message,
                f"Expected error '{case['expected_error']}' not found in '{error_message}'"
            )
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        # Make multiple rapid requests
        query = """
        {
            getWord(word: "bahay") {
                lemma
            }
        }
        """
        
        responses = []
        for _ in range(50):  # Adjust based on rate limit
            response = self.client.post('/graphql',
                                      json={'query': query},
                                      headers=self.headers)
            responses.append(response)
        
        # Check if rate limiting was applied
        rate_limited = any(r.status_code == 429 for r in responses)
        self.assertTrue(rate_limited, "Rate limiting was not applied")
    
    def test_cors_headers(self):
        """Test CORS headers in responses."""
        response = self.client.options('/graphql')
        self.assertEqual(response.status_code, 200)
        self.assertIn('Access-Control-Allow-Origin', response.headers)
        self.assertIn('Access-Control-Allow-Methods', response.headers)
        self.assertIn('Access-Control-Allow-Headers', response.headers)
    
    def test_response_format(self):
        """Test consistency of response formats."""
        query = """
        {
            getWord(word: "bahay") {
                lemma
                definitions {
                    definitionText
                    partOfSpeech
                    examples {
                        text
                        translation
                    }
                }
                etymology {
                    components {
                        text
                        language
                    }
                }
                relationships {
                    synonyms
                    antonyms
                    related
                }
                qualityMetrics {
                    totalScore
                    completeness
                    accuracy
                }
            }
        }
        """
        response = self.client.post('/graphql',
                                  json={'query': query},
                                  headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        
        # Check response structure
        self.assertIn('data', data)
        word_data = data['data']['getWord']
        
        # Check required fields
        required_fields = [
            'lemma',
            'definitions',
            'etymology',
            'relationships',
            'qualityMetrics'
        ]
        for field in required_fields:
            self.assertIn(field, word_data)
        
        # Check definition structure
        for definition in word_data['definitions']:
            self.assertIn('definitionText', definition)
            self.assertIn('partOfSpeech', definition)
            if 'examples' in definition:
                for example in definition['examples']:
                    self.assertIn('text', example)
                    self.assertIn('translation', example)
        
        # Check relationships structure
        relationships = word_data['relationships']
        relationship_types = ['synonyms', 'antonyms', 'related']
        for rel_type in relationship_types:
            self.assertIn(rel_type, relationships)
            self.assertIsInstance(relationships[rel_type], list)
        
        # Check quality metrics
        metrics = word_data['qualityMetrics']
        required_metrics = ['totalScore', 'completeness', 'accuracy']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

if __name__ == '__main__':
    unittest.main() 