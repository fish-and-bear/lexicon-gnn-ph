"""
Test script for the Filipino Dictionary API.
Performs comprehensive testing of all endpoints and compares with direct database queries.
"""

import requests
import psycopg2
from psycopg2.extras import DictCursor
import json
from typing import Dict, Any, List
import time
from datetime import datetime
import logging
from database import get_db_config
import pytest
from models import Word, Definition, Etymology, Relation, Pronunciation, Credit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API base URL
API_BASE = "http://localhost:10000/api/v2"

# Database configuration
DB_CONFIG = get_db_config()

def get_db_connection():
    """Get a database connection for direct queries."""
    return psycopg2.connect(
        host=DB_CONFIG['host'],
        port=DB_CONFIG['port'],
        database=DB_CONFIG['database'],
        user=DB_CONFIG['user'],
        password=DB_CONFIG['password'],
        cursor_factory=DictCursor
    )

def get_sqlalchemy_session():
    """Get a SQLAlchemy session."""
    engine = create_engine(f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}")
    Session = sessionmaker(bind=engine)
    return Session()

class APITester:
    """Class to test API endpoints and compare with database queries."""
    
    def __init__(self):
        self.conn = get_db_connection()
        self.session = get_sqlalchemy_session()
    
    def close(self):
        """Close database connections."""
        self.conn.close()
        self.session.close()
    
    def test_word_lookup(self, word: str):
        """Test word lookup endpoint and compare with database."""
        logger.info(f"Testing word lookup for: {word}")
        
        # API request
        api_response = requests.get(f"{API_BASE}/words/{word}")
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database query
        db_word = self.session.query(Word).filter(Word.normalized_lemma == word.lower()).first()
        db_data = db_word.to_dict() if db_word else None
        
        # Compare results
        self._compare_results("Word lookup", api_data, db_data)
        
        return api_data, db_data
    
    def test_search(self, query: str, **params):
        """Test search endpoint and compare with database."""
        logger.info(f"Testing search for: {query}")
        
        # API request
        api_response = requests.get(f"{API_BASE}/search", params={"q": query, **params})
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database query using SQLAlchemy
        db_query = self.session.query(Word).filter(
            Word.search_text.match(query)
        ).order_by(Word.lemma)
        
        if params.get('language'):
            db_query = db_query.filter(Word.language_code == params['language'])
        
        db_words = db_query.all()
        db_data = {
            "words": [word.to_dict() for word in db_words],
            "total": len(db_words)
        }
        
        # Compare results
        self._compare_results("Search", api_data, db_data)
        
        return api_data, db_data
    
    def test_word_relations(self, word: str):
        """Test word relations endpoint and compare with database."""
        logger.info(f"Testing relations for: {word}")
        
        # API request
        api_response = requests.get(f"{API_BASE}/words/{word}/relations")
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database query
        word_obj = self.session.query(Word).filter(Word.normalized_lemma == word.lower()).first()
        if word_obj:
            db_data = {
                "outgoing_relations": [rel.to_dict() for rel in word_obj.outgoing_relations],
                "incoming_relations": [rel.to_dict() for rel in word_obj.incoming_relations]
            }
        else:
            db_data = None
        
        # Compare results
        self._compare_results("Relations", api_data, db_data)
        
        return api_data, db_data
    
    def test_word_etymology(self, word: str):
        """Test word etymology endpoint and compare with database."""
        logger.info(f"Testing etymology for: {word}")
        
        # API request
        api_response = requests.get(f"{API_BASE}/words/{word}/etymology")
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database query
        word_obj = self.session.query(Word).filter(Word.normalized_lemma == word.lower()).first()
        if word_obj:
            db_data = {
                "etymologies": [etym.to_dict() for etym in word_obj.etymologies],
                "has_etymology": bool(word_obj.etymologies)
            }
        else:
            db_data = None
        
        # Compare results
        self._compare_results("Etymology", api_data, db_data)
        
        return api_data, db_data
    
    def test_word_pronunciation(self, word: str):
        """Test word pronunciation endpoint and compare with database."""
        logger.info(f"Testing pronunciation for: {word}")
        
        # API request
        api_response = requests.get(f"{API_BASE}/words/{word}/pronunciation")
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database query
        word_obj = self.session.query(Word).filter(Word.normalized_lemma == word.lower()).first()
        if word_obj:
            db_data = {
                "pronunciations": [pron.to_dict() for pron in word_obj.pronunciations],
                "has_pronunciation": bool(word_obj.pronunciations)
            }
        else:
            db_data = None
        
        # Compare results
        self._compare_results("Pronunciation", api_data, db_data)
        
        return api_data, db_data
    
    def test_statistics(self):
        """Test statistics endpoint and compare with database."""
        logger.info("Testing statistics")
        
        # API request
        api_response = requests.get(f"{API_BASE}/statistics")
        api_data = api_response.json() if api_response.ok else None
        
        # Direct database queries
        db_data = {
            "total_words": self.session.query(Word).count(),
            "total_definitions": self.session.query(Definition).count(),
            "total_etymologies": self.session.query(Etymology).count(),
            "total_relations": self.session.query(Relation).count(),
            "words_with_baybayin": self.session.query(Word).filter(Word.has_baybayin == True).count(),
            "words_by_language": dict(
                self.session.query(
                    Word.language_code,
                    func.count(Word.id)
                ).group_by(Word.language_code).all()
            )
        }
        
        # Compare results
        self._compare_results("Statistics", api_data, db_data)
        
        return api_data, db_data
    
    def _compare_results(self, test_name: str, api_data: Dict, db_data: Dict):
        """Compare API results with database results."""
        if api_data is None and db_data is None:
            logger.info(f"{test_name}: Both API and DB returned no data")
            return
        
        if api_data is None:
            logger.error(f"{test_name}: API returned no data but DB did")
            return
        
        if db_data is None:
            logger.error(f"{test_name}: DB returned no data but API did")
            return
        
        # Compare key counts
        api_keys = set(api_data.keys())
        db_keys = set(db_data.keys())
        
        if api_keys != db_keys:
            logger.warning(f"{test_name}: Key mismatch - API: {api_keys}, DB: {db_keys}")
        
        # Compare values for common keys
        for key in api_keys & db_keys:
            if api_data[key] != db_data[key]:
                logger.warning(f"{test_name}: Value mismatch for key '{key}'")
                logger.warning(f"API: {api_data[key]}")
                logger.warning(f"DB:  {db_data[key]}")

def run_comprehensive_tests():
    """Run comprehensive tests on the API."""
    tester = APITester()
    
    try:
        # Test basic word lookup
        tester.test_word_lookup("aklat")
        tester.test_word_lookup("bahay")
        
        # Test search with different parameters
        tester.test_search("a", limit=10)
        tester.test_search("bahay", mode="exact")
        tester.test_search("ᜊ", mode="baybayin")
        
        # Test word details
        for word in ["aklat", "bahay", "tao"]:
            tester.test_word_relations(word)
            tester.test_word_etymology(word)
            tester.test_word_pronunciation(word)
        
        # Test statistics
        tester.test_statistics()
        
    finally:
        tester.close()

if __name__ == "__main__":
    run_comprehensive_tests() 