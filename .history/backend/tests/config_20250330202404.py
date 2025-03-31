import os
from typing import Dict, Any

class TestConfig:
    """Configuration for testing environment."""
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "TEST_DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/fil_relex_test"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Redis configuration for rate limiting
    REDIS_URL = os.getenv("TEST_REDIS_URL", "redis://localhost:6379/1")
    
    # Security settings
    SECRET_KEY = "test-secret-key"
    JWT_SECRET_KEY = "test-jwt-secret-key"
    JWT_ACCESS_TOKEN_EXPIRES = 3600  # 1 hour
    
    # Rate limiting
    RATELIMIT_ENABLED = True
    RATELIMIT_STORAGE_URL = REDIS_URL
    RATELIMIT_STRATEGY = "fixed-window"
    RATELIMIT_DEFAULT = "100 per minute"
    
    # CORS settings
    CORS_ORIGINS = ["http://localhost:3000"]
    CORS_METHODS = ["GET", "POST", "OPTIONS"]
    CORS_HEADERS = ["Content-Type", "Authorization"]
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # API settings
    API_PREFIX = "/api/v1"
    GRAPHQL_ENDPOINT = "/graphql"
    
    # Test data
    TEST_WORDS: Dict[str, Any] = {
        "ganda": {
            "lemma": "ganda",
            "normalized_lemma": "ganda",
            "language_code": "tl",
            "has_baybayin": True,
            "baybayin_form": "ᜄᜈ᜔ᜇ",
            "definitions": [
                {
                    "definition_text": "beauty; loveliness; attractiveness",
                    "part_of_speech": "n",
                    "examples": [
                        {
                            "text": "Ang ganda ng araw.",
                            "translation": "What a beautiful day."
                        }
                    ]
                }
            ],
            "etymologies": [
                {
                    "etymology_text": "From Proto-Malayo-Polynesian *ganda",
                    "language_codes": "tl,ceb",
                    "confidence_score": 0.8
                }
            ]
        }
    }
    
    @classmethod
    def get_test_word(cls, word: str) -> Dict[str, Any]:
        """Get test word data."""
        return cls.TEST_WORDS.get(word, {})
    
    @classmethod
    def get_test_database_url(cls) -> str:
        """Get test database URL."""
        return cls.SQLALCHEMY_DATABASE_URI
    
    @classmethod
    def get_test_redis_url(cls) -> str:
        """Get test Redis URL."""
        return cls.REDIS_URL 