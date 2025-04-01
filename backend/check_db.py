"""
Simple script to verify database connectivity.
"""

import os
import sys
import logging
from dotenv import load_dotenv
import psycopg2
from sqlalchemy import create_engine, text

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_db_connection():
    """Check if we can connect to the database."""
    # Get database configuration from .env
    db_config = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 5432)),
        'database': os.getenv('DB_NAME', 'fil_dict_db'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres')
    }
    
    logger.info(f"Attempting to connect to database: {db_config['database']} on {db_config['host']}:{db_config['port']}")
    
    # Try direct psycopg2 connection
    try:
        conn = psycopg2.connect(**db_config)
        with conn.cursor() as cursor:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            logger.info(f"Successfully connected to PostgreSQL: {version[0]}")
            
            # Check if required tables exist
            cursor.execute("""
                SELECT tablename FROM pg_tables 
                WHERE schemaname = 'public'
                ORDER BY tablename;
            """)
            tables = cursor.fetchall()
            logger.info(f"Found {len(tables)} tables in database:")
            for table in tables:
                logger.info(f"  - {table[0]}")
        conn.close()
        logger.info("Database connection test successful")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {str(e)}")
        return False

def check_sqlalchemy_connection():
    """Check if SQLAlchemy can connect to the database."""
    db_url = os.getenv('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5432/fil_dict_db')
    logger.info(f"Attempting SQLAlchemy connection with URL: {db_url}")
    
    try:
        engine = create_engine(db_url)
        with engine.connect() as connection:
            result = connection.execute(text("SELECT COUNT(*) FROM words"))
            count = result.scalar()
            logger.info(f"Words table contains {count} records")
            
            # Check other tables
            tables_to_check = ['definitions', 'etymologies', 'relations', 'parts_of_speech']
            for table in tables_to_check:
                try:
                    result = connection.execute(text(f"SELECT COUNT(*) FROM {table}"))
                    logger.info(f"{table} table contains {result.scalar()} records")
                except Exception as e:
                    logger.warning(f"Error checking {table} table: {str(e)}")
        
        logger.info("SQLAlchemy connection test successful")
        return True
    except Exception as e:
        logger.error(f"SQLAlchemy connection failed: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Database Connectivity Check ===")
    psycopg_result = check_db_connection()
    sqlalchemy_result = check_sqlalchemy_connection()
    
    if psycopg_result and sqlalchemy_result:
        print("✅ All database connectivity tests passed")
        sys.exit(0)
    else:
        print("❌ Database connectivity tests failed")
        sys.exit(1) 