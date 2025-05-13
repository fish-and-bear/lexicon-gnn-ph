# Database utility functions
import logging
import psycopg2
import os
from dotenv import load_dotenv
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_db_connection():
    """Establish a read-only database connection using credentials from .env file."""

    conn = None
    tried_methods = []

    # Load environment variables from .env
    # First try the root directory
    root_env_path = Path(__file__).parent.parent / '.env'
    backend_env_path = Path(__file__).parent.parent / 'backend' / '.env'
    
    # Try root directory first, then backend
    if root_env_path.exists():
        env_path = root_env_path
        logger.info(f"Using .env file from project root directory: {env_path}")
    else:
        env_path = backend_env_path
        logger.info(f"Using .env file from backend directory: {env_path}")
    
    load_dotenv(dotenv_path=env_path)

    # Get credentials from environment variables
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "fil_dict_db")
    db_user = os.getenv("DB_USER", "postgres")
    db_password = os.getenv("DB_PASSWORD", "")

    logger.info(f"Loading database configuration from {env_path}")

    try:
        # Method 1: Try with the configured parameters
        try:
            logger.info(f"Trying to connect to database {db_name} with configured parameters...")

            tried_methods.append("configured parameters")

            conn = psycopg2.connect(
                host=db_host,
                port=db_port,
                dbname=db_name,
                user=db_user,
                password=db_password,
                application_name="FilRelex-ReadOnlyAnalysis"
            )

        except psycopg2.OperationalError as e:
            logger.warning(f"Failed to connect with configured parameters: {e}")

            # Method 2: Try with trust authentication (no password)
            if "no password supplied" in str(e) or "password authentication failed" in str(e):

                logger.info("Trying trust authentication (no password)...")
                tried_methods.append("trust authentication")

                # Create a copy without password for trust-based authentication
                try:
                    conn = psycopg2.connect(
                        host=db_host,
                        port=db_port,
                        dbname=db_name,
                        user=db_user,
                        application_name="FilRelex-ReadOnlyAnalysis"
                    )
                except psycopg2.OperationalError as trust_err:
                    logger.warning(f"Trust authentication failed: {trust_err}")

                    # Method 3: Try default 'postgres' database
                    logger.info("Trying to connect to default 'postgres' database...")
                    tried_methods.append("default postgres database")

                    try:
                        conn = psycopg2.connect(
                            host=db_host,
                            port=db_port,
                            dbname="postgres",
                            user=db_user,
                            application_name="FilRelex-ReadOnlyAnalysis"
                        )
                    except psycopg2.OperationalError as default_err:
                        logger.warning(f"Default database connection failed: {default_err}")
                        raise Exception(f"All connection methods failed: {tried_methods}")
            else:
                # Some other connection error occurred
                raise

        # Set read-only mode
        if conn:
            conn.set_session(readonly=True, autocommit=False)
            logger.info(f"Connected to database {conn.info.dbname} in READ-ONLY mode")

        return conn

    except Exception as e:
        logger.error(f"Failed to connect to database after trying: {tried_methods}")
        logger.error(f"Error: {e}")
        if conn and not conn.closed:
            conn.close()
        raise

# Example usage:
if __name__ == "__main__":
    try:
        # Get database connection
        connection = get_db_connection()
        
        # Execute a simple query
        with connection.cursor() as cursor:
            cursor.execute("SELECT count(*) FROM words")
            result = cursor.fetchone()
            logger.info(f"Total words in database: {result[0]}")
            
    except Exception as e:
        logger.error(f"Error: {e}")
    finally:
        if 'connection' in locals() and connection:
            connection.close()
            logger.info("Database connection closed")
