"""
Migration 004: Add definition_metadata column to definitions table

This migration fixes the missing definition_metadata column that's causing
SQLAlchemy errors in the application.
"""

from sqlalchemy import (
    create_engine,
    Column,
    String,
    Text,
    Integer,
    Boolean,
    Float,
    DateTime,
    ForeignKey,
    MetaData,
    Table,
    JSON,
)
import os
import logging

logger = logging.getLogger(__name__)

def upgrade(engine):
    """
    Add definition_metadata JSONB column to definitions table
    """
    connection = engine.connect()
    transaction = connection.begin()
    
    try:
        # Add definition_metadata column
        connection.execute(
            """
            ALTER TABLE definitions
            ADD COLUMN IF NOT EXISTS definition_metadata JSONB DEFAULT '{}'::jsonb
            """
        )
        
        logger.info("Added definition_metadata column to definitions table")
        transaction.commit()
        
        logger.info("Migration 004 completed successfully")
        return True
    except Exception as e:
        transaction.rollback()
        logger.error(f"Migration 004 failed: {e}")
        return False
    finally:
        connection.close()

def downgrade(engine):
    """
    Remove definition_metadata column from definitions table
    """
    connection = engine.connect()
    transaction = connection.begin()
    
    try:
        # Remove the column
        connection.execute(
            """
            ALTER TABLE definitions
            DROP COLUMN IF EXISTS definition_metadata
            """
        )
        
        logger.info("Removed definition_metadata column from definitions table")
        transaction.commit()
        
        logger.info("Migration 004 downgrade completed successfully")
        return True
    except Exception as e:
        transaction.rollback()
        logger.error(f"Migration 004 downgrade failed: {e}")
        return False
    finally:
        connection.close() 