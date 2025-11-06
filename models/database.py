"""
Database Connection Manager
Centralized database configuration for MongoDB
Shared by both ML Model and Streamlit applications
"""

import os
from typing import Optional
from pymongo import MongoClient
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseConfig:
    """
    Database configuration from environment variables
    """
    # MongoDB Configuration
    MONGO_HOST = os.getenv('MONGO_HOST', 'localhost')
    MONGO_PORT = int(os.getenv('MONGO_PORT', 27017))
    MONGO_DATABASE = os.getenv('MONGO_DATABASE', 'pi2502')
    MONGO_USER = os.getenv('MONGO_USER', '')
    MONGO_PASSWORD = os.getenv('MONGO_PASSWORD', '')

    # Connection Pool Settings
    POOL_SIZE = int(os.getenv('DB_POOL_SIZE', 5))
    MAX_OVERFLOW = int(os.getenv('DB_MAX_OVERFLOW', 10))
    POOL_RECYCLE = int(os.getenv('DB_POOL_RECYCLE', 3600))

    @classmethod
    def get_mongo_url(cls) -> str:
        """Generate MongoDB connection URL"""
        if cls.MONGO_USER and cls.MONGO_PASSWORD:
            return f"mongodb://{cls.MONGO_USER}:{cls.MONGO_PASSWORD}@{cls.MONGO_HOST}:{cls.MONGO_PORT}/{cls.MONGO_DATABASE}"
        return f"mongodb://{cls.MONGO_HOST}:{cls.MONGO_PORT}/{cls.MONGO_DATABASE}"


class MongoDBConnection:
    """
    MongoDB connection manager using PyMongo
    Implements singleton pattern
    """
    _instance: Optional['MongoDBConnection'] = None
    _client: Optional[MongoClient] = None
    _db = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize MongoDB connection"""
        try:
            # Create PyMongo client
            self._client = MongoClient(
                DatabaseConfig.get_mongo_url(),
                maxPoolSize=DatabaseConfig.POOL_SIZE,
                serverSelectionTimeoutMS=5000
            )
            self._db = self._client[DatabaseConfig.MONGO_DATABASE]

            logger.info("MongoDB connection initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB connection: {e}")
            raise

    @property
    def client(self) -> MongoClient:
        """Get PyMongo client"""
        return self._client

    @property
    def db(self):
        """Get MongoDB database instance"""
        return self._db

    def get_collection(self, collection_name: str):
        """Get a specific collection"""
        return self._db[collection_name]

    def test_connection(self) -> bool:
        """Test MongoDB connection"""
        try:
            # Ping the MongoDB server
            self._client.admin.command('ping')
            logger.info("MongoDB connection test successful")
            return True
        except Exception as e:
            logger.error(f"MongoDB connection test failed: {e}")
            return False

    def close_connection(self):
        """Close MongoDB connection"""
        if self._client:
            self._client.close()
            logger.info("MongoDB connection closed")