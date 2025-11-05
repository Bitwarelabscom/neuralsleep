"""
Configuration management for NeuralSleep
"""

import os
from dotenv import load_dotenv


class Config:
    """Application configuration"""

    def __init__(self):
        load_dotenv()

        # Environment
        self.node_env = os.getenv('NODE_ENV', 'development')

        # MemoryCore
        self.memorycore_url = os.getenv('MEMORYCORE_URL', 'http://localhost:3002')
        self.memorycore_api_key = os.getenv('MEMORYCORE_API_KEY', '')

        # LNN Services
        self.semantic_lnn_port = int(os.getenv('SEMANTIC_LNN_PORT', 5000))
        self.episodic_lnn_port = int(os.getenv('EPISODIC_LNN_PORT', 5001))
        self.working_lnn_port = int(os.getenv('WORKING_LNN_PORT', 5002))

        # Database
        self.postgres_host = os.getenv('POSTGRES_HOST', 'localhost')
        self.postgres_port = int(os.getenv('POSTGRES_PORT', 5433))
        self.postgres_db = os.getenv('POSTGRES_DB', 'memorycore')
        self.postgres_user = os.getenv('POSTGRES_USER', 'memorycore_user')
        self.postgres_password = os.getenv('POSTGRES_PASSWORD', '')

        # Redis
        self.redis_host = os.getenv('REDIS_HOST', 'localhost')
        self.redis_port = int(os.getenv('REDIS_PORT', 6380))

        # Model Configuration
        self.semantic_hidden_size = int(os.getenv('SEMANTIC_HIDDEN_SIZE', 1024))
        self.episodic_hidden_size = int(os.getenv('EPISODIC_HIDDEN_SIZE', 512))
        self.working_hidden_size = int(os.getenv('WORKING_HIDDEN_SIZE', 256))

        # Training
        self.semantic_lr = float(os.getenv('SEMANTIC_LEARNING_RATE', 0.0001))
        self.episodic_lr = float(os.getenv('EPISODIC_LEARNING_RATE', 0.001))
        self.working_lr = float(os.getenv('WORKING_LEARNING_RATE', 0.01))

        # Consolidation
        self.consolidation_batch_size = int(os.getenv('CONSOLIDATION_BATCH_SIZE', 32))
        self.replay_speedup_factor = float(os.getenv('REPLAY_SPEEDUP_FACTOR', 10.0))

        # Consciousness
        self.enable_phi_computation = os.getenv('ENABLE_PHI_COMPUTATION', 'true').lower() == 'true'
        self.phi_computation_interval = int(os.getenv('PHI_COMPUTATION_INTERVAL', 3600))

        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE')

    @property
    def postgres_url(self) -> str:
        """PostgreSQL connection URL"""
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    @property
    def redis_url(self) -> str:
        """Redis connection URL"""
        return f"redis://{self.redis_host}:{self.redis_port}"

    def __repr__(self) -> str:
        return f"<Config env={self.node_env}>"
