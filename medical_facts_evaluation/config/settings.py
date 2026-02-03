"""
Application settings with environment variable support.

All configuration is loaded from environment variables or .env file.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from functools import lru_cache

# Try to load dotenv if available
try:
    from dotenv import load_dotenv
    
    # Look for .env in multiple locations
    # 1. Current working directory
    # 2. Package parent directory (RAGAS_Medical_Facts_Agent_refactor)
    package_dir = Path(__file__).parent.parent.parent
    env_file = package_dir / ".env"
    
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()  # Try current working directory
except ImportError:
    pass


@dataclass
class Settings:
    """Application settings loaded from environment variables."""
    
    # RAGFlow API
    ragflow_base_url: str = field(
        default_factory=lambda: os.getenv(
            "RAGFLOW_BASE_URL", 
            "http://172.17.16.150/api/v1/agents_openai"
        )
    )
    medical_facts_auth_token: str = field(
        default_factory=lambda: os.getenv("MEDICAL_FACTS_AUTH_TOKEN", "")
    )
    medical_facts_model: str = field(
        default_factory=lambda: os.getenv("MEDICAL_FACTS_MODEL", "phi-4")
    )
    default_agent_id: str = field(
        default_factory=lambda: os.getenv(
            "DEFAULT_AGENT_ID", 
            "df4cb87efd2011f0b3234afd40f7103b"
        )
    )
    
    # OpenAI for RAGAS
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    openai_embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_EMBEDDING_MODEL", 
            "text-embedding-ada-002"
        )
    )
    
    # Paths
    results_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("RESULTS_DIR", "results/medical_facts_production")
        )
    )
    test_cases_dir: Path = field(
        default_factory=lambda: Path(
            os.getenv("TEST_CASES_DIR", "test_cases")
        )
    )
    
    # API Settings
    api_timeout: int = field(
        default_factory=lambda: int(os.getenv("API_TIMEOUT", "120"))
    )
    api_retries: int = field(
        default_factory=lambda: int(os.getenv("API_RETRIES", "3"))
    )
    
    def get_agent_url(self, agent_id: str) -> str:
        """Build full API URL for an agent."""
        return f"{self.ragflow_base_url}/{agent_id}/chat/completions"
    
    def validate(self) -> list[str]:
        """Validate required settings. Returns list of errors."""
        errors = []
        
        if not self.openai_api_key:
            errors.append("OPENAI_API_KEY environment variable not set")
        
        if not self.medical_facts_auth_token:
            errors.append("MEDICAL_FACTS_AUTH_TOKEN environment variable not set")
        
        return errors


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()
