"""
Medical Facts Agent API Client.

Handles communication with the deployed Medical Facts Agent in RAGFlow.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass

import requests

from ..config.settings import Settings, get_settings


@dataclass
class ApiResponse:
    """Response from the Medical Facts Agent API."""
    content: str
    api_time_seconds: float
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None and bool(self.content.strip())


class MedicalFactsClient:
    """Client for deployed Medical Facts Agent in RAGFlow."""
    
    def __init__(
        self,
        api_url: str,
        auth_token: str,
        model: str = "phi-4",
        timeout: int = 120,
        retries: int = 3,
        verbose: bool = False,
    ):
        """
        Initialize the Medical Facts client.
        
        Args:
            api_url: Full URL to the agent's chat completions endpoint
            auth_token: Bearer token for authentication
            model: Model name to use
            timeout: Request timeout in seconds
            retries: Number of retry attempts on failure
            verbose: Whether to print status messages
        """
        self.api_url = api_url
        self.auth_token = auth_token
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
    
    def extract_facts(self, transcript: str) -> ApiResponse:
        """
        Call the Medical Facts Agent to extract facts from a transcript.
        
        Args:
            transcript: Medical consultation transcript
            
        Returns:
            ApiResponse with content, timing, and any error
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": transcript}]
        }
        
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.retries + 1):
            try:
                if self.verbose:
                    print(f"  📡 API call attempt {attempt + 1}/{self.retries + 1}...")
                
                start_time = time.time()
                response = requests.post(
                    self.api_url,
                    json=payload,
                    headers=headers,
                    timeout=self.timeout
                )
                api_time = time.time() - start_time
                
                response.raise_for_status()
                
                result = response.json()
                content = (
                    result
                    .get('choices', [{}])[0]
                    .get('message', {})
                    .get('content', '')
                )
                
                if not content.strip():
                    if attempt < self.retries:
                        time.sleep(5)
                        continue
                    return ApiResponse(
                        content="",
                        api_time_seconds=api_time,
                        error="Empty response from agent"
                    )
                
                if self.verbose:
                    print(f"  ✅ Response received ({len(content)} chars)")
                
                return ApiResponse(
                    content=content,
                    api_time_seconds=api_time
                )
                
            except requests.exceptions.Timeout:
                if attempt < self.retries:
                    time.sleep(3)
                    continue
                return ApiResponse(
                    content="",
                    api_time_seconds=self.timeout,
                    error=f"Request timeout ({self.timeout}s)"
                )
                
            except requests.exceptions.RequestException as e:
                if attempt < self.retries:
                    time.sleep(3)
                    continue
                return ApiResponse(
                    content="",
                    api_time_seconds=0.0,
                    error=f"API error: {str(e)}"
                )
        
        return ApiResponse(
            content="",
            api_time_seconds=0.0,
            error="Max retries exceeded"
        )
    
    @classmethod
    def from_settings(
        cls,
        agent_id: str,
        settings: Optional[Settings] = None,
        verbose: bool = False,
    ) -> "MedicalFactsClient":
        """
        Create a client from application settings.
        
        Args:
            agent_id: The agent ID to connect to
            settings: Settings object (uses default if not provided)
            verbose: Whether to print status messages
            
        Returns:
            Configured MedicalFactsClient
        """
        if settings is None:
            settings = get_settings()
        
        api_url = settings.get_agent_url(agent_id)
        
        return cls(
            api_url=api_url,
            auth_token=settings.medical_facts_auth_token,
            model=settings.medical_facts_model,
            timeout=settings.api_timeout,
            retries=settings.api_retries,
            verbose=verbose,
        )


def _clean_double_quotes(obj):
    """Recursively clean double-escaped quotes from strings in nested structures."""
    if isinstance(obj, str):
        # Fix ""value"" -> "value"
        if obj.startswith('"') and obj.endswith('"'):
            obj = obj[1:-1]
        return obj
    elif isinstance(obj, list):
        return [_clean_double_quotes(item) for item in obj if item is not None]
    elif isinstance(obj, dict):
        return {k: _clean_double_quotes(v) for k, v in obj.items() if v is not None}
    return obj


def _merge_nested_structure(data: dict) -> dict:
    """
    Merge nested structure from 4-agent no-merger approach into flat structure.
    
    Expected input format:
    {
        "clinical": {"chief_complaint": ..., "symptoms": ..., ...},
        "medications": {"medications_taken": ..., "medications_names": ..., ...},
        "measurements": {"vital_signs": ..., "physical_examination": ..., ...},
        "context": {"medical_history": ..., "therapeutic_interventions": ..., ...}
    }
    
    Output: flat dict with all keys at top level
    """
    nested_keys = ["clinical", "medications", "measurements", "context"]
    
    # Check if this is a nested structure
    if not any(key in data for key in nested_keys):
        return data  # Already flat
    
    merged = {}
    
    for section_key in nested_keys:
        section = data.get(section_key, {})
        if isinstance(section, dict):
            merged.update(section)
        elif isinstance(section, list) and len(section) > 0:
            # Handle case where section is array instead of object
            if isinstance(section[0], dict):
                merged.update(section[0])
    
    # Also copy any top-level keys that aren't the section keys
    for key, value in data.items():
        if key not in nested_keys and key not in merged:
            merged[key] = value
    
    return merged


def parse_medical_facts(content: str) -> Tuple[Optional[dict], Optional[str]]:
    """
    Parse Medical Facts JSON from agent response.
    
    Handles both flat structure (from Merger agent) and nested structure 
    (from 4-agent no-merger approach).
    
    Args:
        content: Raw response content from agent
        
    Returns:
        Tuple of (parsed_dict, error_message)
    """
    import json
    import re
    
    if not content:
        return None, "Empty response"
    
    # Try to clean up common JSON issues before parsing
    cleaned_content = content
    
    # Remove any text before first { or after last }
    first_brace = cleaned_content.find('{')
    last_brace = cleaned_content.rfind('}')
    if first_brace != -1 and last_brace != -1:
        cleaned_content = cleaned_content[first_brace:last_brace + 1]
    
    # Fix double-double quotes: ""value"" -> "value"
    cleaned_content = re.sub(r'""([^"]+)""', r'"\1"', cleaned_content)
    
    # Fix malformed key-value: ""key": "value"" -> "key": "value"
    cleaned_content = re.sub(r'""(\w+)":', r'"\1":', cleaned_content)
    
    try:
        data = json.loads(cleaned_content)
    except json.JSONDecodeError as e:
        # Try original content as fallback
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None, f"JSON parse error: {str(e)}"
    
    # Clean any remaining double-quote issues in the parsed data
    data = _clean_double_quotes(data)
    
    # Merge nested structure if present
    data = _merge_nested_structure(data)
    
    return data, None
