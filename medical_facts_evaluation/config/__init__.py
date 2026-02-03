"""Configuration module for Medical Facts Evaluation."""

from .settings import Settings, get_settings
from .thresholds import QualityThresholds, PRODUCTION, DEVELOPMENT

__all__ = ["Settings", "get_settings", "QualityThresholds", "PRODUCTION", "DEVELOPMENT"]
