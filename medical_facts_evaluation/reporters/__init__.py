"""Reporters for Medical Facts Evaluation output."""

from .console import ConsoleReporter
from .json_reporter import JsonReporter

__all__ = ["ConsoleReporter", "JsonReporter"]
