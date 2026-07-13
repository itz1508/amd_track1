"""
Model Registry

Manages available models from ALLOWED_MODELS environment variable.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ModelRecord:
    """Record for a registered model."""
    model_id: str
    available: bool = True
    successful_probe: bool = False
    latency: Optional[float] = None
    error: Optional[str] = None
    historical_category_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'available': self.available,
            'successful_probe': self.successful_probe,
            'latency': self.latency,
            'error': self.error,
            'historical_category_results': self.historical_category_results
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRecord':
        """Create from dictionary."""
        model_id = data.get('model_id')
        if model_id is None:
            raise ValueError("model_id is required")
        return cls(
            model_id=model_id,
            available=data.get('available', True),
            successful_probe=data.get('successful_probe', False),
            latency=data.get('latency'),
            error=data.get('error'),
            historical_category_results=data.get('historical_category_results', {})
        )


class ModelRegistry:
    """Registry of allowed models."""
    
    def __init__(self):
        self._models: Dict[str, ModelRecord] = {}
        self._allowed_models_str: Optional[str] = None
        self._initialized = False
    
    def initialize(self, allowed_models_str: Optional[str] = None) -> bool:
        """
        Initialize the registry from ALLOWED_MODELS environment variable.
        
        Args:
            allowed_models_str: Override ALLOWED_MODELS value (for testing)
            
        Returns:
            True if initialization succeeded
        """
        if self._initialized:
            return True
        
        # Get from environment or parameter
        if allowed_models_str is None:
            allowed_models_str = os.environ.get('ALLOWED_MODELS', '')
        
        self._allowed_models_str = allowed_models_str
        
        # Parse comma-separated list
        if not allowed_models_str.strip():
            return False
        
        model_ids = [m.strip() for m in allowed_models_str.split(',') if m.strip()]
        
        for model_id in model_ids:
            self._models[model_id] = ModelRecord(
                model_id=model_id,
                available=True,
                successful_probe=False
            )
        
        self._initialized = True
        return True
    
    def get_allowed_models(self) -> List[str]:
        """Get list of allowed model IDs."""
        return list(self._models.keys())
    
    def get_model(self, model_id: str) -> Optional[ModelRecord]:
        """Get model record by ID."""
        return self._models.get(model_id)
    
    def is_model_allowed(self, model_id: str) -> bool:
        """Check if a model is in the allowed list."""
        return model_id in self._models
    
    def record_probe_result(self, model_id: str, success: bool, 
                           latency: Optional[float] = None,
                           error: Optional[str] = None) -> bool:
        """
        Record the result of a probe to a model.
        
        Args:
            model_id: The model that was probed
            success: Whether the probe succeeded
            latency: Response latency in seconds
            error: Error message if failed
            
        Returns:
            True if model exists and was updated
        """
        if model_id not in self._models:
            return False
        
        model = self._models[model_id]
        model.successful_probe = success
        model.latency = latency
        model.error = error
        model.available = success
        
        return True
    
    def record_category_result(self, model_id: str, category: str,
                               success: bool, input_tokens: Optional[int] = None,
                               output_tokens: Optional[int] = None,
                               latency: Optional[float] = None) -> bool:
        """
        Record a result for a specific category.
        
        Args:
            model_id: The model that was used
            category: The task category
            success: Whether the task was successful
            input_tokens: Number of input tokens used
            output_tokens: Number of output tokens used
            latency: Response latency
            
        Returns:
            True if recorded successfully
        """
        if model_id not in self._models:
            return False
        
        model = self._models[model_id]
        
        if category not in model.historical_category_results:
            model.historical_category_results[category] = {
                'attempts': 0,
                'successes': 0,
                'total_input_tokens': 0,
                'total_output_tokens': 0,
                'total_latency': 0.0
            }
        
        cat_stats = model.historical_category_results[category]
        cat_stats['attempts'] += 1
        
        if success:
            cat_stats['successes'] += 1
        
        if input_tokens is not None:
            cat_stats['total_input_tokens'] += input_tokens
        
        if output_tokens is not None:
            cat_stats['total_output_tokens'] += output_tokens
        
        if latency is not None:
            cat_stats['total_latency'] += latency
        
        return True
    
    def get_category_stats(self, model_id: str, category: str) -> Optional[Dict[str, Any]]:
        """
        Get statistics for a model and category.
        
        Args:
            model_id: The model ID
            category: The category
            
        Returns:
            Stats dict or None if not found
        """
        model = self._models.get(model_id)
        if not model:
            return None
        return model.historical_category_results.get(category)
    
    def get_average_latency(self, model_id: str) -> Optional[float]:
        """Get average latency for a model."""
        model = self._models.get(model_id)
        if not model or model.latency is None:
            return None
        return model.latency
    
    def get_success_rate(self, model_id: str, category: Optional[str] = None) -> Optional[float]:
        """
        Get success rate for a model (overall or for a category).
        
        Args:
            model_id: The model ID
            category: Optional category to filter by
            
        Returns:
            Success rate (0.0 to 1.0) or None
        """
        model = self._models.get(model_id)
        if not model:
            return None
        
        if category:
            stats = model.historical_category_results.get(category)
            if not stats or stats['attempts'] == 0:
                return None
            return stats['successes'] / stats['attempts']
        else:
            total_attempts = 0
            total_successes = 0
            for stats in model.historical_category_results.values():
                total_attempts += stats['attempts']
                total_successes += stats['successes']
            
            if total_attempts == 0:
                return None
            return total_successes / total_attempts
    
    def get_average_tokens(self, model_id: str, category: str) -> Tuple[Optional[int], Optional[int]]:
        """
        Get average input and output tokens for a model and category.
        
        Args:
            model_id: The model ID
            category: The category
            
        Returns:
            Tuple of (avg_input_tokens, avg_output_tokens)
        """
        stats = self.get_category_stats(model_id, category)
        if not stats or stats['attempts'] == 0:
            return None, None
        
        avg_input = stats['total_input_tokens'] / stats['attempts']
        avg_output = stats['total_output_tokens'] / stats['attempts']
        
        return int(avg_input), int(avg_output)
    
    def register_local_model(self, model_id: str) -> bool:
        """
        Register a local model without requiring ALLOWED_MODELS.
        
        Args:
            model_id: The local model ID
            
        Returns:
            True if registered successfully
        """
        self._models[model_id] = ModelRecord(
            model_id=model_id,
            available=True,
            successful_probe=True
        )
        self._initialized = True
        return True
    
    def get_available_models(self) -> List[str]:
        """Get list of currently available models."""
        return [mid for mid, model in self._models.items() if model.available]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert registry to dictionary."""
        return {
            'models': {mid: model.to_dict() for mid, model in self._models.items()},
            'allowed_models_str': self._allowed_models_str
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRegistry':
        """Create registry from dictionary."""
        registry = cls()
        registry._initialized = True
        registry._allowed_models_str = data.get('allowed_models_str')
        
        for mid, model_data in data.get('models', {}).items():
            registry._models[mid] = ModelRecord.from_dict(model_data)
        
        return registry


# Singleton instance
_model_registry_instance = None

def get_model_registry() -> ModelRegistry:
    """Get or create the singleton model registry."""
    global _model_registry_instance
    if _model_registry_instance is None:
        _model_registry_instance = ModelRegistry()
    return _model_registry_instance
