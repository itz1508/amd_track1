"""
Input Validation Module

Validates tasks.json before any model calls.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


class InputValidator:
    """Validates input tasks.json structure."""
    
    def __init__(self):
        self._errors = []
        self._warnings = []
        self._malformed_tasks = []
    
    def validate(self, data: Any) -> Tuple[bool, List[Dict], List[Dict]]:
        """
        Validate input data.
        
        Args:
            data: Parsed JSON data from tasks.json
            
        Returns:
            Tuple of (valid, valid_tasks, malformed_tasks)
        """
        self._errors = []
        self._warnings = []
        self._malformed_tasks = []
        
        # Reset state
        valid_tasks = []
        
        # Check 1: Root is a JSON array
        if not isinstance(data, list):
            self._errors.append("Root must be a JSON array")
            return False, [], []
        
        # Check 2: Every item is an object
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                self._errors.append(f"Item {i} is not an object (type: {type(item).__name__})")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': f"Not an object (type: {type(item).__name__})"
                })
                continue
            
            # Check 3: Every task has a non-empty task_id
            if 'task_id' not in item:
                self._errors.append(f"Item {i} missing task_id")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': "Missing task_id"
                })
                continue
            
            task_id = item.get('task_id')
            if not isinstance(task_id, str):
                self._errors.append(f"Item {i} task_id is not a string (type: {type(task_id).__name__})")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': f"task_id is not a string"
                })
                continue
            
            if not task_id.strip():
                self._errors.append(f"Item {i} has empty task_id")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': "Empty task_id"
                })
                continue
            
            # Check 4: Every task has a non-empty prompt
            if 'prompt' not in item:
                self._errors.append(f"Item {i} ({task_id}) missing prompt")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': "Missing prompt"
                })
                continue
            
            prompt = item.get('prompt')
            if not isinstance(prompt, str):
                self._errors.append(f"Item {i} ({task_id}) prompt is not a string")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': "prompt is not a string"
                })
                continue
            
            if not prompt.strip():
                self._errors.append(f"Item {i} ({task_id}) has empty prompt")
                self._malformed_tasks.append({
                    'index': i,
                    'item': item,
                    'error': "Empty prompt"
                })
                continue
            
            # Task is valid, add to valid_tasks
            valid_tasks.append({
                'task_id': task_id,
                'prompt': prompt
            })
        
        # Check 5: Task IDs are unique
        seen_ids = set()
        duplicate_ids = []
        for task in valid_tasks:
            task_id = task['task_id']
            if task_id in seen_ids:
                duplicate_ids.append(task_id)
            else:
                seen_ids.add(task_id)
        
        if duplicate_ids:
            self._errors.append(f"Duplicate task IDs: {duplicate_ids}")
            # Remove duplicates from valid_tasks, keeping first occurrence
            unique_tasks = []
            seen = set()
            for task in valid_tasks:
                if task['task_id'] not in seen:
                    unique_tasks.append(task)
                    seen.add(task['task_id'])
            valid_tasks = unique_tasks
        
        return len(self._errors) == 0, valid_tasks, self._malformed_tasks
    
    def validate_from_file(self, file_path: str) -> Tuple[bool, List[Dict], List[Dict], str]:
        """
        Validate tasks.json from file path.
        
        Args:
            file_path: Path to tasks.json file
            
        Returns:
            Tuple of (valid, valid_tasks, malformed_tasks, error_message)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            return False, [], [], f"Invalid JSON: {str(e)}"
        except Exception as e:
            return False, [], [], f"Error reading file: {str(e)}"
        
        valid, tasks, malformed = self.validate(data)
        return valid, tasks, malformed, ""
    
    def get_errors(self) -> List[str]:
        """Get all validation errors."""
        return self._errors
    
    def get_warnings(self) -> List[str]:
        """Get all validation warnings."""
        return self._warnings
    
    def get_malformed_tasks(self) -> List[Dict]:
        """Get all malformed tasks."""
        return self._malformed_tasks


# Singleton instance
input_validator = InputValidator()
