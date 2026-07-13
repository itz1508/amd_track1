"""
Tool 8: Submission Validator

Validates the final output before writing to results.json.
"""

import json
import os
from typing import Any, Optional, Tuple


class SubmissionValidator:
    """Final submission validator."""
    
    def __init__(self):
        self._input_tasks = None
    
    def load_input_tasks(self, input_path: str) -> bool:
        """
        Load the input tasks for validation.
        
        Args:
            input_path: Path to /input/tasks.json
            
        Returns:
            True if loaded successfully
        """
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                self._input_tasks = json.load(f)
            return True
        except Exception as e:
            self._input_tasks = None
            return False
    
    def validate_results_structure(self, results: list) -> Tuple[bool, list]:
        """
        Validate the structure of results.json output.
        
        Checks:
        - Results is a JSON array
        - Every item is an object
        - Every item has task_id and answer
        - No extra fields
        - All answers are strings
        
        Args:
            results: The results list to validate
            
        Returns:
            Tuple of (valid, errors)
        """
        errors = []
        
        if not isinstance(results, list):
            errors.append("Root must be a JSON array")
            return False, errors
        
        for i, item in enumerate(results):
            if not isinstance(item, dict):
                errors.append(f"Item {i} is not an object")
                continue
            
            # Check required fields
            if 'task_id' not in item:
                errors.append(f"Item {i} missing task_id")
                continue
            if 'answer' not in item:
                errors.append(f"Item {i} ({item.get('task_id', 'unknown')}) missing answer")
                continue
            
            # Check no extra fields
            extra_fields = [k for k in item.keys() if k not in ['task_id', 'answer']]
            if extra_fields:
                errors.append(f"Item {i} ({item.get('task_id', 'unknown')}) has extra fields: {extra_fields}")
            
            # Check answer is a string
            if not isinstance(item['answer'], str):
                errors.append(f"Item {i} ({item.get('task_id', 'unknown')}) answer is not a string")
        
        return len(errors) == 0, errors
    
    def validate_task_coverage(self, results: list, input_tasks: list) -> Tuple[bool, list]:
        """
        Validate that all input tasks are covered in results.
        
        Checks:
        - Every input task ID appears exactly once
        - No unknown task IDs exist
        
        Args:
            results: The results list
            input_tasks: The original input tasks
            
        Returns:
            Tuple of (valid, errors)
        """
        errors = []
        
        if not input_tasks:
            errors.append("No input tasks loaded for validation")
            return False, errors
        
        input_ids = {t['task_id'] for t in input_tasks}
        result_ids = {r['task_id'] for r in results}
        
        # Check for missing IDs
        missing = input_ids - result_ids
        if missing:
            errors.append(f"Missing task IDs: {sorted(missing)}")
        
        # Check for extra IDs
        extra = result_ids - input_ids
        if extra:
            errors.append(f"Unknown task IDs: {sorted(extra)}")
        
        # Check for duplicates in results
        seen = set()
        duplicates = []
        for r in results:
            if r['task_id'] in seen:
                duplicates.append(r['task_id'])
            seen.add(r['task_id'])
        if duplicates:
            errors.append(f"Duplicate task IDs in results: {duplicates}")
        
        return len(errors) == 0, errors
    
    def validate_results_json(self, results: list, input_tasks: list) -> Tuple[bool, list]:
        """
        Comprehensive validation of results.json.
        
        Args:
            results: The results list
            input_tasks: The original input tasks
            
        Returns:
            Tuple of (valid, all_errors)
        """
        all_errors = []
        
        # Validate structure
        struct_valid, struct_errors = self.validate_results_structure(results)
        all_errors.extend(struct_errors)
        
        # Validate coverage
        if input_tasks:
            coverage_valid, coverage_errors = self.validate_task_coverage(results, input_tasks)
            all_errors.extend(coverage_errors)
        
        return len(all_errors) == 0, all_errors
    
    def validate_json_serializable(self, results: list) -> Tuple[bool, str]:
        """
        Validate that results can be serialized to JSON.
        
        Args:
            results: The results list
            
        Returns:
            Tuple of (valid, error_message)
        """
        try:
            json_str = json.dumps(results, ensure_ascii=False)
            # Also verify it can be read back
            json.loads(json_str)
            return True, ""
        except Exception as e:
            return False, str(e)
    
    def validate_output_file(self, output_path: str, input_tasks: list = None) -> Tuple[bool, list]:
        """
        Validate the output file at the specified path.
        
        Args:
            output_path: Path to /output/results.json
            input_tasks: Optional input tasks for coverage validation
            
        Returns:
            Tuple of (valid, errors)
        """
        errors = []
        
        # Check file exists
        if not os.path.exists(output_path):
            errors.append(f"Output file not found: {output_path}")
            return False, errors
        
        # Check file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            errors.append("Output file is empty")
            return False, errors
        
        # Read and parse
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")
            return False, errors
        except Exception as e:
            errors.append(f"Error reading file: {e}")
            return False, errors
        
        # Validate structure and coverage
        valid, validation_errors = self.validate_results_json(results, input_tasks or [])
        errors.extend(validation_errors)
        
        return len(errors) == 0, errors
    
    def create_valid_output(self, task_results: list) -> str:
        """
        Create a valid results.json string from task results.
        
        Args:
            task_results: List of dicts with task_id and answer
            
        Returns:
            JSON string ready to write
        """
        # Ensure all items have only task_id and answer
        cleaned = []
        for item in task_results:
            cleaned.append({
                'task_id': item['task_id'],
                'answer': str(item['answer']) if 'answer' in item else ''
            })
        
        return json.dumps(cleaned, indent=2, ensure_ascii=False)
    
    def atomic_write(self, output_path: str, content: str, temp_suffix: str = '.tmp', input_tasks: list = None) -> bool:
        """
        Atomically write content to output path.
        
        Writes to a temporary file first, validates it, then renames.
        
        Args:
            output_path: Path to write to
            content: JSON content to write
            temp_suffix: Suffix for temporary file
            
        Returns:
            True if write succeeded
        """
        temp_path = output_path + temp_suffix
        
        try:
            # Write to temp file
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Validate the temp file
            valid, errors = self.validate_output_file(temp_path, input_tasks)
            if not valid:
                # Clean up temp file
                try:
                    os.remove(temp_path)
                except:
                    pass
                return False
            
            # Atomic replace
            os.replace(temp_path, output_path)
            return True
            
        except Exception as e:
            # Clean up temp file if it exists
            try:
                os.remove(temp_path)
            except:
                pass
            return False


# Singleton instance
submission_validator = SubmissionValidator()
