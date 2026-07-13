"""
AMD Track 1 Entry Point

Main entry point for the AMD Track 1 general-purpose AI agent.
Reads /input/tasks.json, processes tasks, writes /output/results.json.

FIX: Defer Fireworks configuration validation until a task actually requires Fireworks.
This allows deterministic-only tasks to succeed without FIREWORKS_API_KEY, FIREWORKS_BASE_URL,
or ALLOWED_MODELS set in the environment.
"""

import argparse
import json
import os
import sys
from typing import List, Optional

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from .executor import TaskExecutor, get_executor
from .model_registry import get_model_registry
from .classifier import get_classifier


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='AMD Track 1 General-Purpose AI Agent'
    )
    
    parser.add_argument(
        '--input',
        default='/input/tasks.json',
        help='Path to input tasks.json'
    )
    
    parser.add_argument(
        '--output',
        default='/output/results.json',
        help='Path to output results.json'
    )
    
    parser.add_argument(
        '--skills-dir',
        default=None,
        help='Directory containing skill definitions (default: amd_track1/skills)'
    )
    
    parser.add_argument(
        '--max-concurrency',
        type=int,
        default=4,
        help='Maximum concurrent requests (default: 4)'
    )
    
    parser.add_argument(
        '--timeout',
        type=float,
        default=600.0,  # 10 minutes
        help='Maximum total execution time in seconds (default: 600)'
    )
    
    args = parser.parse_args()
    
    # Set up paths
    input_path = args.input
    output_path = args.output
    
    # Determine skills directory
    if args.skills_dir:
        skills_dir = args.skills_dir
    else:
        # Use default relative to this file
        skills_dir = os.path.join(os.path.dirname(__file__), 'skills')
    
    # Create executor (does not require Fireworks for deterministic-only work)
    try:
        executor = get_executor(
            skills_dir=skills_dir,
            max_concurrency=args.max_concurrency
        )
    except Exception as e:
        print(f"Error initializing executor: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Process input
    # Fireworks configuration is validated lazily inside process_input when/if needed
    try:
        start_time = __import__('time').time()
        
        success, errors = executor.process_input(
            input_path, output_path, total_timeout=args.timeout
        )
        
        elapsed = __import__('time').time() - start_time
        
        if success:
            # Count tasks from the actual output file, not directory contents
            try:
                import json
                with open(output_path, 'r') as f:
                    output_data = json.load(f)
                    task_count = len(output_data)
            except Exception:
                task_count = 0
            print(f"Successfully processed {task_count} tasks in {elapsed:.2f}s", file=sys.stderr)
            sys.exit(0)
        else:
            print(f"Completed with errors in {elapsed:.2f}s:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            sys.exit(1)
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
