"""
Executor

Main execution engine that orchestrates the complete workflow:
- Input validation
- Task classification
- Routing
- Model execution (via Fireworks)
- Validation
- Retry/Escalation
- Output generation
"""

import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

from .input_validation import InputValidator, input_validator
from .capability_profiles import CapabilityDecision, classify_capabilities, compose_system_prompt, select_allowed_model
from .tool_execution import post_validate_answer, tool_registry
from .model_registry import get_model_registry
from .tools.submission_validator import submission_validator


@dataclass
class ExecutionResult:
    """Result of executing a single task."""
    task_id: str
    answer: Optional[str]
    category: str
    model_used: Optional[str]
    success: bool
    validation_errors: List[str] = field(default_factory=list)
    model_error: Optional[str] = None
    attempt_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    latency: Optional[float] = None

    def to_output_dict(self) -> Dict[str, str]:
        """Convert to output format (only task_id and answer)."""
        return {
            'task_id': self.task_id,
            'answer': self.answer or ''
        }


class FireworksClient:
    """Client for Fireworks API."""

    TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}
    PERMANENT_STATUS_CODES = {400, 401, 404, 422}

    def __init__(self, api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_transport_retries: int = 2,
                 transport_retry_base_delay: float = 1.0,
                 transport_retry_max_delay: float = 30.0):
        """
        Initialize Fireworks client.

        Args:
            api_key: FIREWORKS_API_KEY
            base_url: FIREWORKS_BASE_URL
        """
        self.api_key = api_key or os.environ.get('FIREWORKS_API_KEY', '')
        self.base_url = (base_url or os.environ.get('FIREWORKS_BASE_URL', '')).rstrip('/')
        if self.base_url and not self.base_url.endswith('/v1'):
            self.base_url += '/v1'
        self.max_transport_retries = max(1, max_transport_retries)
        self.transport_retry_base_delay = max(0.0, transport_retry_base_delay)
        self.transport_retry_max_delay = max(0.0, transport_retry_max_delay)

        if not self.api_key:
            raise ValueError("FIREWORKS_API_KEY not set")
        if not self.base_url:
            raise ValueError("FIREWORKS_BASE_URL not set")

    @classmethod
    def is_transient_error(cls, error: Optional[str]) -> bool:
        """Return True when an error string represents a retryable provider failure."""
        if not error:
            return False
        error_lower = error.lower()
        if "transport" in error_lower or "connection" in error_lower or "timeout" in error_lower:
            return True
        for status_code in cls.TRANSIENT_STATUS_CODES:
            if f"http {status_code}" in error_lower:
                return True
        return False

    def _retry_delay(self, retry_index: int) -> float:
        """Calculate bounded exponential backoff with small jitter."""
        if self.transport_retry_base_delay <= 0.0:
            return 0.0
        exponential = self.transport_retry_base_delay * (2 ** retry_index)
        jitter = random.uniform(0.0, min(1.0, self.transport_retry_base_delay))
        return min(self.transport_retry_max_delay, exponential + jitter)

    def infer(self, model_id: str, prompt: str,
              timeout: float = 300.0,
              max_tokens: int = 4096,
              system_instruction: Optional[str] = None) -> Tuple[Optional[Any], Optional[str], Optional[int], Optional[int], Optional[float]]:
        """
        Run inference with a model.

        Args:
            model_id: The model to use
            prompt: The prompt to send
            timeout: Timeout in seconds

        Returns:
            Tuple of (answer, error, input_tokens, output_tokens, latency)
        """
        import requests

        start_time = time.time()
        deadline = start_time + timeout

        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }

        messages = [{'role': 'user', 'content': prompt}]
        if system_instruction:
            messages.insert(0, {'role': 'system', 'content': system_instruction})

        payload = {
            'model': model_id,
            'messages': messages,
            'max_tokens': max_tokens,
            'temperature': 0.0,

        }

        last_error = None

        for transport_attempt in range(self.max_transport_retries):
            remaining = deadline - time.time()
            if remaining <= 0:
                latency = time.time() - start_time
                error_msg = last_error or f"Transport timeout: exceeded {timeout:.2f}s model-call budget"
                return None, error_msg, None, None, latency

            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=remaining
                )

                latency = time.time() - start_time

                if response.status_code != 200:
                    error_msg = f"HTTP {response.status_code}: {response.text}"
                    last_error = error_msg

                    if response.status_code in self.TRANSIENT_STATUS_CODES:
                        if transport_attempt == self.max_transport_retries - 1:
                            return None, error_msg, None, None, latency
                        delay = self._retry_delay(transport_attempt)
                        if delay > 0.0:
                            sleep_for = min(delay, max(0.0, deadline - time.time()))
                            if sleep_for > 0.0:
                                time.sleep(sleep_for)
                        continue

                    return None, error_msg, None, None, latency

                data = response.json()

                # Extract answer
                if 'choices' in data and len(data['choices']) > 0:
                    answer = data['choices'][0].get('message', {}).get('content', '')
                else:
                    answer = ''

                # Extract token counts
                input_tokens = data.get('usage', {}).get('prompt_tokens')
                output_tokens = data.get('usage', {}).get('completion_tokens')

                return answer, None, input_tokens, output_tokens, latency

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                latency = time.time() - start_time
                last_error = f"Transport error: {e}"
                if transport_attempt == self.max_transport_retries - 1:
                    return None, last_error, None, None, latency
                delay = self._retry_delay(transport_attempt)
                if delay > 0.0:
                    sleep_for = min(delay, max(0.0, deadline - time.time()))
                    if sleep_for > 0.0:
                        time.sleep(sleep_for)
                continue
            except Exception as e:
                latency = time.time() - start_time
                return None, str(e), None, None, latency

        latency = time.time() - start_time
        return None, last_error or "Transport retry exhausted", None, None, latency


class TaskExecutor:
    """Main task executor."""

    def __init__(self, skills_dir: Optional[str] = None,
                 api_key: Optional[str] = None,
                 base_url: Optional[str] = None,
                 max_concurrency: int = 4,
                 local_client: Optional[Any] = None):
        """
        Initialize executor.

        Args:
            skills_dir: Directory containing skill definitions
            api_key: Fireworks API key (defaults to env)
            base_url: Fireworks base URL (defaults to env)
            max_concurrency: Maximum concurrent requests
            local_client: Ignored legacy compatibility parameter
        """
        self._skills_dir = skills_dir
        self._max_concurrency = max_concurrency

        # Initialize components
        self._validator = input_validator
        # The registry is retained only to read evaluator-provided model IDs.
        # Classification and local execution are owned by classify_capabilities
        # and ToolRegistry below; no router, subagent, verifier, or legacy
        # capability executor is constructed on the production path.
        self._registry = get_model_registry()

        # Initialize Fireworks client (remote fallback)
        self._fireworks_client = None
        if api_key and base_url:
            self._fireworks_client = FireworksClient(api_key, base_url)

        # Production Track 1 is Fireworks-only for model inference. Local-model
        # clients are intentionally ignored even when supplied by legacy callers.
        self._local_client = None

        # Production decisions are local and capability-aware; no planner,
        # verifier, or secondary model call participates in this path.

    def initialize(self, allowed_models: Optional[str] = None) -> bool:
        """
        Initialize from environment.

        Args:
            allowed_models: Override ALLOWED_MODELS

        Returns:
            True if initialization succeeded
        """
        configured_models = allowed_models or os.environ.get('ALLOWED_MODELS', '')
        if not configured_models.strip():
            return False
        self._registry.initialize(configured_models)

        # Initialize Fireworks client if not done (only if remote models are expected)
        if self._fireworks_client is None:
            try:
                self._fireworks_client = FireworksClient()
            except ValueError:
                return False

        return True

    def _select_inference_client(self, model_id: Optional[str]) -> Optional[Any]:
        """
        Select the production Fireworks inference client.

        Args:
            model_id: The requested model ID

        Returns:
            FireworksClient or LocalInferenceClient (both have .infer method)
        """
        return self._fireworks_client

    @staticmethod
    def _allowed_models() -> List[str]:
        """Read exactly the evaluator-published model IDs."""
        return [
            value.strip()
            for value in os.environ.get('ALLOWED_MODELS', '').split(',')
            if value.strip()
        ]

    @staticmethod
    def _select_accuracy_model(allowed_models: List[str], decision: CapabilityDecision) -> Optional[str]:
        """Choose one exact evaluator-allowed generative model for a capability."""
        return select_allowed_model(allowed_models, decision)

    def execute_task(self, task: Dict[str, str],
                     max_attempts: int = 2,
                     deadline: Optional[float] = None) -> ExecutionResult:
        """
        Execute a single task through the complete workflow.

        Args:
            task: Task dict with task_id and prompt
            max_attempts: Maximum attempts per task

        Returns:
            ExecutionResult
        """
        task_id = task['task_id']
        prompt = task['prompt']

        decision = classify_capabilities(prompt)

        # Direct-answer path: try deterministic tools first (no Fireworks config needed)
        tool_results = tool_registry.execute_applicable(prompt, decision)
        for tool_result in tool_results:
            if tool_result.state == "resolved" and tool_result.answer is not None:
                validation_errors = list(post_validate_answer(prompt, tool_result.answer))
                if not validation_errors:
                    print(f'Task {task_id}: resolved by local tool {tool_result.tool_name}', file=sys.stderr)
                    return ExecutionResult(
                        task_id, tool_result.answer, decision.primary, None, True,
                        validation_errors=validation_errors, attempt_count=1, latency=0.0
                    )
                else:
                    print(f'Task {task_id}: local tool {tool_result.tool_name} produced answer but validation failed: {validation_errors}', file=sys.stderr)
                    return ExecutionResult(
                        task_id, tool_result.answer, decision.primary, None, False,
                        validation_errors=validation_errors, model_error="; ".join(validation_errors),
                        attempt_count=1, latency=0.0
                    )

        # Non-deterministic task: requires Fireworks
        allowed_models = self._allowed_models()
        if not allowed_models:
            allowed_models = self._registry.get_allowed_models()
        model_id = self._select_accuracy_model(allowed_models, decision)
        if not model_id:
            return ExecutionResult(task_id, None, decision.primary, None, False,
                                   model_error='ALLOWED_MODELS not set', attempt_count=0)

        if self._fireworks_client is None:
            try:
                self._fireworks_client = FireworksClient()
            except ValueError as exc:
                return ExecutionResult(task_id, None, decision.primary, model_id, False,
                                       model_error=str(exc), attempt_count=0)

        if deadline is not None and deadline <= time.time():
            return ExecutionResult(task_id, None, decision.primary, model_id, False,
                                   model_error='Total execution timeout exceeded', attempt_count=0)

        call_timeout = min(300.0, max(0.1, deadline - time.time())) if deadline else 300.0
        
        # Fall back to Fireworks with direct-answer approach
        system_instruction = compose_system_prompt(decision)
        print(f'Using Fireworks model: {model_id}', file=sys.stderr)
        try:
            answer, model_error, input_tokens, output_tokens, latency = self._fireworks_client.infer(
                model_id, prompt, timeout=call_timeout, system_instruction=system_instruction
            )
        except TypeError as exc:
            if "system_instruction" not in str(exc):
                raise
            answer, model_error, input_tokens, output_tokens, latency = self._fireworks_client.infer(
                model_id, prompt, timeout=call_timeout
            )
        
        if model_error or answer is None or (isinstance(answer, str) and not answer.strip()):
            return ExecutionResult(task_id, None, decision.primary, model_id, False,
                                   model_error=model_error or 'Empty or malformed model response',
                                   attempt_count=1, input_tokens=input_tokens,
                                   output_tokens=output_tokens, latency=latency)
        
        # Extract answer from structured response if present (backward compatibility)
        if isinstance(answer, str) and answer.strip().startswith('{'):
            try:
                import json as json_mod
                parsed = json_mod.loads(answer)
                if isinstance(parsed, dict) and 'answer' in parsed:
                    answer = parsed['answer']
            except (json_mod.JSONDecodeError, TypeError, KeyError):
                pass  # Use answer as-is if not valid JSON or missing answer field
        
        # Validate the direct answer
        validation_errors = list(post_validate_answer(prompt, answer))
        if validation_errors:
            return ExecutionResult(task_id, answer, decision.primary, model_id, False,
                                   validation_errors=validation_errors, model_error="; ".join(validation_errors),
                                   attempt_count=1, input_tokens=input_tokens,
                                   output_tokens=output_tokens, latency=latency)
        
        return ExecutionResult(task_id, answer, decision.primary, model_id, True,
                               validation_errors=validation_errors, attempt_count=1,
                               input_tokens=input_tokens, output_tokens=output_tokens, latency=latency)

    def execute_batch(self, tasks: List[Dict[str, str]],
                      max_concurrency: Optional[int] = None,
                      deadline: Optional[float] = None) -> List[ExecutionResult]:
        """
        Execute a batch of tasks.

        Args:
            tasks: List of task dicts
            max_concurrency: Override max concurrency

        Returns:
            List of ExecutionResult objects
        """
        concurrency = max_concurrency or self._max_concurrency
        results_by_id = {}

        # With a global deadline, dispatch incrementally so work that has not
        # started can be reconciled by process_input instead of piling up.
        if deadline is not None and concurrency == 1:
            for task in tasks:
                if time.time() >= deadline:
                    break
                try:
                    result = self.execute_task(task, deadline=deadline)
                except Exception as exc:
                    result = ExecutionResult(task['task_id'], None, 'unknown', None, False, model_error=str(exc))
                if time.time() >= deadline:
                    result = ExecutionResult(task['task_id'], None, result.category, result.model_used, False,
                                             model_error='Total execution timeout exceeded')
                results_by_id[result.task_id] = result
            return [results_by_id.get(task['task_id'], ExecutionResult(
                task['task_id'], None, 'unknown', None, False, model_error='Total execution timeout exceeded'
            )) for task in tasks]

        # Use ThreadPoolExecutor for concurrent execution
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {
                executor.submit(self.execute_task, task, deadline=deadline): task
                for task in tasks
            }

            for future in as_completed(futures):
                task = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = ExecutionResult(task['task_id'], None, 'unknown', None, False, model_error=str(exc))
                results_by_id[result.task_id] = result

        return [results_by_id[task['task_id']] for task in tasks]

    def process_input(self, input_path: str = '/input/tasks.json',
                      output_path: str = '/output/results.json',
                      total_timeout: Optional[float] = None) -> Tuple[bool, List[str]]:
        """
        Process input tasks and write output results.

        Args:
            input_path: Path to input tasks.json
            output_path: Path to output results.json

        Returns:
            Tuple of (success, errors)
        """
        errors = []
        deadline = time.time() + total_timeout if total_timeout is not None else None

        # Step 1: Read and validate input
        valid, tasks, malformed, read_error = self._validator.validate_from_file(input_path)

        if read_error:
            errors.append(f"Error reading input: {read_error}")
            return False, errors

        if not valid or malformed:
            errors.extend(self._validator.get_errors())
            errors.extend(f"Malformed task at index {item['index']}: {item['error']}" for item in malformed)
            return False, errors
        if not tasks:
            return False, ["Input batch must contain at least one valid task."]

        # Direct-answer path: Fireworks is initialized lazily only when needed.
        # Deterministic tasks can be resolved without Fireworks configuration.
        # Initialize registry with allowed models if available
        allowed_models = self._allowed_models()
        if allowed_models:
            self._registry.initialize(','.join(allowed_models))

        # Step 2: execute_task initializes Fireworks lazily only after the
        # registry proves that no local tool resolved the task.
        results = self.execute_batch(tasks, deadline=deadline)

        # Step 3: Reconcile every valid input task to one contract-valid result.
        output_results = []
        for result in results:
            if result.success and result.answer and result.answer.strip():
                output_results.append(result.to_output_dict())
            else:
                errors.append(f"Task {result.task_id} failed: {result.model_error or result.validation_errors}")
                # Never output internal status phrases. Use the answer if available, even from a failed result.
                answer = result.answer if result.answer and result.answer.strip() else ''
                output_results.append({'task_id': result.task_id, 'answer': answer})

        # Step 4: Validate output
        valid_output, output_errors = submission_validator.validate_results_structure(output_results)
        errors.extend(output_errors)

        if not valid_output:
            return False, errors

        # Step 5: Check coverage
        input_task_ids = {t['task_id'] for t in tasks}
        output_task_ids = {r['task_id'] for r in output_results}

        missing = input_task_ids - output_task_ids
        if missing:
            errors.extend(f"Missing result for task: {tid}" for tid in missing)
            return False, errors

        # Step 6: Atomically write output
        json_str = submission_validator.create_valid_output(output_results)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            try:
                os.makedirs(output_dir)
            except Exception as e:
                errors.append(f"Failed to create output directory: {e}")
                return False, errors

        # Atomic write
        success = submission_validator.atomic_write(output_path, json_str, input_tasks=tasks)
        if not success:
            errors.append("Failed to write output atomically")
            return False, errors

        return True, errors


# Singleton instance
_executor_instance = None

def get_executor(skills_dir: Optional[str] = None,
                api_key: Optional[str] = None,
                base_url: Optional[str] = None,
                max_concurrency: int = 4) -> TaskExecutor:
    """Get or create the singleton executor instance."""
    global _executor_instance
    if _executor_instance is None:
        _executor_instance = TaskExecutor(skills_dir, api_key, base_url, max_concurrency)
    return _executor_instance
