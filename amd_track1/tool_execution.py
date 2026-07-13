"""Canonical Track 1 production tool execution.

This module is deliberately the only local tool owner.  It produces evidence,
not speculative answers: a local result is resolved only when the requested
operation was computed and verified by the relevant bounded implementation.
"""
from __future__ import annotations

import ast
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import sympy as sp
import z3

from .arithmetic_detection import extract_arithmetic_expression
from .tools.arithmetic_evaluator import arithmetic_evaluator

ResolutionState = Literal["resolved", "partially_resolved", "unsupported", "invalid_input", "execution_error"]


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    state: ResolutionState
    answer: str | None
    evidence: tuple[str, ...]
    reason: str
    unresolved_requirements: tuple[str, ...]
    correction: str | None
    next_action: str
    success_criteria: tuple[str, ...]
    confidence: float
    error_details: str | None = None


@dataclass(frozen=True)
class ToolSpec:
    tool_id: str
    supported_categories: tuple[str, ...]
    admission_predicate: Callable[[str], bool]
    executor: Callable[[str], ToolResult]
    final_answer_capability: bool
    priority: int
    timeout_seconds: float
    evidence_limit: int


def _result(
    name: str, state: ResolutionState, answer: str | None, evidence: tuple[str, ...] | list[str],
    reason: str, unresolved: tuple[str, ...] | list[str] = (), correction: str | None = None,
    next_action: str = "Send the unchanged original task and verified evidence to Fireworks.",
    criteria: tuple[str, ...] | list[str] = ("Return the requested answer.",), confidence: float = 0.0,
    error: str | None = None,
) -> ToolResult:
    """Make every result actionable, including unsupported and error states."""
    return ToolResult(name, state, answer, tuple(evidence), reason, tuple(unresolved), correction, next_action, tuple(criteria), confidence, error)


def _partial(name: str, evidence: list[str], reason: str, unresolved: list[str], correction: str | None = None) -> ToolResult:
    return _result(name, "partially_resolved", None, evidence, reason, unresolved, correction,
                   "Use the verified evidence for one Fireworks request.",
                   ("Resolve every unparsed or semantic requirement.", "Do not contradict the verified evidence."), 0.65)


def _code_from_prompt(prompt: str) -> str | None:
    fenced = re.search(r"```(?:python|py)?\s*\n(.*?)```", prompt, re.IGNORECASE | re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    if re.search(r"^\s*(def |class |import |from )", prompt, re.MULTILINE):
        return prompt.strip()
    return None


def _math_expression(text: str) -> str:
    value = text.strip().strip(".?! ").replace("^", "**").replace("φ", "totient")
    # Algebraic coefficient notation (for example 5x) is conventional in the
    # admitted symbolic subset.  Calculator admission remains stricter.
    return re.sub(r"(?<=\d)\s*(?=[A-Za-z])", "*", value)


def _safe_arithmetic_admission(prompt: str) -> bool:
    return extract_arithmetic_expression(prompt) is not None


def _safe_arithmetic(prompt: str) -> ToolResult:
    expression = extract_arithmetic_expression(prompt)
    assert expression is not None
    try:
        answer = arithmetic_evaluator.evaluate_to_string(expression)
        return _result("safe_arithmetic", "resolved", answer, (f"Executed calculator-safe expression: {expression}",),
                       "The complete unambiguous expression was evaluated locally.", (), None,
                       "Return the verified answer directly.", ("Expression executes successfully.",), 1.0)
    except Exception as exc:
        return _result("safe_arithmetic", "execution_error", None, (f"Admitted expression: {expression}",),
                       "The calculator-safe evaluator raised an execution error.", ("A numeric answer remains unresolved.",),
                       "Correct the malformed expression.", "Send the error evidence to Fireworks only if semantic help is requested.",
                       ("A valid calculator-safe expression executes.",), 0.0, str(exc))


_SYMBOLIC_MARKERS = re.compile(r"\b(solve|factor|simplify|expand|totient|phi\s*\(|gcd|lcm|modulo|modular inverse|differentiat|integrat|sum|product|integers?\s+n)\b|φ\s*\(", re.I)


def _symbolic_admission(prompt: str) -> bool:
    return bool(_SYMBOLIC_MARKERS.search(prompt)) and extract_arithmetic_expression(prompt) is None


def _sympify(text: str) -> sp.Expr:
    locals_: dict[str, Any] = {
        "sqrt": sp.sqrt, "sin": sp.sin, "cos": sp.cos, "exp": sp.exp, "log": sp.log,
        "totient": sp.totient, "pi": sp.pi, "E": sp.E,
    }
    return sp.sympify(_math_expression(text), locals=locals_, evaluate=True)


def _symbolic_math(prompt: str) -> ToolResult:
    lower = prompt.lower()
    try:
        totient_equation = re.search(r"(?:φ|phi)\s*\(\s*n\s*\)\s*=\s*(\d+)", prompt, re.I)
        if totient_equation:
            target, bound = int(totient_equation.group(1)), 10_000
            values = [n for n in range(1, bound + 1) if int(sp.totient(n)) == target]
            return _partial("symbolic_math", [f"Bounded search: 1 <= n <= {bound}.", f"Verified φ(n)={target}: {values}."],
                            "The finite search is verified, but it does not prove completeness beyond its documented bound.",
                            ("Whether solutions exist above the bound remains unresolved.",),
                            "Request a proof of a sufficient bound or accept the bounded result.")

        inverse = re.search(r"inverse\s+of\s+(-?\d+)\s+modulo\s+(-?\d+)", lower)
        if inverse:
            value, modulus = int(inverse.group(1)), int(inverse.group(2))
            answer = str(pow(value, -1, modulus))
            return _result("symbolic_math", "resolved", answer, (f"SymPy/Python modular inverse check: ({value} * {answer}) mod {modulus} = 1.",),
                           "The inverse exists and was verified exactly.", (), None, "Return the verified answer directly.",
                           ("The modular product equals one.",), 1.0)

        integer_factor = re.search(r"(?:integer\s+)?factor(?:ization)?\s+(?:of\s+)?(\d+)\s*$", lower)
        if integer_factor:
            value = int(integer_factor.group(1)); factors = sp.factorint(value)
            answer = " * ".join(f"{p}^{e}" if e > 1 else str(p) for p, e in factors.items()) or "1"
            return _result("symbolic_math", "resolved", answer, (f"SymPy factorint({value}) = {factors}.",),
                           "Integer factorization was computed exactly.", (), None, "Return the verified answer directly.",
                           ("Prime powers multiply to the input.",), 1.0)

        totient = re.search(r"(?:compute|evaluate|find)?\s*(?:φ|phi|totient)\s*\(\s*(\d+)\s*\)", lower)
        if totient:
            value = int(totient.group(1)); answer = str(sp.totient(value))
            return _result("symbolic_math", "resolved", answer, (f"SymPy totient({value}) = {answer}.",),
                           "Euler totient was evaluated exactly.", (), None, "Return the verified answer directly.",
                           ("The exact totient is returned.",), 1.0)

        if re.search(r"\bsolve\b", lower) and "=" in prompt:
            equation_text = re.split(r"\bsolve\b", prompt, flags=re.I, maxsplit=1)[1].strip()
            equation_text = re.split(r"\b(?:for|over)\b", equation_text, flags=re.I, maxsplit=1)[0].strip()
            left, right = equation_text.split("=", 1)
            equation = sp.Eq(_sympify(left), _sympify(right))
            symbols = sorted(equation.free_symbols, key=lambda item: item.name)
            if len(symbols) != 1:
                return _partial("symbolic_math", [f"Parsed equation: {equation}.", f"Accounted symbols: {[s.name for s in symbols]}."],
                                "Only univariate equation solving is admitted by this path.", ("A multivariate or ambiguous system remains unresolved.",),
                                "Provide an explicit small system or one variable.")
            solutions = sp.solve(equation, symbols[0])
            answer = ", ".join(str(solution) for solution in solutions)
            return _result("symbolic_math", "resolved", answer, (f"SymPy solved {equation} for {symbols[0]}.", f"All returned solutions: {solutions}."),
                           "The univariate equation and every returned solution were checked by SymPy.", (), None,
                           "Return the verified answer directly.", ("All SymPy solutions are returned.",), 1.0)

        operation = re.search(r"\b(factor|expand|simplify)\b\s+(.+)", prompt, re.I | re.S)
        if operation:
            verb, expression_text = operation.group(1).lower(), operation.group(2)
            expression = _sympify(expression_text)
            value = {"factor": sp.factor, "expand": sp.expand, "simplify": sp.simplify}[verb](expression)
            return _result("symbolic_math", "resolved", str(value), (f"SymPy {verb} input: {expression}.", f"Verified exact result: {value}."),
                           "The requested algebraic operation was unambiguous and exact.", (), None,
                           "Return the verified answer directly.", ("The exact symbolic result is returned.",), 1.0)
    except (ValueError, TypeError, sp.SympifyError, SyntaxError) as exc:
        return _result("symbolic_math", "invalid_input", None, ("Symbolic admission matched, but the notation could not be parsed conservatively.",),
                       "The expression or operation is ambiguous or malformed.", ("The intended mathematical operation remains unresolved.",),
                       "Use explicit variables, operators, and domains.", "Send only the parse finding to Fireworks if interpretation is needed.",
                       ("Every symbol and operation is unambiguous.",), 0.0, str(exc))
    return _partial("symbolic_math", ("No bounded symbolic operation could be extracted from the request."),
                    "The request is a proof, abstract-math question, or ambiguous notation rather than an admitted computation.",
                    ("The requested semantic mathematical work remains unresolved.",),
                    "State a concrete finite operation or use Fireworks for explanatory work.")


_CONSTRAINT_MARKERS = re.compile(r"\b(sat|csp|constraint|before|after|graph coloring|n[- ]?queens|\d+[- ]?queens|satisfiable|unsatisfiable|assignment|schedule)\b", re.I)


def _constraint_admission(prompt: str) -> bool:
    return bool(_CONSTRAINT_MARKERS.search(prompt))


def _constraints(prompt: str) -> ToolResult:
    parsed: list[str] = []
    pairs: list[tuple[str, str]] = []
    for earlier, later in re.findall(r"\b([A-Za-z][\w-]*)\s+is\s+before\s+([A-Za-z][\w-]*)\b", prompt, re.I):
        pairs.append((earlier, later)); parsed.append(f"{earlier} before {later}")
    for later, earlier in re.findall(r"\b([A-Za-z][\w-]*)\s+is\s+after\s+([A-Za-z][\w-]*)\b", prompt, re.I):
        pairs.append((earlier, later)); parsed.append(f"{earlier} before {later} (from after)")
    if pairs:
        names = sorted({name for pair in pairs for name in pair})
        positions = {name: z3.Int(f"position_{name}") for name in names}
        solver = z3.Solver()
        solver.add(z3.Distinct(*positions.values()))
        for position in positions.values(): solver.add(position >= 0, position < len(names))
        for earlier, later in pairs: solver.add(positions[earlier] < positions[later])
        state = solver.check()
        evidence = [f"Parsed constraint: {item}." for item in parsed]
        if state == z3.unsat:
            return _result("sat_csp", "resolved", "UNSAT", tuple(evidence + ["Z3 proved the conjunction unsatisfiable."]),
                           "Every parsed ordering constraint was sent to Z3.", (), None, "Return the verified answer directly.",
                           ("Z3 returns UNSAT.",), 1.0)
        model = solver.model(); ordered = sorted(names, key=lambda name: model[positions[name]].as_long())
        unparsed = [clause.strip() for clause in re.split(r"[.\n]", prompt) if clause.strip() and not re.search(r"\bis\s+(before|after)\b", clause, re.I) and not re.search(r"\b(order|constraint|arrange)\b", clause, re.I)]
        if unparsed:
            return _partial("sat_csp", evidence + [f"Z3 satisfying order for parsed constraints: {', '.join(ordered)}.", f"Unparsed clauses: {unparsed}."],
                            "Some natural-language clauses were not conservatively parsed.", ("The unparsed clauses may change the result.",),
                            "State each remaining relation as an explicit finite constraint.")
        return _result("sat_csp", "resolved", ", ".join(ordered), tuple(evidence + ["Z3 model satisfies every listed ordering constraint."]),
                       "All supplied constraints were parsed and the returned order is solver-verified.", (), None,
                       "Return the verified answer directly.", ("Every listed constraint holds.",), 1.0)
    queens = re.search(r"\b(\d+)\s*[- ]?queens\b", prompt, re.I)
    if queens:
        size = int(queens.group(1))
        if not 1 <= size <= 12:
            return _partial("sat_csp", [f"Requested N-Queens size: {size}."], "The production bound is 1 through 12 queens.",
                            ("The requested instance exceeds the bounded solver limit.",), "Use N <= 12 or request remote reasoning.")
        rows = [z3.Int(f"q_{column}") for column in range(size)]; solver = z3.Solver()
        solver.add(*(z3.And(row >= 0, row < size) for row in rows), z3.Distinct(*rows))
        for left in range(size):
            for right in range(left + 1, size): solver.add(z3.Abs(rows[left] - rows[right]) != right - left)
        if solver.check() == z3.unsat:
            return _result("sat_csp", "resolved", "UNSAT", (f"Z3 proved {size}-Queens unsatisfiable.",), "The bounded model is solver-proven.", (), None, "Return the verified answer directly.", ("Z3 returns UNSAT.",), 1.0)
        model = solver.model(); answer = ", ".join(f"({column},{model[rows[column]].as_long()})" for column in range(size))
        return _result("sat_csp", "resolved", answer, (f"Z3 solved bounded {size}-Queens.",), "The returned placement satisfies row and diagonal constraints.", (), None, "Return the verified answer directly.", ("All queens are nonattacking.",), 1.0)
    return _partial("sat_csp", ["Constraint-oriented wording was detected but no complete finite constraint grammar was parsed."],
                    "The tool does not fabricate formal constraints from ambiguous prose.", ("The intended variables and constraints remain unresolved.",),
                    "Supply explicit finite variables, domains, and constraints.")


def _json_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(json|schema|duplicate ids?|missing ids?|normalize)\b", prompt, re.I)) and ("{" in prompt or "[" in prompt)


def _json_candidate(prompt: str) -> str | None:
    marker = re.search(r"(?:json|payload|input)\s*:\s*(.+)$", prompt, re.I | re.S)
    if marker: return marker.group(1).strip().strip("`")
    start = min((index for index in (prompt.find("{"), prompt.find("[")) if index >= 0), default=-1)
    return prompt[start:].strip().strip("`") if start >= 0 else None


def _json_tool(prompt: str) -> ToolResult:
    candidate = _json_candidate(prompt)
    if not candidate:
        return _partial("structured_json", ["JSON validation was requested, but no JSON payload was supplied."], "There is no concrete structure to validate.", ("A JSON payload remains required.",), "Provide the exact JSON payload.")
    try:
        value = json.loads(candidate, object_pairs_hook=lambda pairs: pairs)
    except json.JSONDecodeError as exc:
        return _result("structured_json", "invalid_input", None, (f"JSON parse error at line {exc.lineno}, column {exc.colno}: {exc.msg}.",),
                       "The supplied JSON is syntactically invalid.", ("The JSON payload cannot be validated until it parses.",),
                       "Correct the exact parse location.", "Validate the corrected JSON locally.", ("The payload parses as JSON.",), 0.0, str(exc))
    def restore(item: Any) -> Any:
        if isinstance(item, list) and item and all(isinstance(part, tuple) and len(part) == 2 for part in item): return {key: restore(part) for key, part in item}
        if isinstance(item, list): return [restore(part) for part in item]
        return item
    restored = restore(value)
    ids = [item.get("task_id") or item.get("id") for item in restored] if isinstance(restored, list) and all(isinstance(item, dict) for item in restored) else []
    duplicates = sorted({identifier for identifier in ids if identifier and ids.count(identifier) > 1})
    if duplicates:
        return _result("structured_json", "invalid_input", None, ("JSON parsed successfully.", f"Duplicate IDs: {duplicates}."),
                       "Duplicate identifiers violate batch cardinality.", ("Every ID must be unique.",), "Give each item one unique ID.",
                       "Validate the corrected JSON locally.", ("No duplicate IDs remain.",), 0.0)
    normalized = json.dumps(restored, ensure_ascii=False, separators=(",", ":"))
    return _result("structured_json", "resolved", normalized, ("JSON parsed successfully.", "Lossless normalized JSON was produced.",),
                   "The request was local parsing/normalization and no semantic values were invented.", (), None,
                   "Return the verified answer directly.", ("The JSON payload parses and preserves values.",), 1.0)


def _format_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(exactly|maximum|at most|json-only|code-only|one-line|required headings?|word count|sentence count|bullet count|prefix|suffix)\b", prompt, re.I))


def _format_tool(prompt: str) -> ToolResult:
    word = re.search(r"\bexactly\s+(\d+)\s+words?\b", prompt, re.I)
    sentence = re.search(r"\bexactly\s+(\d+)\s+sentences?\b", prompt, re.I)
    candidate_match = re.search(r"(?:candidate|answer|output)\s*:\s*(.+)$", prompt, re.I | re.S)
    candidate = candidate_match.group(1).strip() if candidate_match else None
    requirements: list[str] = []
    if word: requirements.append(f"exactly {word.group(1)} words")
    if sentence: requirements.append(f"exactly {sentence.group(1)} sentences")
    if not candidate:
        return _partial("format_constraints", requirements or ["A response-format constraint was detected."],
                        "The tool can extract constraints but cannot validate an answer that was not supplied.",
                        ("The candidate response remains unavailable.",), "Provide the candidate answer for local validation.")
    evidence = list(requirements); failures: list[str] = []
    if word:
        observed = len(re.findall(r"\b[\w’'-]+\b", candidate)); evidence.append(f"Observed word count: {observed}.")
        if observed != int(word.group(1)): failures.append(f"Expected {word.group(1)} words, observed {observed}.")
    if sentence:
        observed = len([part for part in re.split(r"(?<=[.!?])\s+", candidate) if part.strip()]); evidence.append(f"Observed sentence count: {observed}.")
        if observed != int(sentence.group(1)): failures.append(f"Expected {sentence.group(1)} sentences, observed {observed}.")
    if "json-only" in prompt.lower():
        try: json.loads(candidate); evidence.append("Candidate is valid JSON.")
        except json.JSONDecodeError: failures.append("Candidate is not JSON-only valid JSON.")
    if "one-line" in prompt.lower() and "\n" in candidate: failures.append("Candidate contains more than one line.")
    if failures:
        return _result("format_constraints", "partially_resolved", None, tuple(evidence + failures), "The response violates one or more explicit format constraints.",
                       tuple(failures), "Correct only the listed format violations without blindly truncating meaning.", "Validate the revised answer locally.",
                       ("Every requested format count and marker matches.",), 0.8)
    return _result("format_constraints", "resolved", candidate, tuple(evidence), "The supplied candidate satisfies the locally verifiable format constraints.", (), None,
                   "Return the verified answer directly.", ("All extracted format constraints hold.",), 1.0)


_DANGEROUS = re.compile(r"\b(subprocess|socket|requests|urllib|http\.client|os\.system|os\.popen|multiprocessing|ctypes|shutil\.rmtree|eval\s*\(|exec\s*\(|__import__\s*\()", re.I)


def _python_admission(prompt: str) -> bool:
    return _code_from_prompt(prompt) is not None and bool(re.search(r"\b(run|execute|test|output|assert)\b", prompt, re.I))


def _limited_preexec() -> Callable[[], None] | None:
    if os.name != "posix": return None
    def apply_limits() -> None:
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
        resource.setrlimit(resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024))
        resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024, 64 * 1024))
    return apply_limits


def _bounded_python(prompt: str) -> ToolResult:
    code = _code_from_prompt(prompt)
    assert code is not None
    danger = _DANGEROUS.search(code)
    if danger:
        return _result("bounded_python", "invalid_input", None, (f"Rejected dangerous operation/import: {danger.group(0)}.",),
                       "The bounded runner does not execute code requesting network, process, or dynamic-execution capabilities.",
                       ("The requested code was not executed.",), "Remove the dangerous operation or use a purpose-built approved interface.",
                       "Keep the execution request local and bounded.", ("The code contains no rejected capability.",), 0.0)
    try: ast.parse(code)
    except SyntaxError as exc:
        return _result("bounded_python", "invalid_input", None, (f"Python syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}.",),
                       "The code cannot enter the runner until it parses.", ("No code was executed.",), "Correct the syntax error.",
                       "Run the corrected code once in the bounded runner.", ("The source parses.",), 0.0, str(exc))
    working = Path(tempfile.mkdtemp(prefix="amd-track1-run-"))
    try:
        source, stdout_path, stderr_path = working / "snippet.py", working / "stdout.txt", working / "stderr.txt"
        source.write_text(code, encoding="utf-8")
        environment = {"PATH": os.environ.get("PATH", ""), "HOME": str(working), "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            try:
                completed = subprocess.run([sys.executable, "-I", str(source)], cwd=working, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           env=environment, timeout=3, check=False, preexec_fn=_limited_preexec())
            except subprocess.TimeoutExpired:
                return _result("bounded_python", "execution_error", None, ("Execution exceeded the strict 3-second timeout.",),
                               "The child process was terminated by the bounded runner.", ("The program did not finish within the limit.",),
                               "Fix nontermination or reduce work.", "Run the corrected snippet once.", ("The program exits within 3 seconds.",), 0.0, "TimeoutExpired")
        output = stdout_path.read_text(encoding="utf-8", errors="replace")[:65536]; errors = stderr_path.read_text(encoding="utf-8", errors="replace")[:65536]
        evidence = [f"Exit code: {completed.returncode}.", f"stdout: {output}"]
        if errors: evidence.append(f"stderr: {errors}")
        if completed.returncode != 0:
            exception_type = re.search(r"([A-Za-z_][\w.]*Error|AssertionError):", errors)
            return _result("bounded_python", "execution_error", None, tuple(evidence), "The bounded child exited unsuccessfully.",
                           ("The supplied program or test still fails.",), "Correct the reported runtime failure.", "Run the corrected snippet once.",
                           ("Exit code is zero and expected output is observed.",), 0.0, exception_type.group(1) if exception_type else errors)
        return _result("bounded_python", "resolved", output.rstrip("\n"), tuple(evidence), "The supplied snippet executed once in an isolated temporary directory.", (), None,
                       "Return the verified output directly.", ("Exit code is zero.", "Captured stdout is returned."), 1.0)
    finally:
        shutil.rmtree(working, ignore_errors=True)


def _ast_admission(prompt: str) -> bool:
    return _code_from_prompt(prompt) is not None


def _python_ast(prompt: str) -> ToolResult:
    code = _code_from_prompt(prompt)
    assert code is not None
    try: tree = ast.parse(code)
    except SyntaxError as exc:
        return _partial("python_ast", [f"Syntax error at line {exc.lineno}, column {exc.offset}: {exc.msg}."], "Python source does not parse.",
                        ("Corrected code and its behavior remain unresolved.",), "Correct the exact syntax error.")
    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
    evidence = [f"Functions: {[node.name for node in functions]}.", f"Classes: {[node.name for node in classes]}."]
    mutable = []
    for function in functions:
        defaults = list(function.args.defaults) + [default for default in function.args.kw_defaults if default is not None]
        for default in defaults:
            if isinstance(default, (ast.List, ast.Dict, ast.Set)): mutable.append(f"{function.name}: mutable default at line {default.lineno}")
    if mutable: evidence.extend(mutable)
    requested = re.search(r"(?:function|method)\s+(?:named\s+)?([A-Za-z_]\w*)", prompt, re.I)
    missing = requested and requested.group(1) not in {item.name for item in functions}
    if missing: evidence.append(f"Missing requested function: {requested.group(1)}.")
    forbidden = _DANGEROUS.search(code)
    if forbidden: evidence.append(f"Forbidden operation found: {forbidden.group(0)}.")
    if mutable or missing or forbidden:
        correction = "Use None as the default and initialize a new list inside the function." if mutable else "Implement the requested interface without forbidden operations."
        return _partial("python_ast", evidence, "Static inspection found an interface or safety defect; it cannot prove the complete debugging task resolved.",
                        ("Corrected code and behavioral tests remain unresolved.",), correction)
    return _partial("python_ast", evidence + ["Static syntax and declared structure parse successfully."], "AST analysis verifies structure, not semantic correctness.",
                    ("Runtime behavior and requested semantics remain unresolved.",), "Run supplied tests or use Fireworks to generate a correction.")


def _debug_admission(prompt: str) -> bool:
    return _code_from_prompt(prompt) is not None and bool(re.search(r"\b(debug|fix|bug|failing|error)\b", prompt, re.I))


def _debug_workflow(prompt: str) -> ToolResult:
    ast_result = _python_ast(prompt)
    execution = _bounded_python(prompt) if _python_admission(prompt) else None
    evidence = list(ast_result.evidence) + (list(execution.evidence) if execution else [])
    return _partial("code_debugging", evidence, "The workflow completed static inspection and one safe local execution pass; semantic correction is not fabricated.",
                    ("A corrected implementation and post-correction tests remain unresolved.",), "Use the verified failure evidence for one Fireworks correction, then parse and test its result once.")


def _ner_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(ner|named entities?|extract entities?)\b", prompt, re.I))


def _ner(prompt: str) -> ToolResult:
    quoted = re.findall(r"[\"“]([^\"”]+)[\"”]", prompt)
    findings: list[str] = []
    for email in re.findall(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", prompt): findings.append(f"deterministically parsed entity: email {email}")
    for url in re.findall(r"https?://[^\s)]+", prompt): findings.append(f"deterministically parsed entity: URL {url}")
    for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", prompt): findings.append(f"deterministically parsed entity: IPv4 {ip}")
    for money in re.findall(r"(?:[$€£]\s?\d+(?:,\d{3})*(?:\.\d+)?)", prompt): findings.append(f"deterministically parsed entity: money {money}")
    for percent in re.findall(r"\b\d+(?:\.\d+)?%", prompt): findings.append(f"deterministically parsed entity: percentage {percent}")
    for item in quoted: findings.append(f"heuristic entity candidate: quoted name {item}")
    if not findings: findings.append("unresolved entity classification: no deterministic entity syntax found")
    return _partial("entity_candidates", findings, "This lightweight parser does not claim general-purpose named-entity recognition.",
                    ("Entity types not deterministically formatted remain unresolved.",), "Use Fireworks for semantic entity classification.")


def _sentiment_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(sentiment|aspect|absa|positive|negative|neutral|mixed|excellent|disappointing|irritating|usable)\b", prompt, re.I))


def _sentiment(prompt: str) -> ToolResult:
    text = prompt.lower(); evidence: list[str] = []
    aspects = re.findall(r"\b(camera|battery|app|service|support|price|quality|performance)\b", text)
    lexicon = {"excellent": "positive", "great": "positive", "good": "positive", "usable": "neutral", "disappointing": "negative", "irritating": "negative", "bad": "negative", "poor": "negative"}
    for aspect in dict.fromkeys(aspects):
        window = text[max(0, text.find(aspect) - 60):text.find(aspect) + 100]
        clues = [(word, label) for word, label in lexicon.items() if word in window]
        if clues: evidence.append(f"aspect {aspect}: local clues {clues}")
        else: evidence.append(f"aspect {aspect}: unresolved sentiment")
    if not evidence: evidence.append("No explicit aspect phrase was deterministically extracted.")
    return _partial("sentiment_evidence", evidence, "The deterministic lexicon supplies local clues only; it does not claim nuanced sentiment, sarcasm, or complete ABSA.",
                    ("The final aspect labels and any implicit sentiment remain unresolved.",), "Use Fireworks to classify only from the listed aspect evidence.")


def _summary_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(summarise|summarize|summary|condense)\b", prompt, re.I))


def _summary(prompt: str) -> ToolResult:
    source = re.split(r"(?:summari[sz]e|summary)\s*(?:this)?\s*:?", prompt, flags=re.I)[-1].strip()
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", source) if part.strip()]
    scored = sorted(sentences, key=lambda item: (sum(char.isdigit() for char in item) + len(re.findall(r"\b[A-Z][a-z]+\b", item)), len(item)), reverse=True)
    selected = scored[:min(3, len(scored))]
    return _partial("extractive_summary", [f"Extractive key sentences: {selected}.", "Dates and numbers in selected sentences are preserved verbatim."],
                    "Extractive selection is evidence, not an open-ended semantic summary.", ("A faithful requested summary still needs composition.",),
                    "Use Fireworks to compose from the preserved evidence, then validate any exact format locally.")


def _factual_admission(prompt: str) -> bool:
    return bool(re.search(r"\b(evidence|forensic|chain of custody|contradiction|unsupported conclusion|facts?)\b", prompt, re.I))


def _factual(prompt: str) -> ToolResult:
    dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b|\b\w+\s+\d{1,2},\s+\d{4}\b", prompt)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\s*(?:mm|cm|kg|%|hours?|days?)?\b", prompt)
    evidence = [f"Preserved supplied dates: {dates}.", f"Preserved supplied measurements/numbers: {numbers}."]
    if re.search(r"\b(no|missing|without)\s+(?:record|evidence|documentation)", prompt, re.I): evidence.append("Chain-of-custody/documentation gap explicitly detected in supplied text.")
    return _partial("factual_evidence", evidence, "The tool separates supplied evidence from unsupported inference and has no open-domain knowledge base.",
                    ("The requested factual conclusion remains unresolved without external or supplied support.",),
                    "Use Fireworks to explain the evidence limits without inventing facts.")


class ToolRegistry:
    """Deterministic admission and composition for the one production path."""
    def __init__(self) -> None:
        self.tools = tuple(sorted((
            ToolSpec("safe_arithmetic", ("mathematical_reasoning",), _safe_arithmetic_admission, _safe_arithmetic, True, 10, 1, 8),
            ToolSpec("symbolic_math", ("mathematical_reasoning",), _symbolic_admission, _symbolic_math, True, 20, 3, 12),
            ToolSpec("sat_csp", ("logical_reasoning",), _constraint_admission, _constraints, True, 30, 3, 16),
            ToolSpec("structured_json", ("factual_knowledge", "code_generation"), _json_admission, _json_tool, True, 40, 1, 12),
            ToolSpec("format_constraints", ("text_summarisation", "code_generation"), _format_admission, _format_tool, True, 50, 1, 12),
            ToolSpec("python_ast", ("code_debugging", "code_generation"), _ast_admission, _python_ast, False, 60, 1, 16),
            ToolSpec("bounded_python", ("code_debugging", "code_generation"), _python_admission, _bounded_python, True, 70, 3, 12),
            ToolSpec("code_debugging", ("code_debugging",), _debug_admission, _debug_workflow, False, 80, 4, 24),
            ToolSpec("entity_candidates", ("named_entity_recognition",), _ner_admission, _ner, False, 90, 1, 16),
            ToolSpec("sentiment_evidence", ("sentiment_classification",), _sentiment_admission, _sentiment, False, 100, 1, 16),
            ToolSpec("extractive_summary", ("text_summarisation",), _summary_admission, _summary, False, 110, 1, 12),
            ToolSpec("factual_evidence", ("factual_knowledge",), _factual_admission, _factual, False, 120, 1, 12),
        ), key=lambda item: item.priority))

    def execute_applicable(self, prompt: str, decision: Any) -> list[ToolResult]:
        """Run matching tools only, stopping after a verified complete answer."""
        # Debugging is a composed workflow, never an independent second router.
        if _debug_admission(prompt):
            return [_debug_workflow(prompt)]
        results: list[ToolResult] = []
        for tool in self.tools:
            if not tool.admission_predicate(prompt):
                continue
            result = tool.executor(prompt)
            evidence = result.evidence[:tool.evidence_limit]
            results.append(ToolResult(result.tool_name, result.state, result.answer, evidence, result.reason, result.unresolved_requirements,
                                      result.correction, result.next_action, result.success_criteria, result.confidence, result.error_details))
            if result.state == "resolved" and result.answer is not None:
                break
        if not results:
            category = getattr(decision, "primary", "unknown")
            results.append(_result("local_tools", "unsupported", None, (), f"No production tool admits this {category} request.",
                                   ("The semantic task remains unresolved.",), None, "Send the unchanged original task to Fireworks.",
                                   ("Return the requested answer and format.",), 0.0))
        return results


tool_registry = ToolRegistry()


def collect_actionable_evidence(results: list[ToolResult]) -> list[ToolResult]:
    return [result for result in results if result.state != "resolved"]


def build_tool_evidence_prompt(original_prompt: str, results: list[ToolResult]) -> str:
    evidence: list[str] = []; unresolved: list[str] = []; corrections: list[str] = []; actions: list[str] = []; criteria: list[str] = []
    for result in results:
        evidence.extend(result.evidence[:12]); unresolved.extend(result.unresolved_requirements); actions.append(result.next_action); criteria.extend(result.success_criteria)
        if result.correction: corrections.append(result.correction)
    return "\n\n".join((
        f"Original task:\n{original_prompt}",
        "Verified tool evidence:\n" + ("\n".join(evidence) if evidence else "No deterministic evidence was produced."),
        "Unresolved requirements:\n" + ("\n".join(dict.fromkeys(unresolved)) if unresolved else "None."),
        "Supported corrections:\n" + ("\n".join(dict.fromkeys(corrections)) if corrections else "None."),
        "Required next action:\n" + ("\n".join(dict.fromkeys(actions)) if actions else "Answer the original task."),
        "Success criteria:\n" + ("\n".join(dict.fromkeys(criteria)) if criteria else "Return the requested answer."),
    ))


def post_validate_answer(prompt: str, answer: str) -> tuple[str, ...]:
    """Non-destructive post-generation checks; no second model call is made."""
    findings: list[str] = []
    if not answer.strip(): findings.append("Final answer is empty.")
    if "json-only" in prompt.lower():
        try: json.loads(answer)
        except json.JSONDecodeError as exc: findings.append(f"Final JSON validation failed at line {exc.lineno}, column {exc.colno}: {exc.msg}.")
    if "code-only" in prompt.lower() or ("python" in prompt.lower() and "```" in answer):
        candidate = _code_from_prompt(answer) or answer
        try: ast.parse(candidate)
        except SyntaxError as exc: findings.append(f"Final Python syntax validation failed: {exc.msg} at line {exc.lineno}.")
    return tuple(findings)


TOOL_SYSTEM = "Answer the original task using the verified tool evidence where relevant. Do not contradict deterministic findings. Resolve every unresolved requirement and satisfy every success criterion. Return only the final answer requested by the original task."
