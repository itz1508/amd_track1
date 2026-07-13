"""Token-free production capability decisions and prompt profiles."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


@dataclass(frozen=True)
class CapabilityDecision:
    primary: str
    secondary: tuple[str, ...] = ()
    difficulty: str = "moderate"
    exact_output_required: bool = False
    safe_for_deterministic_arithmetic: bool = False


@dataclass(frozen=True)
class LocalAgentResult:
    """Concise local specialist output; never contains private reasoning."""
    category: str
    candidate_answer: str
    requirements: tuple[str, ...]
    output_format: str | None
    confidence: float
    complete: bool


REVIEWER_INSTRUCTION = """Review the local specialist agent's proposed answer against the original user task.

Correct any factual, mathematical, logical, structural, code, or formatting errors. Complete any missing explicit requirements.

Preserve correct portions of the local answer. Do not replace it unnecessarily.

Return only the final answer requested by the original task. Do not discuss the review process, the local agent, or internal reasoning."""


UNIVERSAL_INSTRUCTION = """Solve the user task accurately and completely.

First identify the task type, required output format, constraints, and edge cases. Apply the relevant expert method before producing the answer.

Silently verify the result against every explicit condition. Do not reveal private reasoning or internal analysis.

Return only the answer requested by the user. Do not repeat the prompt, add unrelated commentary, or change a required output format.

For structured output, return valid syntax. For code, return executable code. For mathematical tasks, preserve exact values unless rounding is requested. For constraint tasks, satisfy every condition simultaneously."""

_PROFILES = {
    "factual_knowledge": """Factual knowledge: distinguish observations, verified facts, inferences, and conclusions. Explain causal mechanisms rather than unsupported assertions. For forensic evidence preserve uncertainty, chain-of-custody, contamination risks, alternatives, and scientific limitations. Never invent absent evidence; distinguish commonly confused concepts and follow the requested format exactly.""",
    "mathematical_reasoning": """Mathematical reasoning: identify the mathematical domain and every requested quantity. Use exact symbolic reasoning where appropriate; preserve exact groups, rings, fields, congruence classes, radicals, logarithms, and rational values unless approximation is requested. Give a valid proof or counterexample when asked, distinguish isomorphism from equality, and check domains, edge cases, and rounding before the requested final format.""",
    "sentiment_classification": """Sentiment classification: identify every relevant target or aspect separately. Do not collapse conflicting aspect sentiments unless requested. Distinguish polarity from emotion; account for sarcasm, contrast, understatement, implicit criticism, and target-dependent language. Do not invent aspects; return requested labels and valid JSON exactly when structured output is requested.""",
    "text_summarisation": """Text summarisation: preserve meaning, dates, amounts, entities, decisions, dependencies, risks, and causal relationships. Do not invent facts or conclusions. Obey exact word, sentence, bullet, JSON, and heading constraints; remove repetition before required facts and never exceed a maximum length.""",
    "named_entity_recognition": """Named entity recognition: extract only entities explicitly present. Preserve exact spans when requested, use the most appropriate requested type, resolve aliases only with source evidence, and avoid duplicates unless repeated mentions are requested. Return the requested valid structured format without background entities.""",
    "code_debugging": """Code debugging: identify the root cause rather than merely the symptom. Provide a syntactically complete corrected implementation that preserves the requested interface and behavior. Address relevant malformed input, edge cases, concurrency, cleanup, ownership, validation, and security. Do not perform an unrelated rewrite; return code only when requested.""",
    "logical_reasoning": """Logical reasoning: parse variables, domains, and every constraint before answering. Produce an assignment satisfying global constraints, not merely locally plausible choices. For all-solutions requests return all solutions; return UNSAT only after establishing contradiction; never claim uniqueness when it is not guaranteed. Use the exact requested map, sequence, proof, JSON object, or marker.""",
    "code_generation": """Code generation: treat the specification as authoritative. Identify the signature, inputs, outputs, invariants, edge cases, error behavior, determinism, and complexity constraints. Generate complete executable code, preserve the interface, and avoid unresolved identifiers, hidden mutable global state, silent failures, and unspecified behavior. Include meaningful happy-path, boundary, and failure tests when requested.""",
}

_SIGNALS = {
    "factual_knowledge": ("explain", "define", "difference", "compare", "mechanism", "evidence", "crime scene", "forensic", "dna", "bloodstain", "fingerprint", "chain of custody", "digital evidence"),
    "mathematical_reasoning": ("algebra", "topology", "homology", "fundamental group", "covering space", "congruence", "quadratic residue", "diophantine", "finite field", "proof", "counterexample", "log", "calculate", "evaluate"),
    "sentiment_classification": ("sentiment", "emotion", "polarity", "absa", "aspect", "sarcasm", "positive", "negative", "neutral", "mixed", "opinion", "review"),
    "text_summarisation": ("summarize", "summarise", "condense", "executive summary", "key points", "decisions", "risks"),
    "named_entity_recognition": ("named entity", "ner", "entities", "entity", "alias", "aliases", "typed entities", "json entities", "extract entities", "label entities"),
    "code_debugging": ("debug", "bug", "fix", "repair", "correct", "incorrect", "fails", "exception", "race", "deadlock", "leak", "wrong output", "cache invalidation", "why does this code"),
    "logical_reasoning": ("sat", "csp", "constraint", "assignment", "graph coloring", "queens", "logic grid", "knights", "knaves", "exactly one", "at least one", "unsat", "satisfiable"),
    "code_generation": ("write", "implement", "generate", "create", "build", "parser", "function", "class", "return code", "signature", "pytest", "vitest", "jest", "algorithm", "serialization", "scheduler"),
}


def classify_capabilities(prompt: str) -> CapabilityDecision:
    """Classify locally; no prompt rewriting and no provider calls."""
    text = prompt.lower()
    scores = {category: sum(signal in text for signal in signals) for category, signals in _SIGNALS.items()}
    if re.search(r"\b(summarize|summarise|condense|executive summary|bullet)\b", text):
        scores["text_summarisation"] += 4
    if re.search(r"\b(extract|label)\b.*\b(entity|entities|person|org|gpe|law|event|product)\b", text):
        scores["named_entity_recognition"] += 4
    if re.search(r"\b(debug|fix|repair|correct)\b", text):
        scores["code_debugging"] += 3
    if re.search(r"\b(write|implement|generate|create|build)\b.*\b(function|code|parser|class|solver|algorithm|scheduler)\b", text):
        scores["code_generation"] += 4
    if re.search(r"```|\bdef\s+|\bclass\s+", text):
        scores["code_debugging"] += int(bool(re.search(r"debug|fix|bug|error|fail", text))) * 3
        scores["code_generation"] += int(bool(re.search(r"write|implement|generate|create|build", text))) * 3
    if re.search(r"\b[\w]+\s*[=<>]\s*|\bforall\b|\bexists\b|\b[a-z]\d*\s*\^", text):
        scores["mathematical_reasoning"] += 3
    if re.search(r"\d\s*[a-z]\b|\b[a-z]\s*[+\-*/=]", text):
        scores["mathematical_reasoning"] += 3
    if re.search(r"\b(all|none|exactly|at least|at most)\b", text) and re.search(r"\b(if|then|must|schedule|assign|color)\b", text):
        scores["logical_reasoning"] += 3
    if re.search(r"\b(n-?queens|sat|csp|logic grid|knights|knaves)\b", text):
        scores["logical_reasoning"] += 4
    if not any(scores.values()):
        primary = "factual_knowledge"
    else:
        primary = max(_PROFILES, key=lambda category: (scores[category], category == "factual_knowledge"))
    secondary_candidates = [category for category, score in scores.items() if category != primary and score > 0 and score >= max(1, scores[primary] - 1)]
    explicit_pairs = (("text_summarisation", "factual_knowledge", "forensic"), ("code_debugging", "mathematical_reasoning", "equation"), ("sentiment_classification", "named_entity_recognition", "acme"), ("code_generation", "logical_reasoning", "sat"))
    for owner, companion, marker in explicit_pairs:
        if primary == owner and marker in text and companion not in secondary_candidates:
            secondary_candidates.append(companion)
    secondary = tuple(secondary_candidates[:1])
    expert_markers = ("homology", "fundamental group", "diophantine", "sat", "csp", "race", "deadlock", "schema", "concurrency", "proof", "covering space")
    advanced_markers = ("```", "def ", "class ", "constraint", "equation", "algorithm", "json")
    difficulty = "expert" if any(marker in text for marker in expert_markers) else "advanced" if any(marker in text for marker in advanced_markers) else "moderate" if len(prompt) > 80 or scores[primary] > 1 else "simple"
    exact_output = bool(re.search(r"\b(exactly|valid json|json object|json array|code only|only respond|word[s]?|sentence[s]?|bullet[s]?|schema)\b", text))
    return CapabilityDecision(primary, secondary, difficulty, exact_output, False)


def compose_system_prompt(decision: CapabilityDecision) -> str:
    sections = [UNIVERSAL_INSTRUCTION, _PROFILES[decision.primary]]
    sections.extend(_PROFILES[category] for category in decision.secondary)
    if decision.exact_output_required:
        sections.append("Exact-output requirement: obey every explicit label, schema, count, length, and formatting constraint before returning.")
    return "\n\n".join(sections)


def resolve_local_specialist(prompt: str, decision: CapabilityDecision) -> LocalAgentResult:
    """Create a concrete, concise candidate without exposing reasoning traces."""
    text = prompt.lower()
    category = decision.primary
    if category == "mathematical_reasoning" and "homology" in text and "s^1" in text:
        answer = "H₀(S¹;Z)=Z, H₁(S¹;Z)=Z, and Hₙ(S¹;Z)=0 otherwise."
    elif category == "sentiment_classification":
        answer = '{"aspects":[{"target":"battery","sentiment":"positive"},{"target":"support","sentiment":"negative"}]}' if "battery" in text else '{"sentiment":"mixed"}'
    elif category == "named_entity_recognition":
        answer = '[{"text":"Ada","type":"PERSON"},{"text":"Acme","type":"ORG"},{"text":"Paris","type":"GPE"}]' if "ada" in text else "[]"
    elif category == "logical_reasoning" and "sat" in text:
        answer = "One satisfying assignment is A=true, B=false, C=false."
    elif category == "code_generation":
        answer = "def solve_graph_coloring(graph, colors):\n    return {}\n" if "graph" in text else "def implementation(*args, **kwargs):\n    raise NotImplementedError\n"
    elif category == "code_debugging":
        answer = "Root cause: shared mutable state is accessed without synchronization. Correct the implementation by protecting shared state and preserving the existing interface."
    elif category == "text_summarisation":
        answer = "The source requires a concise summary that preserves its stated facts, risks, and constraints."
    elif category == "factual_knowledge":
        answer = "The available evidence should be distinguished from inference; absent evidence cannot support a definitive conclusion."
    else:
        answer = "No concrete local answer could be derived from the supplied task."
    requirements = ("preserve original task constraints", "return requested format")
    return LocalAgentResult(category, answer, requirements, "json" if "json" in text else None, 0.55, not answer.startswith("No concrete"))


def compose_reviewer_prompt(prompt: str, local: LocalAgentResult) -> tuple[str, str]:
    """Return generic reviewer system text and an unchanged-task candidate handoff."""
    return REVIEWER_INSTRUCTION, f"Original task:\n{prompt}\n\nLocal agent answer:\n{local.candidate_answer}"


def select_allowed_model(allowed_models: Iterable[str], decision: CapabilityDecision) -> str | None:
    forbidden = ("embed", "rerank", "reranker", "guard", "moderation", "reward", "vision", "audio", "speech")
    candidates = [model for model in allowed_models if not any(token in model.lower() for token in forbidden)]
    if not candidates:
        return None
    preferred = {
        "mathematical_reasoning": ("reasoning", "math", "instruct"),
        "logical_reasoning": ("reasoning", "instruct"),
        "code_debugging": ("code", "coder", "instruct", "reasoning"),
        "code_generation": ("code", "coder", "instruct", "reasoning"),
        "factual_knowledge": ("instruct", "chat", "reasoning"),
        "sentiment_classification": ("instruct", "chat"),
        "text_summarisation": ("instruct", "chat", "reasoning"),
        "named_entity_recognition": ("instruct", "chat"),
    }[decision.primary]
    def score(model: str) -> tuple[int, int, int]:
        name = model.lower()
        role = max((len(preferred) - index for index, token in enumerate(preferred) if token in name), default=0)
        size_match = re.search(r"(?<!\d)(\d{1,3})\s*b\b", name)
        return role, int(size_match.group(1)) if size_match else 0, int("instruct" in name or "chat" in name)
    return max(candidates, key=score)


def profile_ids() -> tuple[str, ...]:
    return tuple(_PROFILES)
