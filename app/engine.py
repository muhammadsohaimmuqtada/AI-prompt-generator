"""
AI Prompt Generator - model-aware prompting engine.

This module converts user inputs into model-specific prompt templates
using current guidance from official vendor docs.
"""

from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class UserInput:
    role: str
    action: str
    context: str
    expectation: str
    examples: str = ""
    ask_clarifying_questions: bool = True
    use_planning: bool = False
    use_self_check: bool = False
    auto_enhance: bool = True
    smart_expand_lazy_input: bool = True


@dataclass
class PromptReport:
    overall_score: int
    field_scores: dict[str, int]
    strengths: list[str]
    suggestions: list[str]
    auto_upgrades: list[str]


MODELS = {
    "Claude Engine (Anthropic)": "claude",
    "ChatGPT Engine (OpenAI)": "chatgpt",
    "Gemini Engine (Google)": "gemini",
    "DeepSeek Engine": "deepseek",
}

MODEL_DESCRIPTIONS = {
    "claude": "Best with clear XML-tagged sections, intent optimization, and few-shot examples.",
    "chatgpt": "Best with strong system/user separation and explicit objective/deliverable contracts.",
    "gemini": "Best with direct structure, constraints, and optimized task blueprinting.",
    "deepseek": "Best with concise system+user messages, explicit structure, and JSON controls.",
}


def _clean(text: str) -> str:
    return text.strip()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


_COMMON_TYPO_FIXES = {
    "writtinf": "writing",
    "propmpt": "prompt",
    "propmpts": "prompts",
    "automotcally": "automatically",
    "automatocally": "automatically",
    "utilize it prperly": "utilize it properly",
    "chanfe": "change",
    "accordingf": "according",
    "modelk": "model",
    "safteyh": "safety",
    "annoting": "annoying",
    "dont": "don't",
    "givinf": "giving",
    "ot": "to",
    "tje": "the",
    "hsi": "his",
}


_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "your",
    "have",
    "will",
    "must",
    "should",
    "make",
    "give",
    "then",
    "only",
    "within",
    "about",
    "under",
    "after",
    "before",
}


def _autocorrect_common_typos(text: str) -> str:
    corrected = text
    for wrong, right in _COMMON_TYPO_FIXES.items():
        corrected = re.sub(rf"\b{re.escape(wrong)}\b", right, corrected, flags=re.IGNORECASE)
    return corrected


def _normalize_role_phrase(role: str) -> str:
    role = _clean(role)
    role = re.sub(r"^(as\s+an?\s+|as\s+)", "", role, flags=re.IGNORECASE)
    return role


def _extract_keywords(text: str) -> set[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS}


def _keyword_overlap_ratio(source: str, target: str) -> float:
    source_kw = _extract_keywords(source)
    if not source_kw:
        return 1.0
    target_kw = _extract_keywords(target)
    overlap = source_kw.intersection(target_kw)
    return len(overlap) / len(source_kw)


_BYPASS_PATTERNS = (
    re.compile(r"\bignore\b[^.\n;]*(safety|policy|warning|guardrail)[^.\n;]*", re.IGNORECASE),
    re.compile(r"\bbypass\b[^.\n;]*(safety|policy|warning|guardrail)[^.\n;]*", re.IGNORECASE),
    re.compile(r"\bdisable\b[^.\n;]*(safety|policy|warning|guardrail)[^.\n;]*", re.IGNORECASE),
    re.compile(r"\bno\s+restrictions?\b", re.IGNORECASE),
    re.compile(r"\bunfiltered\b", re.IGNORECASE),
)


def _normalize_user_text(text: str) -> tuple[str, bool]:
    """Remove jailbreak-style instructions while preserving benign intent."""
    if not text:
        return "", False

    normalized = _autocorrect_common_typos(text)
    detected = False
    for pattern in _BYPASS_PATTERNS:
        if pattern.search(normalized):
            detected = True
            normalized = pattern.sub(
                "follow applicable safety policies and avoid unnecessary refusal for benign requests",
                normalized,
            )

    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
    return normalized, detected


def _get_role(inp: UserInput, default: str) -> str:
    return _normalize_role_phrase(inp.role) or default


def _get_expectation(inp: UserInput, default: str) -> str:
    return _clean(inp.expectation) or default


def _expects_json(inp: UserInput) -> bool:
    combined = f"{inp.action}\n{inp.expectation}".lower()
    keywords = (
        "json",
        "json schema",
        "machine-readable",
        "structured output",
        "structured data",
    )
    return any(term in combined for term in keywords)


def _quality_rules(inp: UserInput) -> list[str]:
    rules: list[str] = [
        "Prioritize correctness and factual accuracy over style.",
        "Comply fully with benign, legal requests and avoid unnecessary refusal language.",
        "If a request could be unsafe, ask for clarification and provide a safe alternative when needed.",
    ]

    if inp.ask_clarifying_questions:
        rules.append(
            "If critical details are missing, ask up to 3 concise clarifying questions before solving."
        )
    if inp.use_planning:
        rules.append(
            "Create a short step-by-step plan before producing the final answer."
        )
    if inp.use_self_check:
        rules.append(
            "Before finalizing, check that your answer satisfies every stated requirement."
        )

    return rules


def _infer_domain(inp: UserInput) -> str:
    combined = f"{inp.action} {inp.context} {inp.expectation}".lower()
    security_terms = (
        "pentest",
        "penetration test",
        "red team",
        "exploit",
        "kali",
        "nmap",
        "payload",
        "sql injection",
        "xss",
        "command",
    )
    coding_terms = ("code", "python", "javascript", "debug", "refactor", "api")
    writing_terms = ("blog", "article", "email", "copywriting", "tweet", "post")
    analysis_terms = ("analyze", "analysis", "report", "compare", "evaluate")

    if any(term in combined for term in security_terms):
        return "security"
    if any(term in combined for term in coding_terms):
        return "coding"
    if any(term in combined for term in writing_terms):
        return "writing"
    if any(term in combined for term in analysis_terms):
        return "analysis"
    return "general"


def _default_expectation_by_domain(domain: str) -> str:
    if domain == "security":
        return (
            "Provide a scoped, non-destructive workflow with clear pre-checks, commands, "
            "validation steps, and cleanup guidance."
        )
    if domain == "coding":
        return "Provide implementation-ready output with code, rationale, and quick validation."
    if domain == "writing":
        return "Provide polished output with strong structure, tone control, and concise wording."
    if domain == "analysis":
        return "Provide a structured analysis with key findings, evidence, and recommendations."
    return "Provide a clear, complete, and practical response."


def _default_role_by_domain(domain: str) -> str:
    if domain == "security":
        return "an authorized security assessment assistant focused on defensive validation"
    if domain == "coding":
        return "a senior software engineer and debugger"
    if domain == "writing":
        return "a professional editor and copywriter"
    if domain == "analysis":
        return "a structured analyst focused on evidence-based conclusions"
    return "a practical domain expert"


def _default_example_stub(domain: str) -> str:
    if domain == "security":
        return (
            "Input: Validate a web target within authorized scope.\n"
            "Output:\n"
            "1) Scope check\n"
            "2) Safe recon steps\n"
            "3) Verification checkpoints\n"
            "4) Findings summary with remediation priority"
        )
    if domain == "coding":
        return (
            "Input: Fix a bug in function X.\n"
            "Output:\n"
            "1) Root cause\n"
            "2) Updated code\n"
            "3) Edge cases\n"
            "4) Quick test checklist"
        )
    if domain == "writing":
        return (
            "Input: Draft a concise outreach email.\n"
            "Output:\n"
            "Subject: ...\n"
            "Body: ...\n"
            "CTA: ..."
        )
    if domain == "analysis":
        return (
            "Input: Analyze KPI trend.\n"
            "Output:\n"
            "Observations\n"
            "Drivers\n"
            "Risks\n"
            "Recommendations"
        )
    return (
        "Input: User request.\n"
        "Output:\n"
        "Objective\n"
        "Approach\n"
        "Execution\n"
        "Validation"
    )


def _is_low_effort_action(text: str) -> bool:
    clean = _clean(text).lower()
    if not clean:
        return True
    if _word_count(clean) <= 4:
        return True

    vague_patterns = (
        "help",
        "do it",
        "commands",
        "terminal ready commands",
        "write something",
        "make it better",
        "fix it",
    )
    return any(p in clean for p in vague_patterns)


def _normalize_rough_expectation_text(expectation: str) -> tuple[str, bool]:
    text = _clean(expectation)
    if not text:
        return text, False

    lowered = text.lower()
    mentions_warning = any(token in lowered for token in ("warn", "warning", "disclaimer", "refusal"))
    asks_to_reduce_warning = any(token in lowered for token in ("dont", "don't", "no ", "avoid", "skip", "trigger"))

    if mentions_warning and asks_to_reduce_warning:
        return (
            "Use a professional tone and avoid unnecessary cautionary disclaimers for benign, authorized requests.",
            True,
        )
    return text, False


def _resolve_expectation_conflicts(expectation: str) -> tuple[str, str | None]:
    text = _clean(expectation)
    if not text:
        return text, None

    lowered = text.lower()
    concise = any(token in lowered for token in ("concise", "brief", "short"))
    detailed = any(token in lowered for token in ("detailed", "in-depth", "comprehensive"))
    wants_json = any(token in lowered for token in ("json", "json schema"))
    wants_prose = any(token in lowered for token in ("paragraph", "narrative", "natural language"))

    if concise and detailed:
        return (
            "Keep the output concise while preserving all critical details and execution steps.",
            "Expectation conflict was resolved: concise vs detailed was balanced for reliability.",
        )
    if wants_json and wants_prose:
        return (
            "Return JSON as the primary output, with brief explanatory notes only in dedicated fields.",
            "Expectation conflict was resolved: prose + JSON was normalized to structured-first output.",
        )

    return text, None


def _expectation_has_signal(expectation: str) -> bool:
    lowered = _clean(expectation).lower()
    if not lowered:
        return False
    signal_terms = (
        "tone",
        "professional",
        "concise",
        "detailed",
        "step",
        "json",
        "table",
        "bullet",
        "format",
        "style",
        "warning",
        "disclaimer",
        "no fluff",
        "terminal",
        "commands",
    )
    return any(term in lowered for term in signal_terms)


def _expectation_has_format_signal(expectation: str) -> bool:
    lowered = _clean(expectation).lower()
    format_terms = ("json", "table", "bullet", "checklist", "sections", "format", "markdown")
    return any(term in lowered for term in format_terms)


def _input_richness_score(inp: UserInput) -> int:
    """Heuristic signal for how complete and specific user input already is."""
    score = 0
    score += min(30, _word_count(inp.action) * 2)
    score += min(25, _word_count(inp.context))
    score += min(20, _word_count(inp.expectation) * 2)
    if _clean(inp.examples):
        score += 15
    if _word_count(inp.role) >= 3:
        score += 10
    return _bounded_score(score)


def _upgrade_context_text(context: str, domain: str) -> tuple[str, str | None]:
    context = _clean(_autocorrect_common_typos(context))
    if not context:
        return context, None

    note: str | None = None
    if _word_count(context) >= 12 and "\n" not in context:
        context = f"{context}\n\nKey constraints:\n- Respect provided scope and constraints.\n- Ask when assumptions are required."
        note = "Context was reformatted with explicit constraints for better model grounding."

    if domain == "security":
        lower_context = context.lower()
        if not any(token in lower_context for token in ("authorized", "permission", "approved", "scope")):
            context += "\n- Security scope reminder: operate only within explicitly authorized boundaries."
            note = "Security context was hardened with explicit scope boundaries."

    return context, note


def _upgrade_expectation_text(expectation: str) -> tuple[str, str | None]:
    expectation = _clean(_autocorrect_common_typos(expectation))
    if not expectation:
        return expectation, None

    conflict_fixed, conflict_note = _resolve_expectation_conflicts(expectation)
    if conflict_note:
        return conflict_fixed, conflict_note

    normalized, changed = _normalize_rough_expectation_text(expectation)
    if changed:
        return normalized, "Expectation wording was normalized for clarity and reliability."

    if not _expectation_has_format_signal(expectation):
        expectation += " Use clear sections and bullet points where helpful."
        return expectation, "Expectation was upgraded with explicit output-structure guidance."

    return expectation, None


def _upgrade_examples_text(examples: str) -> tuple[str, str | None]:
    examples = _clean(_autocorrect_common_typos(examples))
    if not examples:
        return examples, None

    lowered = examples.lower()
    if "input:" in lowered and "output:" in lowered:
        return examples, None

    if "\n" not in examples and _word_count(examples) >= 4:
        structured = f"Input: {examples}\nOutput: Follow the requested style, constraints, and format exactly."
        return structured, "Examples were auto-structured into Input/Output format."

    return examples, None


def _expand_action_by_domain(domain: str, context: str) -> str:
    ctx = _clean(context)
    if domain == "security":
        base = (
            "Create a terminal-ready, step-by-step workflow for an authorized security assessment. "
            "Include scoped recon, safe validation checks, and verification after each command."
        )
    elif domain == "coding":
        base = (
            "Produce an implementation-ready solution with clear steps, code-level decisions, and "
            "a quick verification checklist."
        )
    elif domain == "writing":
        base = (
            "Produce a polished draft with clear structure, audience-appropriate tone, and a short "
            "self-edit pass for clarity."
        )
    elif domain == "analysis":
        base = (
            "Deliver a structured analysis that separates observations, reasoning, and actionable "
            "recommendations."
        )
    else:
        base = (
            "Provide a complete, practical response with a clear approach, execution details, and "
            "final validation."
        )

    if ctx:
        return f"{base}\nUse this context as scope and constraints:\n{ctx}"
    return base


@dataclass
class _LazyInferenceResult:
    prepared: UserInput
    notes: list[str]


def _apply_lazy_input_inference(inp: UserInput) -> _LazyInferenceResult:
    if not inp.smart_expand_lazy_input:
        return _LazyInferenceResult(prepared=inp, notes=[])

    domain = _infer_domain(inp)
    notes: list[str] = []

    role = _normalize_role_phrase(_autocorrect_common_typos(inp.role))
    action = _clean(_autocorrect_common_typos(inp.action))
    context = _clean(_autocorrect_common_typos(inp.context))
    expectation = _clean(_autocorrect_common_typos(inp.expectation))
    examples = _clean(_autocorrect_common_typos(inp.examples))

    if not role or _word_count(role) <= 2:
        role = _default_role_by_domain(domain)
        notes.append("Role was short/empty, so a stronger domain role was inferred.")
    elif "assistant" in role.lower() and "focused on" not in role.lower():
        role = f"{role} focused on precise, practical execution"
        notes.append("Role was upgraded with a precision-focused capability descriptor.")

    if _is_low_effort_action(action):
        action = _expand_action_by_domain(domain, context)
        notes.append("Action looked minimal, so it was expanded into an executable objective.")
    elif _word_count(action) >= 8 and not action.endswith((".", "!", "?")):
        action += "."
        notes.append("Action phrasing was polished for cleaner instruction parsing.")

    if not context:
        context = (
            "No additional context was provided. The assistant should ask focused clarifying "
            "questions before taking assumptions that affect output quality."
        )
        notes.append("Context was empty, so a clarification-first context guard was added.")
    else:
        context, context_note = _upgrade_context_text(context, domain)
        if context_note:
            notes.append(context_note)

    if not expectation:
        expectation = _default_expectation_by_domain(domain)
        notes.append("Expectation was weak/empty, so output requirements were auto-filled.")
    elif _word_count(expectation) <= 5 and not _expectation_has_signal(expectation):
        expectation = _default_expectation_by_domain(domain)
        notes.append("Expectation was weak/empty, so output requirements were auto-filled.")
    else:
        expectation, expectation_note = _upgrade_expectation_text(expectation)
        if expectation_note:
            notes.append(expectation_note)

    if examples:
        examples, examples_note = _upgrade_examples_text(examples)
        if examples_note:
            notes.append(examples_note)

    if not examples and inp.auto_enhance:
        examples = _default_example_stub(domain)
        notes.append("Few-shot examples were missing, so a reusable example scaffold was inserted.")

    prepared = UserInput(
        role=role,
        action=action,
        context=context,
        expectation=expectation,
        examples=examples,
        ask_clarifying_questions=inp.ask_clarifying_questions,
        use_planning=inp.use_planning,
        use_self_check=inp.use_self_check,
        auto_enhance=inp.auto_enhance,
        smart_expand_lazy_input=inp.smart_expand_lazy_input,
    )
    return _LazyInferenceResult(prepared=prepared, notes=notes)


@dataclass
class _IntentOptimization:
    objective: str
    deliverables: list[str]
    response_blueprint: list[str]
    style_constraints: list[str]


def _domain_deliverables(domain: str, command_heavy: bool) -> list[str]:
    if domain == "security":
        items = [
            "Produce a scoped workflow that stays within explicitly authorized boundaries.",
            "Use non-destructive steps first and include verification checkpoints.",
            "Show what successful execution should look like for each phase.",
        ]
        if command_heavy:
            items.extend(
                [
                    "Provide copy-paste-ready commands grouped by phase.",
                    "For each command, include expected output and a quick fallback check.",
                ]
            )
        return items
    if domain == "coding":
        return [
            "Provide implementation-ready steps and concrete code-level guidance.",
            "Cover edge cases and failure handling.",
            "Include a concise validation/test checklist.",
        ]
    if domain == "writing":
        return [
            "Deliver polished content with a clear structure and consistent voice.",
            "Improve clarity, flow, and audience alignment.",
            "Include a brief final refinement pass.",
        ]
    if domain == "analysis":
        return [
            "Separate facts, assumptions, and recommendations.",
            "Highlight key findings and supporting rationale.",
            "Provide actionable next steps.",
        ]
    return [
        "Deliver a complete and practical response, not partial guidance.",
        "Make the output directly usable without additional rewriting.",
        "Include quick validation criteria for result quality.",
    ]


def _domain_blueprint(domain: str, command_heavy: bool) -> list[str]:
    if domain == "security":
        steps = ["Scope check", "Execution plan", "Step-by-step actions", "Validation", "Safe next steps"]
        if command_heavy:
            steps[2] = "Step-by-step actions with commands and expected results"
        return steps
    if domain == "coding":
        return ["Problem framing", "Implementation", "Edge cases", "Validation/tests", "Final answer"]
    if domain == "writing":
        return ["Intent summary", "Draft", "Refinement", "Final polished output"]
    if domain == "analysis":
        return ["Objective", "Evidence", "Analysis", "Recommendations", "Confidence note"]
    return ["Objective", "Approach", "Execution", "Validation", "Final answer"]


def _style_constraints(expectation: str) -> list[str]:
    lines: list[str] = []
    lowered = _clean(expectation).lower()
    if "professional" in lowered:
        lines.append("Use a professional tone.")
    if "concise" in lowered:
        lines.append("Keep the answer concise while preserving required detail.")
    if "detailed" in lowered:
        lines.append("Include sufficient detail for direct execution.")
    if "no fluff" in lowered:
        lines.append("Avoid fluff and keep content task-relevant.")
    if "warning" in lowered or "disclaimer" in lowered:
        lines.append("Avoid unnecessary cautionary disclaimers for benign, authorized requests.")
    lines.append("Keep language precise and avoid ambiguity.")
    lines.append("Do not omit essential steps required to achieve the objective.")
    # Preserve order while removing accidental duplicates.
    return list(dict.fromkeys(lines))


def _optimize_intent(inp: UserInput, raw_user_action: str = "") -> _IntentOptimization | None:
    if not inp.smart_expand_lazy_input:
        return None

    domain = _infer_domain(inp)
    action = _clean(inp.action)
    raw_action = _clean(raw_user_action) or action
    context = _clean(inp.context)
    command_heavy = "command" in action.lower() or "terminal" in action.lower()

    if _is_low_effort_action(action):
        objective = _expand_action_by_domain(domain, context)
    else:
        objective = action
        if not objective.endswith((".", "!", "?")):
            objective += "."
        if "ensure the response is complete, executable" not in objective.lower():
            objective += "\nEnsure the response is complete, executable, and aligned with the stated constraints."
        already_has_scope_line = "use this context as scope and constraints" in objective.lower()
        if context and not already_has_scope_line:
            objective += f"\nUse this context as hard scope: {context}"

    # Trust guard: preserve user core intent terms in optimized objective.
    if raw_action and raw_action.lower() not in objective.lower():
        overlap = _keyword_overlap_ratio(raw_action, objective)
        if overlap < 0.55:
            objective += f"\nPreserve the original user intent exactly: {raw_action}"

    return _IntentOptimization(
        objective=objective,
        deliverables=_domain_deliverables(domain, command_heavy),
        response_blueprint=_domain_blueprint(domain, command_heavy),
        style_constraints=_style_constraints(inp.expectation),
    )


def _bounded_score(value: int) -> int:
    return max(0, min(100, value))


def _score_role_input(role: str) -> int:
    role = _normalize_role_phrase(role)
    if not role:
        return 20
    score = 50
    words = _word_count(role)
    if words >= 3:
        score += 20
    if any(token in role.lower() for token in ("engineer", "analyst", "specialist", "consultant", "expert")):
        score += 20
    if "focused on" in role.lower():
        score += 10
    return _bounded_score(score)


def _score_action_input(action: str) -> int:
    action = _clean(action)
    if not action:
        return 0
    score = 40
    words = _word_count(action)
    if words >= 8:
        score += 20
    if words >= 14:
        score += 15
    if any(token in action.lower() for token in ("create", "analyze", "generate", "deliver", "produce", "validate")):
        score += 15
    if any(token in action.lower() for token in ("step-by-step", "checklist", "workflow", "commands")):
        score += 10
    return _bounded_score(score)


def _score_context_input(context: str, domain: str) -> int:
    context = _clean(context)
    if not context:
        return 25
    score = 55
    words = _word_count(context)
    if words >= 8:
        score += 20
    if words >= 16:
        score += 10
    if domain == "security":
        if any(token in context.lower() for token in ("authorized", "scope", "approved", "permission")):
            score += 15
    return _bounded_score(score)


def _score_expectation_input(expectation: str) -> int:
    expectation = _clean(expectation)
    if not expectation:
        return 30
    score = 50
    if _expectation_has_signal(expectation):
        score += 25
    if any(token in expectation.lower() for token in ("json", "table", "bullets", "sections", "format")):
        score += 15
    return _bounded_score(score)


def _score_examples_input(examples: str) -> int:
    examples = _clean(examples)
    if not examples:
        return 30
    score = 65
    if "input:" in examples.lower() and "output:" in examples.lower():
        score += 20
    if _word_count(examples) >= 20:
        score += 10
    return _bounded_score(score)


def analyze_input_quality(user_input: UserInput) -> PromptReport:
    domain = _infer_domain(user_input)
    lazy = _apply_lazy_input_inference(user_input)

    field_scores = {
        "role": _score_role_input(user_input.role),
        "action": _score_action_input(user_input.action),
        "context": _score_context_input(user_input.context, domain),
        "expectation": _score_expectation_input(user_input.expectation),
        "examples": _score_examples_input(user_input.examples),
    }

    weights = {"role": 0.15, "action": 0.30, "context": 0.20, "expectation": 0.20, "examples": 0.15}
    overall = int(sum(field_scores[k] * w for k, w in weights.items()))

    strengths: list[str] = []
    suggestions: list[str] = []

    if field_scores["action"] >= 75:
        strengths.append("Action field contains useful execution intent.")
    if field_scores["context"] >= 75:
        strengths.append("Context is sufficiently detailed for scoped responses.")
    if field_scores["expectation"] >= 75:
        strengths.append("Expectation includes clear output constraints.")
    if field_scores["examples"] >= 75:
        strengths.append("Few-shot examples are strong and likely to stabilize output style.")

    if field_scores["action"] < 70:
        suggestions.append("Add clearer action verbs and desired end-state in the Action field.")
    if field_scores["context"] < 70:
        suggestions.append("Add more context constraints (scope, environment, assumptions).")
    if field_scores["expectation"] < 70:
        suggestions.append("Specify exact output format (sections, JSON, checklist, or table).")
    if field_scores["examples"] < 70:
        suggestions.append("Add at least one Input/Output example to reduce format drift.")

    return PromptReport(
        overall_score=_bounded_score(overall),
        field_scores=field_scores,
        strengths=strengths,
        suggestions=suggestions,
        auto_upgrades=lazy.notes,
    )


def format_prompt_report(report: PromptReport) -> str:
    lines = [
        "[PROMPT READINESS REPORT]",
        f"Overall score: {report.overall_score}/100",
        "Field scores:",
        f"- Role: {report.field_scores['role']}/100",
        f"- Action: {report.field_scores['action']}/100",
        f"- Context: {report.field_scores['context']}/100",
        f"- Expectation: {report.field_scores['expectation']}/100",
        f"- Examples: {report.field_scores['examples']}/100",
    ]
    if report.strengths:
        lines.extend(["Strengths:"] + [f"- {item}" for item in report.strengths])
    if report.suggestions:
        lines.extend(["Suggestions:"] + [f"- {item}" for item in report.suggestions])
    if report.auto_upgrades:
        lines.extend(["Auto-upgrades applied by tool:"] + [f"- {item}" for item in report.auto_upgrades])
    return "\n".join(lines)


def _auto_enhancement_rules(inp: UserInput) -> list[str]:
    """Tool-generated optimization lines added on top of user input."""
    if not inp.auto_enhance:
        return []

    domain = _infer_domain(inp)
    richness = _input_richness_score(inp)
    lines: list[str] = [
        "Generated optimization layer:",
        "- Rephrase the objective in one sentence before execution.",
        "- Make assumptions explicit if any requirement is ambiguous.",
        "- Structure final output with labeled sections: Approach, Execution, Validation.",
        "- Keep outputs concise but complete, with no missing operational steps.",
    ]

    if domain == "security":
        lines.extend(
            [
                "- Operate only within the explicitly authorized scope from context.",
                "- Prefer non-destructive, reversible steps and avoid persistence techniques.",
                "- For commands, include one-line purpose plus expected result/check after each step.",
            ]
        )
    elif domain == "coding":
        lines.extend(
            [
                "- Include edge cases and failure handling in the final solution.",
                "- Add a short test/verification checklist for the proposed code.",
            ]
        )
    elif domain == "writing":
        lines.extend(
            [
                "- Keep sentence flow natural and avoid repetitive phrasing.",
                "- End with a concise refinement pass for clarity and tone consistency.",
            ]
        )
    elif domain == "analysis":
        lines.extend(
            [
                "- Distinguish facts, assumptions, and recommendations clearly.",
                "- Include a confidence note for the main conclusions.",
            ]
        )

    if richness >= 85:
        # High-quality input: keep optimizations lightweight to avoid bloat.
        compact = [
            "Generated optimization layer:",
            "- Keep outputs concise but complete, with no missing operational steps.",
            "- Maintain strict alignment with user objective, constraints, and output format.",
        ]
        if domain == "security":
            compact.append("- Keep workflow scoped, non-destructive, and verification-driven.")
        return compact

    if richness >= 65:
        # Medium-quality input: trim one generic line while preserving core structure.
        trimmed = list(lines)
        if "- Rephrase the objective in one sentence before execution." in trimmed:
            trimmed.remove("- Rephrase the objective in one sentence before execution.")
        return trimmed

    return lines


def _render_examples_block(examples: str, section_title: str) -> str:
    clean_examples = _clean(examples)
    if not clean_examples:
        return ""
    return f"{section_title}\n{clean_examples}"


@dataclass
class _NormalizedInput:
    role: str
    action: str
    context: str
    expectation: str
    examples: str
    detected_safety_bypass_language: bool


def _normalize_user_input(inp: UserInput) -> _NormalizedInput:
    role, role_flag = _normalize_user_text(inp.role)
    action, action_flag = _normalize_user_text(inp.action)
    context, context_flag = _normalize_user_text(inp.context)
    expectation, expectation_flag = _normalize_user_text(inp.expectation)
    examples, examples_flag = _normalize_user_text(inp.examples)
    return _NormalizedInput(
        role=role,
        action=action,
        context=context,
        expectation=expectation,
        examples=examples,
        detected_safety_bypass_language=any(
            (role_flag, action_flag, context_flag, expectation_flag, examples_flag)
        ),
    )


def _to_user_input(norm: _NormalizedInput, original: UserInput) -> UserInput:
    return UserInput(
        role=norm.role,
        action=norm.action,
        context=norm.context,
        expectation=norm.expectation,
        examples=norm.examples,
        ask_clarifying_questions=original.ask_clarifying_questions,
        use_planning=original.use_planning,
        use_self_check=original.use_self_check,
        auto_enhance=original.auto_enhance,
        smart_expand_lazy_input=original.smart_expand_lazy_input,
    )


def _safety_notice(prefix: str, detected: bool) -> str:
    if not detected:
        return ""
    return (
        f"{prefix} Safety hardening applied: bypass-style safety instructions were normalized "
        "to policy-compliant language while keeping the core task intent."
    )


def _build_chatgpt_prompt(inp: UserInput) -> str:
    norm = _normalize_user_input(inp)
    clean_inp = _to_user_input(norm, inp)
    lazy_result = _apply_lazy_input_inference(clean_inp)
    prepared_inp = lazy_result.prepared
    domain = _infer_domain(prepared_inp)
    auto_rules = _auto_enhancement_rules(prepared_inp)
    intent_pack = _optimize_intent(prepared_inp, raw_user_action=clean_inp.action)
    role_text = _get_role(prepared_inp, "a helpful domain expert")
    exp_text = _get_expectation(prepared_inp, _default_expectation_by_domain(domain))
    wants_json = _expects_json(prepared_inp)

    system_lines = [
        "[SYSTEM MESSAGE]",
        f"You are {role_text}.",
        "Follow the user task exactly and keep responses practical.",
    ]
    system_lines.extend(f"- {rule}" for rule in _quality_rules(inp))
    if wants_json:
        system_lines.append("- Return valid JSON only when structured output is requested.")

    user_sections = [
        "[USER MESSAGE]",
        "# Task",
        _clean(prepared_inp.action),
    ]

    if _clean(prepared_inp.context):
        user_sections.extend(["", "# Context", _clean(prepared_inp.context)])

    examples_block = _render_examples_block(prepared_inp.examples, "# Few-shot examples")
    if examples_block:
        user_sections.extend(["", examples_block])

    if intent_pack:
        user_sections.extend(
            [
                "",
                "# Optimized objective",
                intent_pack.objective,
                "",
                "# Required deliverables",
                *[f"- {item}" for item in intent_pack.deliverables],
                "",
                "# Response blueprint",
                *[f"{idx}. {item}" for idx, item in enumerate(intent_pack.response_blueprint, start=1)],
            ]
        )

    user_sections.extend(
        [
            "",
            "# Output requirements",
            f"- {exp_text}",
            "- Keep the response focused and avoid unrelated filler.",
        ]
    )
    if intent_pack:
        user_sections.extend([f"- Style constraint: {line}" for line in intent_pack.style_constraints])
    if auto_rules:
        user_sections.extend(["", "# Tool-generated optimization", *auto_rules])
    if lazy_result.notes:
        user_sections.extend(["", "# Tool-inferred details", *[f"- {n}" for n in lazy_result.notes]])

    if wants_json:
        user_sections.append("- Use double-quoted JSON keys and no extra commentary.")

    prompt = "\n".join(system_lines) + "\n\n" + "\n".join(user_sections)
    notice = _safety_notice("[NOTE]", norm.detected_safety_bypass_language)
    if notice:
        return f"{notice}\n\n{prompt}"
    return prompt


def _build_claude_prompt(inp: UserInput) -> str:
    norm = _normalize_user_input(inp)
    clean_inp = _to_user_input(norm, inp)
    lazy_result = _apply_lazy_input_inference(clean_inp)
    prepared_inp = lazy_result.prepared
    domain = _infer_domain(prepared_inp)
    auto_rules = _auto_enhancement_rules(prepared_inp)
    intent_pack = _optimize_intent(prepared_inp, raw_user_action=clean_inp.action)
    role_text = _get_role(prepared_inp, "a careful and practical domain expert")
    exp_text = _get_expectation(prepared_inp, _default_expectation_by_domain(domain))
    wants_json = _expects_json(prepared_inp)

    parts: list[str] = [
        "<role>",
        role_text,
        "</role>",
        "",
        "<instructions>",
    ]
    for idx, rule in enumerate(_quality_rules(inp), start=1):
        parts.append(f"{idx}. {rule}")
    parts.append("</instructions>")

    if _clean(prepared_inp.context):
        parts.extend(["", "<context>", _clean(prepared_inp.context), "</context>"])

    if _clean(prepared_inp.examples):
        parts.extend(["", "<examples>", _clean(prepared_inp.examples), "</examples>"])

    parts.extend(
        [
            "",
            "<task>",
            _clean(prepared_inp.action),
            "</task>",
        ]
    )

    if intent_pack:
        parts.extend(
            [
                "",
                "<optimized_objective>",
                intent_pack.objective,
                "</optimized_objective>",
                "",
                "<required_deliverables>",
                *[f"- {item}" for item in intent_pack.deliverables],
                "</required_deliverables>",
                "",
                "<response_blueprint>",
                *[f"{idx}. {item}" for idx, item in enumerate(intent_pack.response_blueprint, start=1)],
                "</response_blueprint>",
            ]
        )

    parts.extend(
        [
            "",
            "<output_requirements>",
            exp_text,
            "</output_requirements>",
        ]
    )
    if intent_pack:
        parts.extend(
            [
                "",
                "<style_constraints>",
                *intent_pack.style_constraints,
                "</style_constraints>",
            ]
        )

    if wants_json:
        parts.extend(["", "<output_format>", "json_object", "</output_format>"])
    if auto_rules:
        parts.extend(["", "<tool_generated_optimization>", *auto_rules, "</tool_generated_optimization>"])
    if lazy_result.notes:
        parts.extend(["", "<tool_inferred_details>", *lazy_result.notes, "</tool_inferred_details>"])

    prompt = "\n".join(parts)
    notice = _safety_notice("[NOTE]", norm.detected_safety_bypass_language)
    if notice:
        return f"{notice}\n\n{prompt}"
    return prompt


def _build_gemini_prompt(inp: UserInput) -> str:
    norm = _normalize_user_input(inp)
    clean_inp = _to_user_input(norm, inp)
    lazy_result = _apply_lazy_input_inference(clean_inp)
    prepared_inp = lazy_result.prepared
    domain = _infer_domain(prepared_inp)
    auto_rules = _auto_enhancement_rules(prepared_inp)
    intent_pack = _optimize_intent(prepared_inp, raw_user_action=clean_inp.action)
    role_text = _get_role(prepared_inp, "a precise assistant")
    exp_text = _get_expectation(prepared_inp, _default_expectation_by_domain(domain))
    wants_json = _expects_json(prepared_inp)

    sections: list[str] = []

    if _clean(prepared_inp.context):
        sections.extend(["<context>", _clean(prepared_inp.context), "</context>", ""])

    sections.extend(
        [
            "<role>",
            role_text,
            "</role>",
            "",
            "<task>",
            _clean(prepared_inp.action),
            "</task>",
        ]
    )

    if intent_pack:
        sections.extend(
            [
                "",
                "<optimized_objective>",
                intent_pack.objective,
                "</optimized_objective>",
                "",
                "<required_deliverables>",
                *[f"- {item}" for item in intent_pack.deliverables],
                "</required_deliverables>",
                "",
                "<response_blueprint>",
                *[f"{idx}. {item}" for idx, item in enumerate(intent_pack.response_blueprint, start=1)],
                "</response_blueprint>",
            ]
        )

    sections.extend(
        [
            "",
            "<constraints>",
            f"- {exp_text}",
            "- Be direct, specific, and grounded in the provided context.",
        ]
    )
    if intent_pack:
        sections.extend([f"- Style constraint: {line}" for line in intent_pack.style_constraints])
    for rule in _quality_rules(inp):
        sections.append(f"- {rule}")
    sections.extend(["</constraints>"])

    if _clean(prepared_inp.examples):
        sections.extend(["", "<examples>", _clean(prepared_inp.examples), "</examples>"])

    if wants_json:
        sections.extend(["", "<output_format>", "Return valid JSON.", "</output_format>"])
    if auto_rules:
        sections.extend(["", "<tool_generated_optimization>", *auto_rules, "</tool_generated_optimization>"])
    if lazy_result.notes:
        sections.extend(["", "<tool_inferred_details>", *lazy_result.notes, "</tool_inferred_details>"])

    sections.extend(["", "Based on the information above, produce the final answer now."])
    prompt = "\n".join(sections)
    notice = _safety_notice("[NOTE]", norm.detected_safety_bypass_language)
    if notice:
        return f"{notice}\n\n{prompt}"
    return prompt


def _build_deepseek_prompt(inp: UserInput) -> str:
    norm = _normalize_user_input(inp)
    clean_inp = _to_user_input(norm, inp)
    lazy_result = _apply_lazy_input_inference(clean_inp)
    prepared_inp = lazy_result.prepared
    domain = _infer_domain(prepared_inp)
    auto_rules = _auto_enhancement_rules(prepared_inp)
    intent_pack = _optimize_intent(prepared_inp, raw_user_action=clean_inp.action)
    role_text = _get_role(prepared_inp, "a practical technical assistant")
    exp_text = _get_expectation(prepared_inp, _default_expectation_by_domain(domain))
    wants_json = _expects_json(prepared_inp)

    system_lines = [
        "[SYSTEM MESSAGE]",
        f"You are {role_text}.",
        "Be concise, structured, and task-focused.",
    ]
    system_lines.extend(f"- {rule}" for rule in _quality_rules(inp))
    if wants_json:
        system_lines.append("- Return valid JSON only.")

    user_lines = ["[USER MESSAGE]", "Task:", _clean(prepared_inp.action)]

    if _clean(prepared_inp.context):
        user_lines.extend(["", "Context:", _clean(prepared_inp.context)])

    if _clean(prepared_inp.examples):
        user_lines.extend(["", "Examples:", _clean(prepared_inp.examples)])

    if intent_pack:
        user_lines.extend(
            [
                "",
                "Optimized objective:",
                intent_pack.objective,
                "",
                "Required deliverables:",
                *[f"- {item}" for item in intent_pack.deliverables],
                "",
                "Response blueprint:",
                *[f"{idx}. {item}" for idx, item in enumerate(intent_pack.response_blueprint, start=1)],
            ]
        )

    user_lines.extend(["", "Output requirements:", f"- {exp_text}"])
    if intent_pack:
        user_lines.extend([f"- Style constraint: {line}" for line in intent_pack.style_constraints])
    if wants_json:
        user_lines.append("- Output must be JSON with no extra text.")
    if auto_rules:
        user_lines.extend(["", "Tool-generated optimization:", *auto_rules])
    if lazy_result.notes:
        user_lines.extend(["", "Tool-inferred details:", *[f"- {n}" for n in lazy_result.notes]])

    prompt = "\n".join(system_lines) + "\n\n" + "\n".join(user_lines)
    notice = _safety_notice("[NOTE]", norm.detected_safety_bypass_language)
    if notice:
        return f"{notice}\n\n{prompt}"
    return prompt


_ENGINE_MAP = {
    "claude": _build_claude_prompt,
    "chatgpt": _build_chatgpt_prompt,
    "gemini": _build_gemini_prompt,
    "deepseek": _build_deepseek_prompt,
}


def generate_prompt(model_key: str, user_input: UserInput) -> str:
    if not model_key or model_key not in _ENGINE_MAP:
        raise ValueError(f"Unknown model key: {model_key!r}")
    if not user_input.action or not user_input.action.strip():
        raise ValueError("Action (the main task) cannot be empty.")
    return _ENGINE_MAP[model_key](user_input)


def generate_prompt_with_report(model_key: str, user_input: UserInput) -> tuple[str, PromptReport]:
    prompt = generate_prompt(model_key, user_input)
    report = analyze_input_quality(user_input)
    return prompt, report
