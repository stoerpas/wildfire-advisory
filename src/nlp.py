"""
nlp.py
NLP module: OpenAI-powered explanation and advisory chat.

Three prompt strategies are implemented and compared:
  Strategy A – "Brief":      2-sentence summary + 3 bullet safety tips (~80 words).
  Strategy B – "Detailed":   Full factor explanation + plain-language index
                              glossary + prioritised action plan (~220 words).
  Strategy C – "Structured": JSON-like sections with explicit headings so the
                              app can render risk level, explanation, and actions
                              in separate UI components (~200 words).

The Gradio app uses Strategy B by default.
Strategy comparison is documented in notebooks/evaluate_prompts.py.
"""

import os
from openai import OpenAI
from src.config import OPENAI_MODEL, MAX_TOKENS

# ── Client (lazy singleton) ───────────────────────────────
_client = None

def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. "
                "Copy .env.example → .env and add your key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


# ── Shared context builder ────────────────────────────────

def _build_context(input_values: dict, prediction: dict) -> str:
    """
    Assembles a structured context block that is injected into every
    prompt so the LLM always grounds its response in the actual ML output.
    """
    top = prediction["top_features"]
    top_str = ", ".join(f"{feat} ({val:.3f})" for feat, val in top)

    risk_pct = prediction["probability"] * 100
    return (
        f"=== ML Prediction ===\n"
        f"Result : {prediction['label']}\n"
        f"Fire probability : {risk_pct:.1f}%\n"
        f"Top 3 driving features : {top_str}\n\n"
        f"=== Sensor Readings ===\n"
        f"Temperature : {input_values.get('Temperature')} °C\n"
        f"Relative Humidity : {input_values.get('RH')} %\n"
        f"Wind Speed : {input_values.get('Ws')} km/h\n"
        f"Rainfall : {input_values.get('Rain')} mm\n\n"
        f"=== FWI System Indices ===\n"
        f"FFMC : {input_values.get('FFMC')}   "
        f"DMC : {input_values.get('DMC')}   "
        f"DC : {input_values.get('DC')}\n"
        f"ISI : {input_values.get('ISI')}   "
        f"BUI : {input_values.get('BUI')}   "
        f"FWI : {input_values.get('FWI')}"
    )


# ── System prompt ─────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are WildfireAdvisor, an expert wildfire risk analyst and safety "
    "communicator. You receive structured weather sensor readings and "
    "Fire Weather Index (FWI) system values alongside a machine-learning "
    "risk prediction.\n\n"
    "Your role:\n"
    "- Explain the risk level in plain, accessible language.\n"
    "- Reference the specific features that drove the prediction.\n"
    "- Give practical, prioritised safety advice.\n"
    "- Be factual and calm — avoid unnecessary alarm or false reassurance.\n"
    "- Never invent numbers; only reference values provided to you.\n"
    "- Keep the general public as your target audience."
)


# ── Strategy A: Brief ─────────────────────────────────────

def explain_brief(input_values: dict, prediction: dict) -> str:
    """
    Strategy A – Brief (~80 words).
    Best for: quick dashboard summaries, mobile views.
    Comparison criterion: information density vs word count.
    """
    context = _build_context(input_values, prediction)
    user_msg = (
        f"{context}\n\n"
        "In exactly 2 sentences, summarise the current wildfire risk and "
        "the main reason for it. Then list exactly 3 bullet-point safety "
        "recommendations. Total response must be under 100 words."
    )
    resp = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=200,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Strategy B: Detailed ──────────────────────────────────

def explain_detailed(input_values: dict, prediction: dict) -> str:
    """
    Strategy B – Detailed (~220 words).  DEFAULT strategy used in the app.
    Best for: main explanation panel; users who want to understand why.
    Comparison criterion: completeness, usefulness of advice.
    """
    context = _build_context(input_values, prediction)
    user_msg = (
        f"{context}\n\n"
        "Please provide a thorough explanation structured as follows:\n\n"
        "1. RISK ASSESSMENT — Why is the risk at this level? "
        "Reference the top driving features by name.\n\n"
        "2. WHAT THE INDICES MEAN — Briefly explain the key FWI indices "
        "in plain language (what they measure, why they matter).\n\n"
        "3. ACTION PLAN — A numbered list of prioritised safety steps "
        "appropriate for this risk level.\n\n"
        "4. LIMITATIONS — One sentence on what the model cannot account for.\n\n"
        "Write for a non-technical audience. Be specific, not generic."
    )
    resp = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Strategy C: Structured sections ──────────────────────

def explain_structured(input_values: dict, prediction: dict) -> str:
    """
    Strategy C – Structured with explicit markdown headings (~200 words).
    Best for: rendering in UI components with separate sections.
    Comparison criterion: parseability, UI-friendliness.
    """
    context = _build_context(input_values, prediction)
    user_msg = (
        f"{context}\n\n"
        "Respond using exactly these four markdown headings:\n\n"
        "## Risk Level\n"
        "One sentence stating the risk and confidence.\n\n"
        "## Why This Risk?\n"
        "2-3 sentences referencing the specific top features.\n\n"
        "## Recommended Actions\n"
        "A numbered list of 4 concrete safety actions.\n\n"
        "## Important Caveats\n"
        "One sentence about model limitations.\n\n"
        "Do not add any other sections or text outside these headings."
    )
    resp = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
    )
    return resp.choices[0].message.content.strip()


# ── Multi-turn chat ───────────────────────────────────────

def chat_followup(
    input_values: dict,
    prediction: dict,
    conversation_history: list[dict],
    user_message: str,
) -> str:
    """
    Multi-turn follow-up chat grounded in the current session's prediction.

    Parameters
    ----------
    conversation_history : list of {"role": "user"|"assistant", "content": str}
        Full prior turns so the model has memory of the conversation.
    user_message : str
        The new user question.
    """
    context = _build_context(input_values, prediction)
    system_with_context = (
        f"{SYSTEM_PROMPT}\n\n"
        f"--- Current session data (do not contradict these values) ---\n"
        f"{context}"
    )
    messages = (
        [{"role": "system", "content": system_with_context}]
        + conversation_history
        + [{"role": "user", "content": user_message}]
    )
    resp = _get_client().chat.completions.create(
        model=OPENAI_MODEL,
        max_tokens=MAX_TOKENS,
        messages=messages,
    )
    return resp.choices[0].message.content.strip()
