"""
evaluate_prompts.py
Compare all three NLP prompt strategies (A: Brief, B: Detailed, C: Structured)
across three representative wildfire risk scenarios.

Outputs:
  - Console: formatted comparison table
  - notebooks/prompt_evaluation_results.md   (human-readable report)

Run with:
    python notebooks/evaluate_prompts.py
Requires OPENAI_API_KEY set in .env or environment.
"""

import sys
import os
import time
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dotenv import load_dotenv
load_dotenv()

from src.predict import predict
from src.nlp import explain_brief, explain_detailed, explain_structured

# ── Test scenarios ────────────────────────────────────────

SCENARIOS = {
    "High Risk": {
        "Temperature": 39, "RH": 22, "Ws": 22, "Rain": 0,
        "FFMC": 92.0, "DMC": 62.0, "DC": 210.0,
        "ISI": 16.0, "BUI": 66.0, "FWI": 29.0,
        "description": "Peak summer drought — all indices elevated",
    },
    "Medium Risk": {
        "Temperature": 33, "RH": 50, "Ws": 15, "Rain": 0,
        "FFMC": 78.0, "DMC": 28.0, "DC": 90.0,
        "ISI": 6.0, "BUI": 32.0, "FWI": 11.0,
        "description": "Warm dry day — moderate indices",
    },
    "Low Risk": {
        "Temperature": 24, "RH": 82, "Ws": 9, "Rain": 3.5,
        "FFMC": 45.0, "DMC": 7.0, "DC": 18.0,
        "ISI": 0.8, "BUI": 7.0, "FWI": 1.2,
        "description": "After rainfall — humid and cool",
    },
}

STRATEGIES = {
    "A: Brief":      explain_brief,
    "B: Detailed":   explain_detailed,
    "C: Structured": explain_structured,
}

# ── Evaluation criteria ───────────────────────────────────
# Scored manually (0-3) on four axes:
#   Clarity     — easy to understand for general public
#   Completeness— covers risk reason, indices, actions
#   Actionability — advice is specific and prioritised
#   UI fit      — suitable for display in the Gradio app

MANUAL_SCORES = {
    # Format: (Clarity, Completeness, Actionability, UI_fit)
    "A: Brief":      (3, 1, 2, 2),
    "B: Detailed":   (2, 3, 3, 3),
    "C: Structured": (3, 2, 3, 3),
}


def word_count(text: str) -> int:
    return len(text.split())


def run_evaluation():
    results = {}   # {scenario: {strategy: response_text}}

    print("=" * 65)
    print("  WildfireAdvisor — Prompt Strategy Evaluation")
    print("=" * 65)

    for scenario_name, inputs in SCENARIOS.items():
        desc = inputs.pop("description")
        print(f"\n{'─'*65}")
        print(f"  Scenario: {scenario_name}  |  {desc}")
        print(f"{'─'*65}")

        pred = predict(inputs, model="Random Forest")
        print(f"  ML prediction: {pred['label']}  "
              f"({pred['probability']*100:.1f}% probability)")

        results[scenario_name] = {}
        for strat_name, fn in STRATEGIES.items():
            print(f"\n  [{strat_name}] generating …", end=" ", flush=True)
            t0 = time.time()
            try:
                response = fn(inputs, pred)
                elapsed = time.time() - t0
                wc = word_count(response)
                print(f"✔  {wc} words  ({elapsed:.1f}s)")
            except Exception as e:
                response = f"[ERROR: {e}]"
                print(f"✘  {e}")
            results[scenario_name][strat_name] = response

        inputs["description"] = desc  # restore

    return results


def build_report(results: dict) -> str:
    lines = []
    lines.append("# Prompt Strategy Evaluation Report\n")
    lines.append(
        "Compares three prompt strategies across three wildfire risk scenarios.\n"
        "The ML model (Random Forest) provides the prediction; the NLP layer explains it.\n"
    )

    # ── Scoring table ─────────────────────────────────────
    lines.append("## Evaluation Criteria Scores (Manual, 0–3)\n")
    lines.append(
        "| Strategy | Clarity | Completeness | Actionability | UI Fit | **Total** |\n"
        "|---|---|---|---|---|---|\n"
    )
    for strat, (cl, co, ac, ui) in MANUAL_SCORES.items():
        total = cl + co + ac + ui
        lines.append(
            f"| {strat} | {cl} | {co} | {ac} | {ui} | **{total}** |\n"
        )
    lines.append(
        "\n**Winner: B (Detailed)** — highest completeness and actionability, "
        "selected as the default strategy in the app.\n"
        "**Runner-up: C (Structured)** — best UI fit; used as an alternative rendering mode.\n"
    )

    # ── Per-scenario outputs ──────────────────────────────
    lines.append("\n## Full Outputs by Scenario\n")
    for scenario_name, strat_outputs in results.items():
        inputs = SCENARIOS[scenario_name].copy()
        inputs.pop("description", None)
        pred = predict(inputs, model="Random Forest")

        lines.append(f"### {scenario_name}\n")
        lines.append(
            f"**Inputs:** Temp={inputs['Temperature']}°C, RH={inputs['RH']}%, "
            f"FWI={inputs['FWI']}, FFMC={inputs['FFMC']}  \n"
            f"**Prediction:** {pred['label']} ({pred['probability']*100:.1f}%)\n"
        )

        for strat_name, text in strat_outputs.items():
            wc = word_count(text)
            lines.append(f"\n#### Strategy {strat_name}  *(~{wc} words)*\n")
            lines.append(f"{text}\n")

    # ── Analysis ──────────────────────────────────────────
    lines.append("\n## Analysis & Conclusion\n")
    lines.append(
        "**Strategy A (Brief)** produces concise outputs but omits explanation "
        "of *why* the risk is elevated and does not explain the FWI indices. "
        "Suitable as a quick-glance summary but insufficient as a standalone response.\n\n"
        "**Strategy B (Detailed)** is the most complete. It explicitly links the "
        "ML model's top features to the risk explanation, provides a plain-language "
        "glossary of FWI indices, and gives a prioritised numbered action plan. "
        "Slightly longer but well within a comfortable reading length. "
        "**Selected as the default strategy.**\n\n"
        "**Strategy C (Structured)** uses markdown headings making it easy to "
        "parse programmatically or render in separate UI cards. However, the "
        "constrained format sometimes produces generic advice. Used as an "
        "alternative rendering mode for the structured output panel.\n\n"
        "**Integration with ML block:** Every strategy receives the ML prediction "
        "label, probability, and top-3 feature importances as explicit context. "
        "The LLM is instructed to reference these values directly, ensuring the "
        "NLP output is grounded in — and explains — the numeric model's decision.\n"
    )

    return "".join(lines)


def save_report(report: str) -> Path:
    out = Path(__file__).parent / "prompt_evaluation_results.md"
    out.write_text(report, encoding="utf-8")
    print(f"\n✔ Report saved → {out}")
    return out


if __name__ == "__main__":
    results = run_evaluation()
    report  = build_report(results)
    save_report(report)

    print("\n" + "=" * 65)
    print("  Evaluation complete.")
    print("  See notebooks/prompt_evaluation_results.md for full report.")
    print("=" * 65)
