"""
app.py
WildfireAdvisory – Gradio application entry point.
Deployed on Hugging Face Spaces.

Run locally:
    python app.py

Environment variables (set in .env locally, or as HF Space secrets):
    OPENAI_API_KEY   – required for AI explanations
"""

import os
from dotenv import load_dotenv
load_dotenv()

import gradio as gr
from src.predict import predict
from src.nlp import explain_detailed, explain_brief, chat_followup

# ── Preset scenarios ──────────────────────────────────────
PRESETS = {
    "🔴 Extreme risk (peak drought)": {
        "Temperature": 39, "RH": 22, "Ws": 22, "Rain": 0.0,
        "FFMC": 92.0, "DMC": 62.0, "DC": 210.0,
        "ISI": 16.0,  "BUI": 66.0, "FWI": 29.0,
    },
    "🟠 Moderate risk (warm dry day)": {
        "Temperature": 33, "RH": 50, "Ws": 15, "Rain": 0.0,
        "FFMC": 78.0, "DMC": 28.0, "DC": 90.0,
        "ISI": 6.0,   "BUI": 32.0, "FWI": 11.0,
    },
    "🟢 Low risk (after rainfall)": {
        "Temperature": 24, "RH": 82, "Ws": 9,  "Rain": 3.5,
        "FFMC": 45.0, "DMC":  7.0, "DC": 18.0,
        "ISI": 0.8,   "BUI":  7.0, "FWI":  1.2,
    },
}

# ── Risk gauge HTML ───────────────────────────────────────

def _risk_gauge_html(probability: float, label: str) -> str:
    pct        = probability * 100
    bar_color  = (
        "#d32f2f" if pct >= 70 else
        "#f57c00" if pct >= 40 else
        "#388e3c"
    )
    icon  = "🔴" if pct >= 70 else "🟠" if pct >= 40 else "🟢"
    title = "FIRE RISK" if label == "Fire Risk" else "NO FIRE RISK"

    return f"""
<div style="font-family:sans-serif; padding:16px; border-radius:12px;
            background:#1e1e2e; color:#e0e0e0; margin-bottom:8px;">
  <div style="font-size:1.6rem; font-weight:700; margin-bottom:6px;">
    {icon} {title}
  </div>
  <div style="font-size:0.95rem; margin-bottom:10px; color:#aaa;">
    Fire probability: <strong style="color:#fff;">{pct:.1f}%</strong>
  </div>
  <div style="background:#333; border-radius:8px; height:18px; overflow:hidden;">
    <div style="width:{pct:.1f}%; background:{bar_color};
                height:100%; border-radius:8px;
                transition:width 0.4s ease;"></div>
  </div>
</div>
"""


def _top_features_html(top_features: list) -> str:
    rows = ""
    max_val = max(v for _, v in top_features) or 1
    colors  = ["#ef5350", "#ff7043", "#ffa726"]
    for i, (feat, val) in enumerate(top_features):
        bar_w = int((val / max_val) * 100)
        rows += (
            f'<div style="margin-bottom:6px;">'
            f'  <span style="display:inline-block;width:52px;'
            f'font-size:0.8rem;color:#aaa;">{feat}</span>'
            f'  <div style="display:inline-block;background:#333;'
            f'border-radius:4px;height:14px;width:160px;vertical-align:middle;">'
            f'    <div style="width:{bar_w}%;background:{colors[i]};'
            f'height:100%;border-radius:4px;"></div>'
            f'  </div>'
            f'  <span style="font-size:0.8rem;color:#ccc;margin-left:6px;">'
            f'{val:.3f}</span>'
            f'</div>'
        )
    return (
        f'<div style="font-family:sans-serif;background:#1e1e2e;'
        f'border-radius:10px;padding:12px;margin-top:4px;">'
        f'<div style="font-size:0.85rem;color:#aaa;margin-bottom:8px;">'
        f'Top driving features</div>'
        f'{rows}</div>'
    )


# ── Main prediction handler ───────────────────────────────

def run_prediction(
    temperature, rh, ws, rain,
    ffmc, dmc, dc, isi, bui, fwi,
    model_choice,
):
    input_values = {
        "Temperature": temperature, "RH": rh, "Ws": ws, "Rain": rain,
        "FFMC": ffmc, "DMC": dmc, "DC": dc,
        "ISI": isi,   "BUI": bui, "FWI": fwi,
    }

    try:
        result      = predict(input_values, model=model_choice)
        explanation = explain_detailed(input_values, result)
    except EnvironmentError as e:
        # No API key — still show ML result, warn about NLP
        result      = predict(input_values, model=model_choice)
        explanation = (
            "⚠️  AI explanation unavailable: OPENAI_API_KEY not set.\n\n"
            f"ML prediction: {result['label']} "
            f"({result['probability']*100:.1f}% fire probability).\n\n"
            "Set your API key in the Space secrets (or .env locally) "
            "to enable full explanations."
        )
    except Exception as e:
        result = predict(input_values, model=model_choice)
        explanation = f"⚠️  Could not generate explanation: {e}"

    gauge      = _risk_gauge_html(result["probability"], result["label"])
    feat_bars  = _top_features_html(result["top_features"])

    return gauge, feat_bars, explanation, input_values, result


# ── Preset loader ─────────────────────────────────────────

def load_preset(preset_name):
    p = PRESETS.get(preset_name, {})
    if not p:
        return [gr.update()] * 10
    return [
        p["Temperature"], p["RH"],  p["Ws"],  p["Rain"],
        p["FFMC"],        p["DMC"], p["DC"],
        p["ISI"],         p["BUI"], p["FWI"],
    ]


# ── Chat handler ──────────────────────────────────────────

def respond(user_message, history, input_values_state, prediction_state):
    if not user_message.strip():
        return history, ""

    if not input_values_state:
        history = history + [{"role": "assistant",
                               "content": "⚠️ Please run a prediction first, "
                                          "then ask your question."}]
        return history, ""

    # Convert Gradio message format to OpenAI format
    history_openai = [
        {"role": m["role"], "content": m["content"]}
        for m in history
        if m["role"] in ("user", "assistant")
    ]

    try:
        reply = chat_followup(
            input_values_state,
            prediction_state,
            history_openai,
            user_message,
        )
    except EnvironmentError:
        reply = "⚠️ AI chat unavailable: OPENAI_API_KEY not set."
    except Exception as e:
        reply = f"⚠️ Error: {e}"

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


# ── FWI reference accordion content ──────────────────────

FWI_REFERENCE = """
**The Canadian Fire Weather Index (FWI) System** translates daily weather
readings into standardised fire danger ratings used worldwide.

| Index | Measures | Danger range |
|---|---|---|
| **FFMC** | Fine fuel (grass/litter) moisture | >75 = elevated, >90 = extreme |
| **DMC** | Moderate-depth organic layer moisture | >30 = elevated |
| **DC** | Deep organic layer / drought severity | >200 = severe drought |
| **ISI** | Fire spread rate (wind + FFMC) | >10 = high, >15 = extreme |
| **BUI** | Total fuel available (DMC + DC) | >40 = high |
| **FWI** | Overall fire danger (ISI + BUI) | >20 = very high, >25 = extreme |

*Data source: Algerian Forest Fires dataset (UCI / Kaggle), summer 2012.*
"""


# ── UI layout ─────────────────────────────────────────────

CSS = """
.risk-panel { min-height: 120px; }
.chat-panel { border-top: 1px solid #333; margin-top: 12px; }
footer { display: none !important; }
"""

with gr.Blocks(title="🔥 WildfireAdvisor") as demo:

    # ── Shared state ──────────────────────────────────────
    input_state      = gr.State({})
    prediction_state = gr.State({})

    # ── Header ────────────────────────────────────────────
    gr.Markdown(
        "# 🔥 WildfireAdvisor\n"
        "Enter weather and Fire Weather Index (FWI) readings to assess "
        "wildfire risk. The ML model (Random Forest) predicts fire probability; "
        "the AI advisor explains the key drivers and recommends safety actions."
    )

    # ── Preset row ────────────────────────────────────────
    with gr.Row():
        preset_dd = gr.Dropdown(
            choices=list(PRESETS.keys()),
            label="Quick-load a preset scenario",
            value=None,
        )

    # ── Main two-column layout ────────────────────────────
    with gr.Row():

        # Left: inputs
        with gr.Column(scale=1, min_width=280):
            gr.Markdown("### 🌡️ Weather Inputs")
            temperature = gr.Slider(15, 45,  value=33,   step=1,   label="Temperature (°C)")
            rh          = gr.Slider(10, 100, value=50,   step=1,   label="Relative Humidity (%)")
            ws          = gr.Slider(0,  50,  value=15,   step=1,   label="Wind Speed (km/h)")
            rain        = gr.Slider(0,  20,  value=0.0,  step=0.1, label="Rainfall (mm)")

            gr.Markdown("### 📊 FWI System Indices")
            ffmc = gr.Slider(28,  93,  value=78.0,  step=0.1, label="FFMC")
            dmc  = gr.Slider(1,   68,  value=28.0,  step=0.1, label="DMC")
            dc   = gr.Slider(7,   220, value=90.0,  step=0.5, label="DC")
            isi  = gr.Slider(0,   18,  value=6.0,   step=0.1, label="ISI")
            bui  = gr.Slider(1,   68,  value=32.0,  step=0.1, label="BUI")
            fwi  = gr.Slider(0,   31,  value=11.0,  step=0.1, label="FWI")

            gr.Markdown("### ⚙️ Model")
            model_choice = gr.Radio(
                ["Random Forest", "Logistic Regression"],
                value="Random Forest",
                label="Select ML model",
            )
            predict_btn = gr.Button("🔍 Assess Risk", variant="primary", size="lg")

        # Right: outputs
        with gr.Column(scale=2, min_width=400):
            gr.Markdown("### 🎯 Risk Assessment")
            gauge_html    = gr.HTML(elem_classes=["risk-panel"])
            feat_html     = gr.HTML()

            gr.Markdown("### 🤖 AI Explanation & Safety Advice")
            explanation_box = gr.Textbox(
                label="",
                lines=12,
                interactive=False,
                placeholder="Click 'Assess Risk' to generate an explanation …",
                buttons=["copy"],
            )

    # ── FWI reference ─────────────────────────────────────
    with gr.Accordion("📖 FWI Index Reference", open=False):
        gr.Markdown(FWI_REFERENCE)

    # ── Chat section ──────────────────────────────────────
    gr.Markdown("---\n### 💬 Ask the AI Advisor a Follow-up Question")
    gr.Markdown(
        "*After running a prediction, ask anything: 'What does a high ISI mean?', "
        "'Should I evacuate?', 'How does temperature affect fire spread?'*"
    )
    chatbot = gr.Chatbot(
        label="",
        height=340,
        buttons=["copy_all"],
        placeholder="Run a prediction above, then ask your question here …",
    )
    with gr.Row():
        chat_in  = gr.Textbox(
            placeholder="e.g. What does FWI 29 mean in practice?",
            label="",
            scale=5,
            container=False,
        )
        chat_btn = gr.Button("Send", variant="primary", scale=1)

    # ── Footer ────────────────────────────────────────────
    gr.Markdown(
        "---\n"
        "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
        "WildfireAdvisor · ML: Random Forest trained on Algerian Forest Fires dataset (UCI) · "
        "NLP: GPT-4o-mini · Built for educational purposes only"
        "</div>"
    )

    # ── Event wiring ──────────────────────────────────────
    slider_inputs = [temperature, rh, ws, rain, ffmc, dmc, dc, isi, bui, fwi]

    # Preset → populate sliders
    preset_dd.change(
        fn=load_preset,
        inputs=[preset_dd],
        outputs=slider_inputs,
    )

    # Predict button
    predict_btn.click(
        fn=run_prediction,
        inputs=slider_inputs + [model_choice],
        outputs=[gauge_html, feat_html, explanation_box, input_state, prediction_state],
    )

    # Chat — button click
    chat_btn.click(
        fn=respond,
        inputs=[chat_in, chatbot, input_state, prediction_state],
        outputs=[chatbot, chat_in],
    )

    # Chat — Enter key
    chat_in.submit(
        fn=respond,
        inputs=[chat_in, chatbot, input_state, prediction_state],
        outputs=[chatbot, chat_in],
    )


if __name__ == "__main__":
    demo.launch(
        show_error=True,
        theme=gr.themes.Soft(primary_hue="orange", neutral_hue="slate"),
    )
