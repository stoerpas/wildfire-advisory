<<<<<<< HEAD
---
title: Wildfire Advisory
emoji: 📊
colorFrom: indigo
colorTo: indigo
sdk: gradio
sdk_version: 6.15.1
python_version: '3.13'
app_file: app.py
pinned: false
short_description: ai application project
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# 📊 WildfireAdvisor

> **Predict wildfire risk from weather conditions and get AI-powered safety advice.**

A combined **ML + NLP** application that uses the Canadian Fire Weather Index (FWI)
system alongside a trained Random Forest classifier to assess wildfire risk, then
uses GPT-4o-mini to explain the prediction in plain language and guide users
through appropriate safety actions.

---

## How It Works

```
User inputs weather + FWI readings
        ↓
[ML Block]  Random Forest (or Logistic Regression)
        → fire risk prediction + probability + top 3 feature importances
        ↓
[NLP Block] GPT-4o-mini receives ML output as structured context
        → plain-language explanation of WHY + safety action plan
        → multi-turn follow-up chat
```

The NLP block is **grounded in the ML output** — it always receives the
prediction label, probability, and top driving features, so explanations
are specific to the current reading, not generic fire advice.

---

## Project Structure

```
wildfire-advisor/
├── app.py                        # Gradio app (HF Spaces entry point)
├── requirements.txt
├── .env.example                  # Copy to .env and add your API key
├── data/
│   ├── raw/                      # Original downloaded CSVs
│   └── processed/                # Cleaned, model-ready CSVs
├── notebooks/
│   ├── eda.py                    # Exploratory Data Analysis script
│   ├── evaluate_prompts.py       # NLP prompt strategy comparison
│   ├── prompt_evaluation_results.md
│   └── figures/                  # EDA + training plots (11 figures)
├── src/
│   ├── config.py                 # Paths, feature lists, hyperparameters
│   ├── preprocess.py             # Data loading & cleaning
│   ├── train.py                  # Model training & comparison (run once)
│   ├── predict.py                # Inference only (used by app)
│   └── nlp.py                    # OpenAI prompt strategies A/B/C + chat
├── models/                       # Saved .pkl artefacts (from train.py)
└── docs/
    └── documentation.md          # Project documentation template
```

---

## Datasets

| Dataset | Source | Role |
|---|---|---|
| **Algerian Forest Fires** (UCI) | [Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset) | ML training & evaluation |
| **1.88M US Wildfires** | [Kaggle](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires) | EDA / contextual analysis |

Neither dataset was used during the semester.

---

## ML Results

| Model | Accuracy | ROC-AUC | F1 (fire) | CV AUC (5-fold) |
|---|---|---|---|---|
| **Random Forest** | **97.96%** | **1.000** | **0.983** | 0.997 ± 0.004 |
| Logistic Regression | 93.88% | 0.993 | 0.947 | 0.993 ± 0.011 |

---

## Quickstart (local)

```bash
# 1. Clone
git clone https://github.com/stoerpas/wildfire-advisory.git
cd wildfire-advisor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI key
cp .env.example .env
# Edit .env → OPENAI_API_KEY=sk-...

# 4. Place datasets in data/raw/ (download from Kaggle links above)

# 5. Train models (only needed once; skip if models/ already contains .pkl files)
python -m src.train

# 6. Launch app
python app.py
```

On **Hugging Face Spaces**, set `OPENAI_API_KEY` as a Space secret
(Settings → Variables and secrets → New secret).

---

## Authors

- stoerpas 
=======
# wildfire-advisory
ai application project
>>>>>>> 9abeb2f5ae073567fc67c8cc6637b473c7ebe2b4
