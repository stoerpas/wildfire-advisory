
---

## Project Metadata

- **Project title:** WildfireAdvisor — ML-powered Wildfire Risk Prediction with AI Safety Explanations
- **Student:** stoerpas
- **GitHub repository URL:** https://github.com/stoerpas/wildfire-advisor
- **Deployment URL:** https://huggingface.co/spaces/stoerpas/wildfire-advisorywildfire-advisor
- **Submission date:** June 7, 2026

### Mandatory Setup Checks

- [x] At least 2 blocks selected
- [x] Multiple and different data sources used
- [x] Deployment URL provided
- [x] Required GitHub users added to repository (`jasminh`, `bkuehnis`)

---

## Selected AI Blocks

- [x] ML Numeric Data
- [x] NLP
- [ ] Computer Vision

**Primary block 1:** ML Numeric Data  
**Primary block 2:** NLP

---

## 1. Project Foundation (Short)

### 1.1 Problem Definition

- **Problem statement:** Wildfire risk is often communicated through raw meteorological indices (FWI system) that are opaque to the general public. Emergency managers and residents cannot easily interpret what a "FWI of 28" means in practice, or what they should do about it.
- **Goal:** Build an end-to-end application that (1) predicts binary wildfire risk from structured weather and FWI readings using a trained ML classifier, and (2) uses an LLM to translate the prediction into a plain-language explanation with a prioritised safety action plan.
- **Success criteria:** ML model accuracy ≥ 90% and ROC-AUC ≥ 0.95 on held-out test data; NLP output correctly references the top driving features from the ML model and produces actionable, scenario-specific safety advice across all three risk levels tested.

### 1.2 Integration Logic

- **How the selected blocks interact:** The ML block produces a structured prediction object (label, fire probability, top-3 feature importances). This object is injected verbatim as context into the NLP prompt. The LLM is explicitly instructed to reference these values — it explains *why* the ML model reached its conclusion, not fire risk in general.
- **Data and output flow between blocks:**

```
Weather + FWI inputs (10 numeric features)
        │
        ▼
[ML Block — src/predict.py]
  Random Forest classifier
        │
        ▼
  { label, probability, top_3_features }
        │
        ▼
[NLP Block — src/nlp.py]
  GPT-4o-mini with structured context
        │
        ▼
  Plain-language explanation + safety action plan
        │
        ▼
[Gradio UI — app.py]
  Risk gauge + feature bars + explanation panel + chat
```

See [`src/predict.py`](../src/predict.py) for the ML inference interface and [`src/nlp.py`](../src/nlp.py) for the prompt injection logic.

---

## 2. Block Documentation

### 2A. ML Numeric Data

#### 2A.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | [Algerian Forest Fires Dataset (UCI via Kaggle)](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset) | Structured CSV (tabular) | 243 rows × 14 columns (10 features + target + date + region) | Primary training and evaluation dataset |
| 2 | [1.88 Million US Wildfires (Kaggle)](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires) | SQLite database (tabular) | 1.88M records across 24 years | Contextual EDA — distribution of fire sizes, seasonal patterns, geographic spread; informed feature understanding |

Neither dataset was used during the semester.

#### 2A.2 Preprocessing and Features

- **Cleaning steps:** The raw Algerian CSV has a two-region structure with region-label rows, duplicate header rows, blank separator lines, and Windows-style line endings. These are handled in [`src/preprocess.py`, lines 41–84](../src/preprocess.py#L41-L84) by reading line-by-line, detecting region headers, skipping malformed rows, and tagging each observation with its source region. After cleaning: 243 rows, 0 missing values.
- **Preprocessing steps:** All string values stripped of whitespace; target column (`Classes`) mapped from `"fire"/"not fire"` to binary 1/0; all 10 feature columns cast to float64. `StandardScaler` applied before Logistic Regression (not needed for Random Forest). See [`src/train.py`, lines 37–43](../src/train.py#L37-L43).
- **Feature engineering and selection:** No new features were constructed — the 10 FWI system indices are domain-engineered features that already encode meteorological relationships (e.g., FWI combines ISI and BUI). Wind speed (`Ws`) showed near-zero correlation with the target (r = −0.07) and was considered for removal; it was retained because it feeds into the ISI calculation and dropping it would break FWI system integrity. See EDA findings in [`notebooks/eda.py`](../notebooks/eda.py) and `notebooks/figures/04_correlation_heatmap.png`.

#### 2A.3 Model Selection

- **Models tested:** Random Forest (`n_estimators=200, max_depth=10`) and Logistic Regression (`C=1.0, max_iter=1000`). See [`src/config.py`, lines 37–50](../src/config.py#L37-L50).
- **Why these models were chosen:** Random Forest handles non-linear feature interactions (FWI indices interact multiplicatively) without requiring feature scaling, provides native feature importances for the NLP integration, and is robust on small datasets. Logistic Regression serves as an interpretable linear baseline and provides calibrated probabilities. Both are well-suited to binary classification on tabular data of this size.

#### 2A.4 Model Comparison and Iterations

| Iteration | Objective | Key changes | Models used | Main metric | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Baseline — establish that the task is learnable | Default sklearn parameters, all 10 features, 80/20 stratified split | RF, LR | Accuracy | RF: 95.9%, LR: 89.8% (baseline) |
| 2 | Improve RF depth and ensemble size | RF: `n_estimators` 100→200, `max_depth` None→10 (prevent overfit); LR: added `max_iter=1000` for convergence | RF, LR | ROC-AUC | RF: 1.000 (+), LR: 0.993 (+); LR convergence errors resolved |
| 3 | Validate stability with 5-fold CV | Added `StratifiedKFold(n_splits=5)` cross-validation on full dataset | RF, LR | CV ROC-AUC | RF: 0.997 ± 0.004; LR: 0.993 ± 0.011 — RF more stable |

See [`src/train.py`, lines 54–105](../src/train.py#L54-L105) and evaluation figures `notebooks/figures/08_confusion_matrices.png`, `09_roc_curves.png`, `11_cv_scores.png`.

#### 2A.5 Evaluation and Error Analysis

- **Metrics used:** Accuracy, ROC-AUC, Precision/Recall/F1 per class, Confusion Matrix, 5-fold stratified cross-validation AUC.
- **Final results (test set, n=49):**

| Model | Accuracy | ROC-AUC | Precision (fire) | Recall (fire) | F1 (fire) |
|---|---|---|---|---|---|
| **Random Forest** | **97.96%** | **1.000** | **0.966** | **1.000** | **0.983** |
| Logistic Regression | 93.88% | 0.993 | 0.931 | 0.964 | 0.947 |

- **Error patterns and likely causes:** Random Forest made 1 error: 1 false positive (predicted fire, actual not-fire). Logistic Regression made 3 errors: 2 false positives and 1 false negative. The single RF false positive occurred on a borderline observation with moderate FWI (≈12) and elevated temperature — a day the model conservatively rated as fire-risk. False positives are preferable to false negatives in a safety-critical application. The small dataset size (243 rows) means individual borderline observations have a measurable impact on test metrics.

See [`notebooks/figures/08_confusion_matrices.png`](../notebooks/figures/08_confusion_matrices.png).

#### 2A.6 Integration with Other Block(s)

- **Inputs received from other block(s):** None — the ML block processes raw numeric sensor inputs directly from the user.
- **Outputs provided to other block(s):** The `predict()` function in [`src/predict.py`](../src/predict.py) returns a dict `{ label, probability, top_features }`. This dict is passed unchanged to the NLP block as the grounding context for prompt construction. See [`src/nlp.py`, lines 31–52](../src/nlp.py#L31-L52) for how these values are injected into the prompt.

---

### 2B. NLP

#### 2B.1 Data Source(s)

| Entry | Source name or link | Type | Size | Role in this block |
| --- | --- | --- | --- | --- |
| 1 | ML model output (from Block 2A) | Structured dict (label, probability, top-3 features) | 1 record per inference call | Primary input — grounds every LLM response in the actual prediction |
| 2 | User-provided weather/FWI readings | Numeric inputs (10 values) | 1 record per inference call | Included in prompt context so LLM can reference raw sensor values |
| 3 | User follow-up questions (live chat) | Free-form natural language text | Variable | Input to the multi-turn chat handler |

#### 2B.2 Preprocessing and Prompt Design

- **Text preprocessing:** No classical NLP preprocessing (tokenisation, stemming, etc.) is required. The numeric ML output is formatted into a structured plain-text context block with labelled sections (`=== ML Prediction ===`, `=== Sensor Readings ===`, `=== FWI System Indices ===`) to maximise LLM readability and prevent the model from misattributing values. See [`src/nlp.py`, lines 31–52](../src/nlp.py#L31-L52).
- **Prompt design:** Three strategies were implemented and evaluated. All share the same system prompt (`SYSTEM_PROMPT`, [`src/nlp.py`, lines 55–65](../src/nlp.py#L55-L65)) which establishes the WildfireAdvisor persona and constraints (factual, calm, never invent numbers). User-turn prompts differ in structure and length:
  - **Strategy A (Brief):** 2-sentence summary + 3 bullets, ≤100 words — [`src/nlp.py`, lines 68–83](../src/nlp.py#L68-L83)
  - **Strategy B (Detailed):** 4-section structured response (risk assessment, index glossary, action plan, caveats) — [`src/nlp.py`, lines 86–107](../src/nlp.py#L86-L107)
  - **Strategy C (Structured):** Explicit markdown headings for programmatic rendering — [`src/nlp.py`, lines 110–132](../src/nlp.py#L110-L132)

#### 2B.3 Approach Selection

- **Approach used:** Prompt engineering with a hosted LLM (GPT-4o-mini via OpenAI API). The ML prediction output serves as structured context injected into each prompt — this is a lightweight form of retrieval augmentation where the "retrieved" content is the ML model's decision and feature importances.
- **Alternatives considered:** RAG over a corpus of fire safety documents was considered but rejected: the FWI system indices provide sufficient domain specificity without requiring a document corpus, and a simpler architecture is more maintainable. Classical NLP (sentiment, keyword extraction) was not applicable — the task requires generation, not classification.

#### 2B.4 Comparison and Iterations

| Iteration | Objective | Key changes | Model or prompt setup | Main metric or qualitative check | Change vs previous |
| --- | --- | --- | --- | --- | --- |
| 1 | Establish baseline output quality | Single prompt with unstructured context; no persona | GPT-4o-mini, no system prompt | Qualitative: relevance to prediction | Generic fire advice, not grounded in ML output |
| 2 | Ground output in ML features | Added structured context block with labelled sections; added SYSTEM_PROMPT with factual constraint | Strategy A (Brief) | Qualitative: feature references present | Responses now cite specific feature values (e.g. "ISI of 16.0") |
| 3 | Compare 3 strategies for completeness and actionability | Implemented Strategies B and C; manual scoring on 4 axes across 3 scenarios | Strategies A, B, C | Manual score (0–3 on Clarity, Completeness, Actionability, UI fit) | B and C tied at 11/12; B selected as default for completeness |

Full outputs and scoring in [`notebooks/prompt_evaluation_results.md`](../notebooks/prompt_evaluation_results.md).

#### 2B.5 Evaluation and Error Analysis

- **Evaluation strategy:** Manual qualitative evaluation across 3 representative scenarios (High/Medium/Low risk) × 3 strategies = 9 outputs. Each scored 0–3 on: Clarity, Completeness (covers risk reason + index explanation + action plan), Actionability (specific, prioritised steps), UI Fit (suitable for Gradio rendering). See [`notebooks/evaluate_prompts.py`](../notebooks/evaluate_prompts.py).
- **Results:**

| Strategy | Clarity | Completeness | Actionability | UI Fit | Total |
|---|---|---|---|---|---|
| A: Brief | 3 | 1 | 2 | 2 | **8** |
| **B: Detailed** | 2 | 3 | 3 | 3 | **11** ← default |
| C: Structured | 3 | 2 | 3 | 3 | **11** |

- **Error patterns and likely causes:** Strategy A consistently omits the FWI index glossary, leaving users without context for the numbers. At Low Risk (0% probability), all strategies tend toward generic "maintain standard safety practices" advice — acceptable given there is genuinely little to advise, but less differentiated. Strategy C occasionally produces overly brief "Why This Risk?" sections when constrained by the heading format. The LLM sometimes hedges with "conditions can change" even when the current risk is extreme — appropriate caution but can dilute urgency.

#### 2B.6 Integration with Other Block(s)

- **Inputs received from other block(s):** The prediction dict from the ML block `{ label, probability, top_features }` and the raw input values dict are both passed to every NLP function. See [`app.py`, lines 60–80](../app.py#L60-L80) for how `run_prediction()` calls both `predict()` and `explain_detailed()` and threads the results through the UI state.
- **Outputs provided to other block(s):** The NLP block is the final output layer — its text is displayed directly to the user and drives the multi-turn chat. It does not feed back into the ML block.

---

## 3. Deployment

- **Deployment URL:** https://huggingface.co/spaces/stoerpas/wildfire-advisorywildfire-advisor
- **Main user flow:**
  1. User selects a preset scenario (or adjusts sliders manually)
  2. Clicks **Assess Risk** → ML model returns prediction in <1s; OpenAI call returns explanation in ~2–4s
  3. Risk gauge (colour-coded red/orange/green), feature importance bars, and detailed AI explanation are displayed
  4. User asks follow-up questions in the chat panel; responses are grounded in the current session's prediction
  5. User can switch ML model (Random Forest / Logistic Regression) and re-run to compare outputs

- **Screenshots:** See `docs/screenshots/` folder in the repository for annotated screenshots of:
  - High-risk scenario with red gauge and evacuation advice
  - Low-risk scenario with green gauge
  - Chat follow-up interaction

**Environment variable required on HF Spaces:**  
Set `OPENAI_API_KEY` as a Space secret under *Settings → Variables and secrets → New secret*.  
Without it, the ML prediction still runs and a warning is shown; the chat and explanation panels require the key.

---

## 4. Execution Instructions

- **Environment setup:**
```bash
git clone https://github.com/[your-username]/wildfire-advisor.git
cd wildfire-advisor
pip install -r requirements.txt
cp .env.example .env
# Edit .env → set OPENAI_API_KEY=sk-...
```

- **Data setup:**
  1. Download **Algerian Forest Fires** from [Kaggle](https://www.kaggle.com/datasets/nitinchoudhary012/algerian-forest-fires-dataset) → place `Algerian_forest_fires_dataset_UPDATE.csv` in `data/raw/`
  2. (Optional for EDA) Download **1.88M US Wildfires** from [Kaggle](https://www.kaggle.com/datasets/rtatman/188-million-us-wildfires) → place in `data/raw/`

- **Training command (run once — skip if `models/` already contains `.pkl` files):**
```bash
python -m src.train
# Outputs: models/random_forest.pkl, logistic_regression.pkl, scaler.pkl
# Outputs: notebooks/figures/08_* through 11_* (evaluation figures)
```

- **EDA command (optional):**
```bash
python notebooks/eda.py
# Outputs: notebooks/figures/01_* through 07_* (EDA figures)
```

- **Prompt evaluation command (requires OPENAI_API_KEY):**
```bash
python notebooks/evaluate_prompts.py
# Outputs: notebooks/prompt_evaluation_results.md
```

- **Inference / run command:**
```bash
python app.py
# Opens Gradio UI at http://localhost:7860
```

- **Reproducibility notes:**
  - All random seeds fixed at 42 (`RANDOM_STATE` in [`src/config.py`, line 52](../src/config.py#L52))
  - Model hyperparameters defined centrally in `src/config.py` — no magic numbers in training code
  - Pre-trained `.pkl` files can be committed to the repo to skip the training step entirely
  - Python 3.10+ required (uses `list[dict]` type hints); tested on Python 3.12
  - Key package versions: `scikit-learn>=1.4`, `gradio>=4.36`, `openai>=1.30`

---

## 5. Optional Bonus Evidence

- [ ] Third selected block implemented with strong quality
- [x] More than two data sources used with clear added value
- [ ] A core section is done exceptionally well
- [x] Extended evaluation
- [x] Ethics, bias, or fairness analysis
- [ ] Creative or exceptional use case

**Evidence for selected bonus items:**

**More than two data sources:** The 1.88M US Wildfires dataset was used during EDA to contextualise the Algerian dataset — specifically to verify that seasonal fire patterns (peak in summer months) and size distributions are consistent across geographies, strengthening the validity of conclusions drawn from the smaller Algerian dataset. See [`notebooks/eda.py`](../notebooks/eda.py).

**Extended evaluation:** Beyond standard test-set metrics, the project includes: (1) 5-fold stratified cross-validation for both ML models, (2) systematic qualitative evaluation of 3 NLP prompt strategies across 3 risk scenarios scored on 4 independent axes, and (3) explicit error analysis identifying the type, count, and likely cause of every misclassification. See `notebooks/figures/11_cv_scores.png` and [`notebooks/prompt_evaluation_results.md`](../notebooks/prompt_evaluation_results.md).

**Ethics, bias, and fairness:** Several limitations and ethical considerations apply to this system:

1. **Geographic bias:** The ML model was trained exclusively on data from two regions of Algeria (summer 2012). Performance may degrade significantly on weather patterns from other climates (e.g., Mediterranean vs. boreal forests, Southern Hemisphere seasons). The app's FWI reference panel notes this limitation explicitly.

2. **Temporal bias:** A single year (2012) is represented. Long-term climate shifts (drier summers, earlier fire seasons) mean the training distribution may not reflect current conditions. The model has no mechanism to account for climate trend drift.

3. **False negative risk:** In a safety-critical application, a false negative (predicting "No Fire Risk" when fire is likely) is more dangerous than a false positive. The RF model achieved 0 false negatives on the test set, but this cannot be guaranteed on out-of-distribution inputs. Users are reminded in the app that the model is for educational purposes only and should not replace official fire authority guidance.

4. **LLM hallucination risk:** GPT-4o-mini is instructed never to invent numbers, but prompt injection or adversarial inputs could elicit hallucinated values. The system prompt explicitly constrains the model to reference only provided values. Chat inputs are not sanitised beyond what the OpenAI API provides.

5. **Access inequality:** The NLP explanation layer requires an OpenAI API key with associated cost. In a real deployment for public safety use, this creates a dependency on a commercial service that may be unavailable during emergencies or for under-resourced agencies.
