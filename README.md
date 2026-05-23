# ClimateWins — Weather Condition Prediction with Machine Learning

> *Predicting favorable and dangerous weather conditions across Europe using supervised machine learning — built for a European climate nonprofit.*

---

## Project Overview

ClimateWins is a European nonprofit concerned with the increasing frequency of extreme weather events over the past two decades. This project delivers a supervised machine learning pipeline that predicts whether daily weather conditions at a given station will be **favorable or unfavorable** — with the longer-term goal of predicting dangerous conditions in advance.

This is a complete, end-to-end data science project covering business framing, exploratory data analysis, feature engineering, model development, business recommendations, stakeholder presentation, and ethical analysis.

---

## Business Problem

ClimateWins needs a reliable, explainable model that:
- Predicts favorable vs unfavorable weather conditions from daily measurements
- Minimises **false positives** — dangerous days incorrectly labelled as safe
- Is interpretable enough for non-technical stakeholders to trust and act on
- Generalises to genuinely unseen future dates, not just held-out random samples

**Success definition:** A model meaningfully above the 76.5% majority-class baseline, with particular focus on reducing false reassurance (false positives) — the highest-cost error type for organisations making operational decisions based on weather predictions.

---

## Dataset

| Property | Detail |
|---|---|
| Source | European Climate Assessment & Dataset (ECA&D) |
| Stations | 18 weather stations across mainland Europe |
| Period | January 2000 – January 2010 (3,653 daily records) |
| Features | Temperature, global radiation, cloud cover, humidity, precipitation, sunshine, wind, pressure |
| Target | Binary: favorable / unfavorable day (based on picnic weather proxy label) |
| Missing values | None |

**Key dataset characteristics identified during EDA:**
- Sonnblick (alpine, 3,100m) excluded — 0% favorable days across entire period, unsuitable for binary classification
- Strong temporal autocorrelation — consecutive days are related, requiring chronological splitting
- Class imbalance: 74.3% unfavorable, 25.7% favorable at Basel reference station
- Seasonal patterns confirmed coherent — data passes physical sanity checks

---

## Project Structure

```
ClimateWins/
├── Data/
│   ├── Supervised/
│   │   ├── weather_prediction_dataset.csv            # Raw weather features
│   │   ├── weather_prediction_picnic_labels.csv      # Raw target labels
│   │   └── weather_merged.csv                        # Cleaned, merged dataset
│   └── Unsupervised/
├── Notebooks/
│   ├── 01_data_understanding.ipynb                   # Loading, merging, cleaning
│   └── 02_supervised_learning_basel.ipynb            # EDA, feature analysis, modelling
├── Deliverables/
│   ├── ClimateWins_Presentation.pptx                 # Stakeholder presentation
│   └── Exercise_1_2_Ethics_and_Data_Preparation.docx # Ethics and data write-up 
└── README.md
```

---

## Methodology

### 1. Data Preparation
- Merged weather observations with picnic labels on DATE key
- Converted DATE from integer format (YYYYMMDD) to datetime
- Removed incomplete 2010 record (1 day — statistically meaningless)
- Verified zero missing values across all 3,653 records

### 2. Exploratory Data Analysis
- Confirmed seasonal temperature coherence (northern hemisphere pattern)
- Identified Sonnblick as a distributional outlier — excluded from modelling
- Analysed class balance across all 18 stations (range: 0% to 48.5% favorable)
- Constructed feature correlation matrix to identify multicollinearity

### 3. Feature Selection
Starting from 9 Basel weather features, reduced to 5 based on:
- **Target correlation analysis** — global radiation (0.656) and sunshine (0.613) are strongest predictors
- **Multicollinearity removal** — temp_min, temp_max, and sunshine dropped (correlation >0.85 with retained features)
- **Low-signal removal** — pressure dropped (target correlation: 0.041)

**Final features selected:**

| Feature | Target Correlation | Rationale |
|---|---|---|
| `BASEL_global_radiation` | +0.656 | Strongest predictor; proxy for sunshine signal |
| `BASEL_temp_mean` | +0.562 | Representative temperature; min/max redundant |
| `BASEL_cloud_cover` | -0.424 | Independent negative signal |
| `BASEL_humidity` | -0.443 | Independent negative signal |
| `BASEL_precipitation` | -0.257 | Independent negative signal |

### 4. Train/Test Split
**Critical decision: chronological split, not random shuffle.**

Weather data has strong temporal autocorrelation. A random split allows the model to see dates adjacent to test dates during training — inflating accuracy artificially. The chronological split ensures the model only ever learns from the past to predict the future.

| Split | Period | Records | Favorable Rate |
|---|---|---|---|
| Train | 2000–2007 | 2,922 | 26.2% |
| Test | 2008–2009 | 731 | 23.5% |

Seasonal distribution validated as consistent between splits — test results reflect genuine generalisation.

### 5. Feature Scaling
`StandardScaler` applied — fitted on training data only, applied to both sets. Prevents test set information leaking into preprocessing.

---

## Results

| Model | Test Accuracy | False Positives | False Negatives | Recommended |
|---|---|---|---|---|
| Baseline (majority class) | 76.5% | — | — | No |
| KNN (k=5) | 92.2% | 45 | 12 | Fallback only |
| Decision Tree (depth=4) | **98.2%** | **8** | **5** | **Yes** |

**Best model: Decision Tree (max_depth=4)**
- 21.8 percentage points above baseline
- 82% reduction in dangerous false positives vs KNN
- Train accuracy 97.8% / Test accuracy 98.2% — gap of -0.4% confirms no overfitting
- Root node splits on global radiation — consistent with feature correlation analysis
- Fully interpretable — decision logic is visible and explainable to stakeholders

---

## Key Findings

**1. Global radiation is the dominant predictor.**

The Decision Tree's root split uses global radiation at threshold 0.657. This single feature separates the majority of favorable from unfavorable days — consistent with its 0.656 correlation with the target. It functions as a composite signal encoding sunshine, cloud cover, and season simultaneously.

**2. The chronological split was the most important methodological decision.**

A random split would have produced artificially inflated accuracy due to temporal leakage. The chronological split ensures results reflect genuine forecasting ability on unseen future dates.

**3. Accuracy alone is a misleading metric for this problem.**

The baseline model achieves 76.5% accuracy by predicting "unfavorable" every day — learning nothing. Every model must be evaluated against this baseline and assessed on false positive rate specifically, given the asymmetric cost of dangerous false reassurance.

**4. Station-level variation is significant.**

Favorable day rates range from 0% (Sonnblick) to 48.5% (Perpignan). A single global model cannot serve all stations equally. Future work should evaluate station-specific models for locations with atypical climate profiles.

---

## Business Recommendations

**Recommendation 1 — Deploy the Decision Tree model for Basel.**
With 98.2% accuracy and only 8 false positives across a 2-year test period, the model is reliable enough to inform operational decisions. Its interpretability — the decision path is fully visible — allows stakeholders to understand and trust each prediction.

**Recommendation 2 — Prioritise false positive reduction over accuracy.**
For high-stakes use cases (agricultural planning, infrastructure management), consider raising the classification threshold above 50% to further reduce dangerous false reassurance, accepting more unnecessary caution in exchange.

**Recommendation 3 — Expand to additional stations with validated models.**
Basel serves as the proof-of-concept. The same methodology should be applied to each station individually, with Sonnblick excluded and stations with very low favorable rates (Oslo: 16.9%) treated with additional caution.

**Recommendation 4 — Extend the historical window.**
The 2000–2009 dataset is too short to detect climate change trends reliably. Incorporating ECA&D data from the late 1800s would enable long-term trend analysis and strengthen the model's exposure to historical extreme events.

---

## Ethical Considerations

A full ethical analysis is documented in `Deliverables/Exercise_1_2_Ethics_and_Data_Preparation.docx`. Key concerns are summarised below.

**Proxy label risk**
The "favorable day" label is based on a picnic weather definition — a human judgement call. The model is only as meaningful as this definition. For safety-critical applications, the label must be reviewed and validated by domain experts before deployment.

**Temporal bias**
The model is trained on 2000–2007 data. Climate change — the very phenomenon ClimateWins is concerned about — means historical patterns may no longer represent future conditions accurately. Regular retraining on more recent data is essential.

**Geographic bias**
Results are validated for Basel only. Applying this model to other stations without station-specific revalidation risks unreliable predictions in different climate regimes (alpine, Mediterranean, Scandinavian).

**Automation and human oversight**
Model predictions should inform decisions, not replace them. For safety-critical warnings, human review of borderline predictions is required. The Decision Tree's full interpretability — its decision path is visible and auditable — directly supports this governance requirement.

**Explainability as an ethical requirement**
A model that cannot explain its predictions is difficult to challenge or audit. The Decision Tree was selected partly for this reason: stakeholders can see exactly which weather thresholds drive each prediction.

---

## Deliverables

### Exercise 1.6 — Stakeholder Presentation
**File:** `Deliverables/ClimateWins_Presentation.pptx`

A 10-slide stakeholder-facing presentation designed for a non-technical business audience. Covers the business problem, data foundation, key EDA findings, methodology, model performance results, error type analysis, recommendations, ethical considerations, and a closing summary. No code or technical jargon — built to communicate findings to ClimateWins leadership and partner organisations.

| Slide | Content |
|---|---|
| 1 | Title and project framing |
| 2 | The problem ClimateWins is solving |
| 3 | Data foundation — key statistics |
| 4 | What the data revealed — 4 EDA findings |
| 5 | How the prediction system was built — 4-step methodology |
| 6 | Model performance results with bar chart |
| 7 | Why error type matters more than accuracy |
| 8 | Four prioritised business recommendations |
| 9 | Ethical considerations |
| 10 | Closing summary with headline statistics |

### Exercise 1.2 — Ethics, Data Landscape, and Data Preparation
**File:** `Deliverables/Exercise_1_2_Ethics_and_Data_Preparation.docx`

A formal written deliverable covering all four Exercise 1.2 requirements:

1. **Ethical issues in machine learning and automation** — why ethical issues arise, the amplification effect of automated decisions, and five specific concerns for this project including label definition risk, temporal bias, geographic limitation, automation governance, and explainability.

2. **Data landscape** — source and provenance (ECA&D), full dataset structure, weather variables measured, and three key landscape findings: station-level variation, Sonnblick exclusion rationale, and temporal data structure.

3. **Supervised vs unsupervised learning** — definitions, application to ClimateWins, and a comparison table covering training data, goal, output type, evaluation approach, and project-specific use case for each paradigm.

4. **Data preparation** — documented decisions covering loading and merging, quality assessment, feature selection with justification table, chronological train/test split rationale, and feature scaling protocol including the critical distinction between fitting and transforming.

---

## Tools & Libraries

- **Python 3.x**
- `pandas` — data manipulation and analysis
- `scikit-learn` — preprocessing, modelling, evaluation
- `matplotlib` / `seaborn` — visualisation
- `numpy` — numerical operations

---

## Author

**Benjamin Akingbade**

Early-career Data/ Business Analyst transitioning into data science.
This project was completed as part of the CareerFoundry Machine Learning with Python course
