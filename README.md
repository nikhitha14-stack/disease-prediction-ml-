# Disease Prediction from Symptoms — Machine Learning Project

Predicts the most likely disease from a list of reported symptoms, then returns a description of that
disease and recommended precautions — trained on the Kaggle **Disease Symptom Prediction** dataset
(41 diseases, 132 symptoms).

> ⚠️ Academic/portfolio project trained on a small, synthetic dataset. **Not** a validated medical tool.

## What's inside

- Cleans and reconciles messy symptom text across 4 source files (spacing/naming inconsistencies)
- Builds a severity-weighted feature matrix (not just binary presence/absence) + 3 engineered features
- Trains and compares **7 classifiers**: Logistic Regression, KNN, Naive Bayes, Decision Tree, Random
  Forest, Gradient Boosting, SVM
- Validates with **stratified 5-fold cross-validation**, tunes the best model with **GridSearchCV**
- Full evaluation: classification report, 41×41 confusion matrix, feature importance, PCA visualization
- **Robustness test**: simulates incomplete symptom reporting to see how much accuracy degrades
- Ends with a working `predict_disease()` function that chains prediction → description → precautions

## Project structure

```
disease-prediction-project/
├── Disease_Prediction.ipynb   # main notebook — run this
├── data/
│   ├── dataset.csv                 # 4,920 patient records
│   ├── Symptom-severity.csv        # severity weight per symptom
│   ├── symptom_Description.csv     # description per disease
│   └── symptom_precaution.csv      # precautions per disease
├── requirements.txt
├── README.md
└── .gitignore
```

## How to run it

1. Install Python 3.9+.
2. Create a virtual environment (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Launch Jupyter and run `Disease_Prediction.ipynb` top to bottom:
   ```bash
   jupyter notebook
   ```

## Dataset source

[Disease Symptom Prediction](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset) on Kaggle.

## Honest limitations (worth knowing, not just having)

- Every disease maps to a fixed, non-overlapping set of symptom patterns — the dataset is synthetically
  balanced, which is why accuracy is close to 100%. Real clinical data is far noisier and more ambiguous.
- No demographic or history features (age, sex, pre-existing conditions).
- Static severity weights, not a true measure of symptom intensity over time.

## Next steps

- Inject controlled noise/overlap between diseases to make the task harder and more realistic
- Explain individual predictions with SHAP
- Wrap `predict_disease()` in a small Streamlit/Flask app
- Combine with a larger, real-world clinical dataset
