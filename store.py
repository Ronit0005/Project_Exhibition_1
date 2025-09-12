import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

def build_pipeline(num_attribs, cat_attribs):
    """Builds the preprocessing pipeline for numerical and categorical features."""
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

def features_preprocessing(df):
    """Performs initial preprocessing steps on the feature set."""
    df_copy = df.copy()

    if "Blood Pressure" in df_copy.columns:
        # Drop rows where blood pressure is missing
        df_copy.dropna(subset=['Blood Pressure'], inplace=True)
        
        # Split Blood Pressure into Systolic and Diastolic columns
        try:
            df_copy[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df_copy['Blood Pressure'].str.split('/', expand=True).astype(float)
        except ValueError as e:
            print(f"Error processing Blood Pressure: {e}")
            # Handle cases where splitting might fail if format is wrong
            df_copy[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = np.nan
        df_copy.drop("Blood Pressure", axis=1, inplace=True)

    if 'Patient ID' in df_copy.columns:
        df_copy.drop("Patient ID", axis=1, inplace=True)

    return df_copy

# ==============================================================================
# 1. DATA LOADING AND INITIAL CLEANING
# ==============================================================================
try:
    main_Data = pd.read_csv("main_set.csv")
    # Select relevant columns
    main_Data = main_Data[['Patient ID', 'Age', 'Sex', 'Cholesterol', 'Blood Pressure',
                           'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
                           'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
                           'Previous Heart Problems', 'Medication Use', 'Stress Level',
                           'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
                           'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Country',
                           'Continent', 'Hemisphere','Heart Attack Risk']]

    main_Data.dropna(subset=['Cholesterol'], inplace=True)
    main_Data.reset_index(drop=True, inplace=True)
except FileNotFoundError:
    print("Error: 'main_set.csv' not found. Please ensure the dataset is in the correct directory.")
    exit()

# ==============================================================================
# 2. STRATIFIED SAMPLING
# ==============================================================================
# Create a temporary category for stratified splitting based on Cholesterol
main_Data['cholesterol_cat'] = pd.cut(main_Data["Cholesterol"],
                                      bins=[-np.inf, 140, 160, 180, 200, np.inf],
                                      labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(main_Data, main_Data['cholesterol_cat']):
    train_set = main_Data.loc[train_index]
    strat_test_set = main_Data.loc[test_index]

# Drop the temporary category column
for set_ in (train_set, strat_test_set):
    set_.drop("cholesterol_cat", axis=1, inplace=True)

# ==============================================================================
# 3. PREPROCESSING AND PIPELINE
# ==============================================================================
# Apply initial preprocessing to the entire training set to ensure consistency
train_set_processed = features_preprocessing(train_set)

# Now, separate features and labels from the processed set
y_train = train_set_processed["Heart Attack Risk"].copy()
X_train_processed = train_set_processed.drop("Heart Attack Risk", axis=1)

# Identify numerical and categorical attributes from the processed training data
num_attribs = X_train_processed.select_dtypes(include=[np.number]).columns.tolist()
cat_attribs = X_train_processed.select_dtypes(include=[object]).columns.tolist()

# Build and fit the pipeline on the training data
pipeline = build_pipeline(num_attribs, cat_attribs)
X_train_prepared = pipeline.fit_transform(X_train_processed)

# ==============================================================================
# 4. GRID SEARCH FOR RANDOM FOREST (as in original script)
# ==============================================================================
print("--- Performing GridSearchCV for RandomForestClassifier ---")
param_grid = [
    {'n_estimators': [50, 100, 200], 'max_features': [8, 10, 12]},
    {'bootstrap': [False], 'n_estimators': [50, 100], 'max_features': [4, 6, 8]},
]
forest_clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(forest_clf, param_grid, cv=5, scoring='accuracy', return_train_score=True)
grid_search.fit(X_train_prepared, y_train)

print(f"Best parameters found for RandomForest: {grid_search.best_params_}")
print(f"Best cross-validation accuracy for RandomForest: {grid_search.best_score_ * 100:.2f}%\n")


# ==============================================================================
# 5. TRAINING INDIVIDUAL CLASSIFIERS FOR ENSEMBLE
# ==============================================================================
print("--- Training Individual Models for Ensemble ---")

# Initialize models
lgbm_clf = LGBMClassifier(random_state=42)
xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
cat_clf = CatBoostClassifier(random_state=42, verbose=0)

# Train models
lgbm_clf.fit(X_train_prepared, y_train)
print("LightGBM training complete.")
xgb_clf.fit(X_train_prepared, y_train)
print("XGBoost training complete.")
cat_clf.fit(X_train_prepared, y_train)
print("CatBoost training complete.\n")

# ==============================================================================
# 6. ENSEMBLING (VOTING CLASSIFIER)
# ==============================================================================
print("--- Creating and Training Ensemble Model ---")

voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_clf),
        ('xgb', xgb_clf),
        ('cat', cat_clf)
    ],
    voting='soft'  # 'soft' voting averages probabilities and often performs better
)

voting_clf.fit(X_train_prepared, y_train)
print("Ensemble model training complete.\n")

# ==============================================================================
# 7. FINAL EVALUATION ON THE TEST SET
# ==============================================================================
print("--- Evaluating Final Ensemble Model on Unseen Test Data ---")

# Apply the same preprocessing to the entire test set
strat_test_set_processed = features_preprocessing(strat_test_set)

# Separate features and labels from the processed test set
y_test = strat_test_set_processed["Heart Attack Risk"].copy()
X_test_processed = strat_test_set_processed.drop("Heart Attack Risk", axis=1)


# IMPORTANT: Use .transform() here, not .fit_transform(), to apply the pipeline fitted on the training data
X_test_prepared = pipeline.transform(X_test_processed)

# Make predictions with the final ensemble model
final_predictions = voting_clf.predict(X_test_prepared)

# Display Comprehensive Performance Metrics
print("\nFinal Model Performance Report:\n")
print(classification_report(y_test, final_predictions))

print(f"Accuracy: {accuracy_score(y_test, final_predictions):.4f}")
print(f"F1-Score: {f1_score(y_test, final_predictions):.4f}")
print(f"Precision: {precision_score(y_test, final_predictions):.4f}")
print(f"Recall: {recall_score(y_test, final_predictions):.4f}")

