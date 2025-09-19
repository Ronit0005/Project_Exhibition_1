from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.metrics import recall_score
from sklearn.ensemble import VotingClassifier
import numpy as np
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")),("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("onehot", OneHotEncoder(handle_unknown="ignore"))])
    full_pipeline = ColumnTransformer([("num", num_pipeline, num_attribs),("cat",cat_pipeline, cat_attribs)])
    return full_pipeline

def features_preprocessing(df):
    df_copy = df.copy()

    if "Blood Pressure" in df_copy.columns:
        df_copy.dropna(subset=['Blood Pressure'], inplace=True)
        
        df_copy[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = df_copy['Blood Pressure'].str.split('/', expand=True).astype(float)
        df_copy.drop("Blood Pressure", axis=1, inplace=True)

    if 'Patient ID' in df_copy.columns:
        df_copy.drop("Patient ID", axis=1, inplace=True)

    return df_copy


main_Data = pd.read_csv("main_set.csv")
main_Data= main_Data[['Patient ID', 'Age', 'Sex', 'Cholesterol', 'Blood Pressure',
       'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
       'Previous Heart Problems', 'Medication Use', 'Stress Level',
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Country',
       'Continent', 'Hemisphere','Heart Attack Risk']]

main_Data.dropna(subset=['Cholesterol'], inplace=True)
main_Data.reset_index(drop=True, inplace=True)

main_Data['cholesterol_cat'] = pd.cut(main_Data["Cholesterol"],bins=[-np.inf, 140, 160, 180, 200, np.inf],
labels=[1, 2, 3, 4, 5])


split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)


for train_index,test_index in split.split(main_Data, main_Data['cholesterol_cat']):
        Data = main_Data.loc[train_index]
        Data=Data.drop("cholesterol_cat", axis=1)
        strat_test_set = main_Data.loc[test_index].drop("cholesterol_cat",axis=1)

strat_test_set.to_csv("Input.csv",index=False)
Data_labels = Data["Heart Attack Risk"].copy()
Data_features = Data.drop("Heart Attack Risk", axis=1)



Data_features[['Systolic Blood Pressure', 'Diastolic Blood Pressure']] = Data_features['Blood Pressure'].str.split('/', expand=True).astype(float)
Data_features.drop("Blood Pressure",axis=1,inplace=True)
Data_features.drop("Patient ID",axis=1,inplace=True)

Data_nums = Data_features.select_dtypes(include=[np.number])
Data_cat = Data_features.select_dtypes(include=[object])


num_attribs = Data_nums.columns.tolist()
cat_attribs = Data_cat.columns.tolist()


pipeline = build_pipeline(num_attribs, cat_attribs)


Data_prepared = pipeline.fit_transform(Data_features)


# Decision Tree Classifier :-
tree_reg = DecisionTreeClassifier(random_state=42)
tree_reg.fit(Data_prepared, Data_labels)

tree_rmses = -cross_val_score(
 tree_reg,
    Data_prepared,
    Data_labels,
    scoring="neg_root_mean_squared_error",
    cv=100
)

print("\nCross-Validation Performance (Decision Tree):")
print(pd.Series(tree_rmses).describe()*100)


# Linear Regression Classifier :-
lin_reg = LinearRegression()
lin_reg.fit(Data_prepared, Data_labels)

lin_rmses = -cross_val_score(
 lin_reg,
    Data_prepared,
    Data_labels,
    scoring="neg_root_mean_squared_error",
    cv=100
)

print("\nCross-Validation Performance (Linear Regression):")
print(pd.Series(lin_rmses).describe()*100)


# Random Forest Classifier :-
rand_reg = RandomForestClassifier(random_state=42)
rand_reg.fit(Data_prepared, Data_labels)

rand_rmses = -cross_val_score(
 rand_reg,
    Data_prepared,
    Data_labels,
    scoring="neg_root_mean_squared_error",
    cv=100
)

print("\nCross-Validation Performance (Random Forest):")
print(pd.Series(rand_rmses).describe()*100)

# Gradient Boosting Classifier :-
grad_reg = RandomForestClassifier(random_state=42)
grad_reg.fit(Data_prepared, Data_labels)

grad_rmses = -cross_val_score(
 grad_reg,
    Data_prepared,
    Data_labels,
    scoring="neg_root_mean_squared_error",
    cv=100
)

print("\nCross-Validation Performance (Gradient Boosting ):")
print(pd.Series(grad_rmses).describe()*100)

# Xgboost

xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_scores = -cross_val_score(xgb_clf, Data_prepared, Data_labels, scoring="neg_root_mean_squared_error", cv=100)
print("Cross-Validation Accuracy (XGBoost):")
print(pd.Series(xgb_scores).describe()*100)

# LightGBM
lgbm_clf = LGBMClassifier(random_state=42)
lgbm_scores = -cross_val_score(lgbm_clf, Data_prepared, Data_labels, scoring="neg_root_mean_squared_error", cv=100)

print("Cross-Validation Accuracy (LightGBM):")
print(pd.Series(lgbm_scores).describe()*100)

# CatBoost
cat_clf = CatBoostClassifier(random_state=42, verbose=0)
cat_scores = cross_val_score(cat_clf,
                             Data_prepared,
                             Data_labels,
                             scoring="neg_root_mean_squared_error", cv=100)

print("Cross-Validation Accuracy (CatBoost):")
print(pd.Series(cat_scores).describe()*100)


# Grid Search
param_grid = [
    {'n_estimators': [50, 100, 200],
    'max_features': [8, 10, 12, 14]},
    {'bootstrap': [False],
     'n_estimators': [50, 100],
     'max_features': [4, 6, 8]},
]

rand_reg = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rand_reg, param_grid, cv=5,
                           scoring='accuracy',
                           return_train_score=True)

grid_search.fit(Data_prepared, Data_labels)

print("Grid Search")
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation accuracy: ", grid_search.best_score_*100)


# Ensemble Model : Training 3 models LightGBM , XGBoost , CatBoost

lgbm_clf = LGBMClassifier(random_state=42)
lgbm_clf.fit(Data_prepared, Data_labels)
print("LightGBM training completed")

xgb_clf = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
xgb_clf.fit(Data_prepared, Data_labels)
print("XGBoost training completed")

cat_clf = CatBoostClassifier(random_state=42, verbose=0)
cat_clf.fit(Data_prepared, Data_labels)
print("CatBoost training complete")

voting_clf = VotingClassifier(
    estimators=[
        ('lgbm', lgbm_clf),
        ('xgb', xgb_clf),
        ('cat', cat_clf)
    ],
    voting='soft' 
)

voting_clf.fit(Data_prepared, Data_labels)
print("Ensemble model training complete")

strat_test_set_processed = features_preprocessing(strat_test_set)

y_test = strat_test_set_processed["Heart Attack Risk"].copy()
X_test_processed = strat_test_set_processed.drop("Heart Attack Risk", axis=1)

X_test_prepared = pipeline.transform(X_test_processed)

final_predictions = voting_clf.predict(X_test_prepared)

print("Final Model Performance Report:")
print(classification_report(y_test, final_predictions))

print(f"Accuracy of ensemble model {accuracy_score(y_test, final_predictions):.3f}")