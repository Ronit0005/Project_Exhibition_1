import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# Creating a pixel file for model and data pieline
MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

# Creating a final pipeline containing numerical and categorical pipeline
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

if not os.path.exists(MODEL_FILE):

    # TRAINING PHASE

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
        strat_test_set = main_Data.loc[test_index]

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


    model = DecisionTreeClassifier(random_state=42)
    model.fit(Data_prepared, Data_labels)

    # Saving thee model and data pipeline using the pi
    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
    print("Model Has Been Trained And Saved")

else:

    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)


    input_data = pd.read_csv("Input.csv")
    input_data= input_data[['Patient ID', 'Age', 'Sex', 'Cholesterol', 'Blood Pressure',
       'Heart Rate', 'Diabetes', 'Family History', 'Smoking', 'Obesity',
       'Alcohol Consumption', 'Exercise Hours Per Week', 'Diet',
       'Previous Heart Problems', 'Medication Use', 'Stress Level',
       'Sedentary Hours Per Day', 'Income', 'BMI', 'Triglycerides',
       'Physical Activity Days Per Week', 'Sleep Hours Per Day', 'Country',
       'Continent', 'Hemisphere','Heart Attack Risk']]
    input_data= features_preprocessing(input_data)

    transformed_input_data = pipeline.transform(input_data)
    predictions = model.predict(transformed_input_data)

    input_data["Heart Attack Risk"] = predictions
    input_data.to_csv("Output.csv", index=False)

    print("Inference Has Been Complete And Results Has Been Saved As Output.csv")