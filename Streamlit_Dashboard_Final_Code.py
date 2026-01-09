import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# 1. Load your raw dataset
df = pd.read_csv(r"C:\Users\Nidish Kumaar V\OneDrive\Viknesh\GUVI - Data Science Course Materials\Capstone Projects\NutriClass Food Classification Using Nutritional Data Project\NutriClass_Logic_Fixed.csv")

# 2. Basic Text Cleaning & Mapping
df['Meal_Type'] = df['Meal_Type'].astype(str).str.strip().str.capitalize()
df['Preparation_Method'] = df['Preparation_Method'].astype(str).str.strip().str.capitalize()

meal_map = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Snack': 3, 'Dessert': 4}
prep_map = {'Raw': 0, 'Baked': 1, 'Boiled': 2, 'Fried': 3, 'Grilled': 4, 'Steamed': 5}

df['Meal_Type_Encoded'] = df['Meal_Type'].map(meal_map)
df['Prep_Method_Encoded'] = df['Preparation_Method'].map(prep_map)

# 3. Handle NaNs (using median for raw numbers)
df['Meal_Type_Encoded'] = df['Meal_Type_Encoded'].fillna(df['Meal_Type_Encoded'].mode()[0])
df['Prep_Method_Encoded'] = df['Prep_Method_Encoded'].fillna(df['Prep_Method_Encoded'].mode()[0])
numeric_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 'Sodium', 'Glycemic_Index']
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# 4. Target Encoding
le = LabelEncoder()
df = df.dropna(subset=['Food_Name'])
y = le.fit_transform(df['Food_Name'])

# 5. Define Raw Features
features = numeric_cols + ['Meal_Type_Encoded', 'Prep_Method_Encoded']
X = df[features]

# 6. Train Model on Raw Data
# We use class_weight='balanced' to help with the imbalanced nature of your CSV
model_raw = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model_raw.fit(X, y)

# 7. Save for Dashboard
joblib.dump(model_raw, 'nutriclass_raw_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
joblib.dump(features, 'feature_names.pkl')

print("✅ Model trained on RAW imbalanced data and saved!")

import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load the "Raw" Artifacts
@st.cache_resource
def load_data():
    model = joblib.load('nutriclass_raw_model.pkl')
    le = joblib.load('label_encoder.pkl')
    features = joblib.load('feature_names.pkl')
    return model, le, features

model, le, features = load_data()

# 2. Streamlit UI
st.set_page_config(page_title="NutriClass Raw Dashboard", layout="wide")
st.title("🥗 NutriClass Raw Predictor")
st.write("Predictions are made directly from raw values in the imbalanced dataset.")

# 3. Input Form
with st.form("main_form"):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        calories = st.number_input("Calories", value=200.0)
        protein = st.number_input("Protein (g)", value=15.0)
        fat = st.number_input("Fat (g)", value=8.0)
        
    with col2:
        carbs = st.number_input("Carbs (g)", value=30.0)
        sugar = st.number_input("Sugar (g)", value=5.0)
        fiber = st.number_input("Fiber (g)", value=2.0)
        
    with col3:
        sodium = st.number_input("Sodium (mg)", value=350.0)
        gi = st.slider("GI Index", 0, 100, 50)
        meal_choice = st.selectbox("Meal", ["Breakfast", "Lunch", "Dinner", "Snack", "Dessert"])
        prep_choice = st.selectbox("Prep", ["Raw", "Baked", "Boiled", "Fried", "Grilled", "Steamed"])

    submit = st.form_submit_button("🚀 Predict Food Item")

# 4. Independent Prediction Logic
if submit:
    # Mapping Categories to Training Integers
    m_map = {'Breakfast': 0, 'Lunch': 1, 'Dinner': 2, 'Snack': 3, 'Dessert': 4}
    p_map = {'Raw': 0, 'Baked': 1, 'Boiled': 2, 'Fried': 3, 'Grilled': 4, 'Steamed': 5}
    
    # Create Raw Feature Vector
    input_df = pd.DataFrame([[
        calories, protein, fat, carbs, sugar, fiber, 
        sodium, gi, m_map[meal_choice], p_map[prep_choice]
    ]], columns=features)
    
    # Run Prediction
    prediction_idx = model.predict(input_df)[0]
    food_name = le.inverse_transform([prediction_idx])[0]
    
    # Confidence Level
    probs = model.predict_proba(input_df)
    confidence = np.max(probs) * 100

    # Display results on Dashboard
    st.divider()
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.success(f"### Predicted Food: **{food_name}**")
    with res_col2:
        st.metric("Model Confidence", f"{confidence:.1f}%")
    
    st.progress(int(confidence))

st.caption("Standalone Dashboard | Built for Raw Data Prediction")