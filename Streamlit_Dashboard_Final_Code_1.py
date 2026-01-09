import pandas as pd
import joblib

# Load the raw dataset
df = pd.read_csv(r"C:\Users\Nidish Kumaar V\OneDrive\Viknesh\GUVI - Data Science Course Materials\Capstone Projects\NutriClass Food Classification Using Nutritional Data Project\synthetic_food_dataset_imbalanced.csv")

# Create a clean lookup dictionary
feature_cols = ['Calories', 'Protein', 'Fat', 'Carbs', 'Sugar', 'Fiber', 'Sodium', 'Glycemic_Index', 
                'Meal_Type', 'Preparation_Method', 'Is_Vegan', 'Is_Gluten_Free']

# Aggregate by Food Name
reverse_library = df.groupby('Food_Name')[feature_cols].first().to_dict('index')

# Save the asset
joblib.dump(reverse_library, 'reverse_food_library.pkl')
print(f"✅ Library updated with {len(reverse_library)} foods.")

import streamlit as st
import joblib
import pandas as pd
import plotly.express as px # Added for high-quality percentage charts

try:
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# 1. Load the Library
@st.cache_resource
def load_data():
    return joblib.load('reverse_food_library.pkl')

library = load_data()

# 2. Page Configuration
st.set_page_config(page_title="NutriClass Visual Predictor", layout="wide")
st.title("🔄 Interactive Food & Nutrient Dashboard")

# 3. Sidebar: Food Selection
with st.sidebar:
    st.header("1. Select Base Food")
    food_list = sorted(list(library.keys()))
    selected_food = st.selectbox("Choose Item:", food_list)
    base_data = library[selected_food]
    st.divider()
    st.caption("All values are derived from the raw dataset.")

# 4. Main Panel: Interactive Feature Overrides
st.subheader("🛠️ Interactive Categorical Features")
col_cat1, col_cat2, col_cat3, col_cat4 = st.columns(4)

with col_cat1:
    meal_options = ["Breakfast", "Lunch", "Dinner", "Snack", "Dessert"]
    default_meal_idx = meal_options.index(base_data['Meal_Type']) if base_data['Meal_Type'] in meal_options else 0
    meal_input = st.selectbox("Meal Category", meal_options, index=default_meal_idx)

with col_cat2:
    prep_options = ["Raw", "Baked", "Boiled", "Fried", "Grilled", "Steamed"]
    default_prep_idx = prep_options.index(base_data['Preparation_Method']) if base_data['Preparation_Method'] in prep_options else 0
    prep_input = st.selectbox("Prep Method", prep_options, index=default_prep_idx)

with col_cat3:
    vegan_input = st.checkbox("Is Vegan?", value=bool(base_data['Is_Vegan']))

with col_cat4:
    gf_input = st.checkbox("Is Gluten-Free?", value=bool(base_data['Is_Gluten_Free']))

st.divider()

# 5. Numerical Metrics & Percentage Visualization
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader(f"🧪 Nutrient Profile: {selected_food}")
    m1, m2 = st.columns(2)
    m1.metric("Calories", f"{base_data['Calories']} kcal")
    m1.metric("Protein", f"{base_data['Protein']}g")
    m1.metric("Fat", f"{base_data['Fat']}g")
    m1.metric("Sugar", f"{base_data['Sugar']}g")
    
    m2.metric("Fiber", f"{base_data['Fiber']}g")
    m2.metric("Sodium", f"{base_data['Sodium']}mg")
    m2.metric("Carbs", f"{base_data['Carbs']}g")
    m2.metric("Glycemic Index", f"{base_data['Glycemic_Index']}")

with col_right:
    st.subheader("📊 Nutrient Distribution (%)")
    
    # Prepare data for the Pie Chart
    # We use macronutrients for the percentage breakdown
    pie_data = pd.DataFrame({
        "Nutrient": ["Calories", "Protein", "Fat", "Carbs", "Sugar", "Fiber", "Sodium"],
        "Grams": [
            base_data['Calories'],
            base_data['Protein'], 
            base_data['Fat'], 
            base_data['Carbs'], 
            base_data['Sugar'], 
            base_data['Fiber'],
            base_data['Sodium']
        ]
    })
    
    # THE SORTING STEP
    pie_data = pie_data.sort_values(by="Grams", ascending=False)

    if HAS_PLOTLY:
        # Create an interactive donut chart with sorted labels
        fig = px.pie(
            pie_data, 
            values='Grams', 
            names='Nutrient', 
            hole=0.4,
            # category_orders ensures the legend and slices follow the sorted dataframe
            category_orders={"Nutrient": pie_data["Nutrient"].tolist()},
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        # Force sort in the visualization
        fig.update_traces(sort=False, textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.bar_chart(pie_data.set_index('Nutrient'))

# 7. SUMMARY TABLE
st.divider()
st.write("### ✅ Summary of Custom Configuration")
summary_df = pd.DataFrame({
    "Feature": ["Selected Food", "Meal Category", "Preparation", "Vegan Status", "Gluten-Free Status"],
    "Value": [selected_food, meal_input, prep_input, "Yes" if vegan_input else "No", "Yes" if gf_input else "No"]
})
st.table(summary_df)