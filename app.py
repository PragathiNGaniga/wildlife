import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

st.set_page_config(page_title="WildSight: Predictive Conservation", layout="wide")
st.title("WildSight: Predictive Conservation for Endangered Species")

uploaded_file = st.file_uploader("Upload your endangered species CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(" Dataset uploaded successfully!")

    # Filter data from past 4 years
    df = df[df['assessment_year'] >= 2021]
    df.dropna(subset=['species', 'habitat', 'food', 'environment', 'status'], inplace=True)

    # Normalize text fields
    df['habitat'] = df['habitat'].str.lower().str.strip()
    df['food'] = df['food'].str.lower().str.strip()

    # Feature Engineering
    df['is_critically_endangered'] = df['status'].apply(lambda x: 1 if x == 'Critically Endangered' else 0)
    df['climate_risk'] = df['climate_change_impact'].apply(lambda x: 1 if str(x).lower() == 'high' else 0)
    df['deforestation_rate'] = pd.to_numeric(df['deforestation_rate'], errors='coerce').fillna(0)
    df['poaching_incidents'] = pd.to_numeric(df['poaching_incidents'], errors='coerce').fillna(0)

    features = ['climate_risk', 'deforestation_rate', 'poaching_incidents']
    X = df[features]
    y = df['is_critically_endangered']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    #  Evaluation
    st.subheader(" Model Evaluation")
    st.text(classification_report(y_test, y_pred))

    #  Endangered species report
    endangered_species = df[df['is_critically_endangered'] == 1]
    report = endangered_species[['species', 'habitat', 'food', 'environment', 'status']]

    #  Conservation Advice
    def get_conservation_advice(row):
        if row['climate_risk'] and row['deforestation_rate'] > 50:
            return "Urgent reforestation & climate action needed"
        elif row['poaching_incidents'] > 30:
            return "Increase anti-poaching enforcement"
        else:
            return "Maintain habitat and monitor population"

    endangered_species['conservation_advice'] = endangered_species.apply(get_conservation_advice, axis=1)

    st.subheader(" Endangered Species Report with Conservation Advice")
    st.dataframe(endangered_species[['species', 'habitat', 'food', 'environment', 'status', 'conservation_advice']])

    # Download CSV
    csv = endangered_species.to_csv(index=False).encode('utf-8')
    st.download_button("Download Report as CSV", data=csv, file_name='endangered_species_report.csv')

    # Visualization - Bar chart
    st.subheader("Visualization - Endangered Species by Habitat")
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.countplot(data=endangered_species, y='habitat', order=endangered_species['habitat'].value_counts().index, ax=ax1)
    ax1.set_title('Endangered Species by Habitat')
    st.pyplot(fig1)

    # Trend line of critically endangered species over years
    st.subheader(" Trend of Critically Endangered Species (2021-2024)")
    trend = df.groupby('assessment_year')['is_critically_endangered'].sum().reset_index()
    st.line_chart(trend.rename(columns={'assessment_year': 'Year', 'is_critically_endangered': 'Critically Endangered Count'}))

    #  Feature Importance
    st.subheader(" Feature Importance")
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    st.bar_chart(importance_df.set_index('Feature'))

    #  Sidebar: Predict risk for a new species
    st.sidebar.header(" Predict Risk for New Species")

    climate = st.sidebar.selectbox("Climate Risk", ["High", "Low"])
    deforest = st.sidebar.slider("Deforestation Rate", 0.0, 100.0, 20.0)
    poaching = st.sidebar.slider("Poaching Incidents", 0, 100, 10)

    if st.sidebar.button("Predict"):
        input_df = pd.DataFrame({
            'climate_risk': [1 if climate == "High" else 0],
            'deforestation_rate': [deforest],
            'poaching_incidents': [poaching]
        })
        pred = model.predict(input_df)[0]
        result = "Critically Endangered" if pred == 1 else " Not Critically Endangered"
        st.sidebar.success(f"Prediction: {result}")

else:
    st.info(" Please upload a dataset to begin.")
