WildSight is an interactive AI-powered platform designed to analyze endangered species data and provide actionable conservation insights. The system uses a Random Forest classifier to predict critically endangered species based on environmental and human factors.

Key Features

Upload Your Dataset: Users can upload CSV files with endangered species data.

Predict Critically Endangered Status: Machine learning model predicts risk based on climate impact, deforestation rate, and poaching incidents.

Conservation Advice: Automatically provides tailored advice based on environmental and human threat factors.

Interactive Visualizations:

Bar chart of endangered species by habitat.

Trend of critically endangered species over the years.

Feature importance chart for model interpretability.

Downloadable Reports: Export endangered species report with conservation advice as a CSV file.

Predict New Species Risk: Sidebar allows users to input new species data and get real-time risk predictions.

Technologies Used

Python – core programming language

Streamlit – interactive web interface

Pandas & NumPy – data manipulation

Scikit-learn – machine learning (Random Forest classifier)

Matplotlib & Seaborn – data visualization

How It Works

Upload a CSV dataset containing species information including habitat, food, environment, and threat factors.

The system filters data from the past 4 years and preprocesses it.

Features like climate risk, deforestation, and poaching are used to train the ML model.

Critically endangered species are flagged, with conservation advice generated automatically.

Users can explore visualizations and trends, and download the report for further analysis.

Sidebar allows predicting the risk for new species using selected environmental parameters.

Use Cases

Conservationists and researchers can identify high-risk species early.

Policymakers can prioritize conservation efforts based on predicted threats.

Provides insights for simulating environmental changes and their impact on species survival.