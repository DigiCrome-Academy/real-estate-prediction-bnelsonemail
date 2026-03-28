"""
Real Estate Price Prediction Dashboard

Run with: streamlit run dashboard/app.py

This dashboard provides:
- Price prediction using trained models
- Property similarity recommendations
- Market segmentation visualization
"""

import streamlit as st
import numpy as np
import pandas as pd
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

st.set_page_config(
    page_title="Real Estate Price Prediction Engine",
    page_icon="🏠",
    layout="wide",
)

st.title("🏠 Real Estate Price Prediction Engine")
st.markdown("---")

# =============================================================================
# Sidebar - Navigation
# =============================================================================
page = st.sidebar.selectbox(
    "Navigate",
    ["Price Prediction", "Property Recommendations", "Market Segmentation"]
)

# =============================================================================
# TODO: Load your trained models and data
# =============================================================================
# Hints:
#   from src.data_loader import load_housing_data, preprocess_features
#   from src.ensemble import load_model
#   model = load_model('models/best_model.joblib')


if page == "Price Prediction":
    st.header("💰 Price Prediction")
    st.write("Enter property features to get a price estimate.")

    # TODO: Create input widgets for each feature
    # Example:
    # med_income = st.slider("Median Income (area)", 0.0, 15.0, 3.0)
    # house_age = st.slider("House Age", 1, 52, 20)
    # ...

    # TODO: When user clicks "Predict", run the model
    # if st.button("Predict Price"):
    #     features = np.array([[med_income, house_age, ...]])
    #     features_scaled = scaler.transform(features)
    #     prediction = model.predict(features_scaled)
    #     st.success(f"Estimated Price: ${prediction[0] * 100000:,.0f}")

    st.info("⚠️ Implement the prediction logic in src/ensemble.py first, "
            "then load your trained model here.")


elif page == "Property Recommendations":
    st.header("🔍 Property Recommendations")
    st.write("Find similar properties based on features.")

    # TODO: Let user select a property index or input features
    # TODO: Show top-N similar properties using your recommendation system

    st.info("⚠️ Implement the recommendation logic in src/recommendation.py first.")


elif page == "Market Segmentation":
    st.header("📊 Market Segmentation")
    st.write("Explore property market segments identified by clustering.")

    # TODO: Load clustering results
    # TODO: Show PCA 2D scatter plot with cluster colors
    # TODO: Show cluster statistics table

    st.info("⚠️ Implement the clustering logic in src/clustering.py first.")


# =============================================================================
# Footer
# =============================================================================
st.markdown("---")
st.markdown(
    "Built as part of the Real Estate Price Prediction Engine project. "
    "Uses the California Housing dataset from scikit-learn."
)
