import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# 1. Load model
model_xgb = joblib.load('xgb.pkl')

# 2. Configure SHAP explainer
feature_label = [
    'IC_DL_4', 
    'Zeff_wavelet-LLL_glszm_LongRunLowGrayLevelEmphasis', 
    'Zeff_DL_19', 
    'VMI_wavelet-HLL_gldm_DependenceVariance', 
    'VMI_DL_248', 
    'IC_DL_28', 
    'IC_wavelet-LHL_glcm_ClusterShade', 
    'PEI_DL_132', 
    'VMI_wavelet-LHL_firstorder_Skewness',  
    'Zeff_wavelet-HLL_glszm_ZoneEntropy', 
    'IC_DL_167', 
    'VMI_DL_176', 
    'VMI_DL_91', 
    'IC_DL_235', 
    'IC_wavelet-HHH_glszm_GrayLevelNonUniformityNormalized', 
    'Differentiation', 
    'CT-T stage'
]

# 3. Streamlit input
st.title('Web Predictor for Post-CAS NIILs')
st.sidebar.header('Input Features')

# Input feature form
inputs = {}
for feature in feature_label:
    if feature == 'Differentiation':
        # Categorical: Well/Moderate or Poor
        inputs[feature] = st.sidebar.radio(
            feature, 
            options=['Well/Moderate', 'Poor'], 
            index=0
        )
    elif feature == 'CT-T stage':
        # Categorical: T3 or T4
        inputs[feature] = st.sidebar.radio(
            feature,
            options=['T3', 'T4'],
            index=0
        )
    else:
        inputs[feature] = st.sidebar.number_input(
            feature, 
            min_value=-10.0, 
            max_value=10.0, 
            value=0.0
        )

# Map categorical variables to numerical values
diff_map = {'Well/Moderate': 0, 'Poor': 1}
ctt_map = {'T3': 0, 'T4': 1}

inputs['Differentiation'] = diff_map[inputs['Differentiation']]
inputs['CT-T stage'] = ctt_map[inputs['CT-T stage']]

# Convert input values into a Pandas DataFrame
input_df = pd.DataFrame([inputs])

# 4. Prediction button
if st.sidebar.button('Predict'):
    try:
        # Ensure correct input data
        input_data = xgb.DMatrix(input_df)  # Pass DataFrame format data directly without .values
        prediction = model_xgb.predict(input_data)[0]  # Make prediction

        # Display prediction result
        st.subheader('Predicted probability of GR to nCRT')
        st.write(f'Predicted probability: {prediction}')

        # Compute SHAP values
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)

        # 5. Display SHAP force plot
        st.subheader('SHAP Force Plot')
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            input_df.iloc[0, :], 
            feature_names=feature_label, 
            matplotlib=True, 
            contribution_threshold=0.1
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
