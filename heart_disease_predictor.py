import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('XGBoost.pkl')  # 加载训练好的XGBoost模型

# Define the feature options
cp_options = {
    1: 'Typical Angina (1)',  # 典型心绞痛
    2: 'Atypical Angina (2)',  # 非典型心绞痛
    3: 'Non-Anginal Pain (3)',  # 非心绞痛
    4: 'Asymptomatic (4)'  # 无症状
}

restecg_options = {
    0: 'Normal (0)',  # 正常
    1: 'ST-T Wave Abnormality (1)',  # ST-T波异常
    2: 'Left Ventricular Hypertrophy (2)'  # 左心室肥厚
}

slope_options = {
    1: 'Up-sloping (1)',  # 上升坡（1）
    2: 'Flat (2)',  # 平坦（2）
    3: 'Down-sloping (3)'  # 下降坡（3）
}

thal_options = {
    3: 'Normal (3)',  # 正常
    6: 'Fixed Defect (6)',  # 固定缺损
    7: 'Reversible Defect (7)'  # 可逆缺损
}

# Streamlit UI
st.title("Heart Disease Predictor")  # 心脏病预测器

# Sidebar for input options
st.sidebar.header("Input Sample Data")  # 侧边栏输入样本数据

# Age input
age = st.sidebar.number_input("Age:", min_value=1, max_value=120, value=50)  # 年龄输入框

# Gender input
sex = st.sidebar.selectbox("Gender (0=Female, 1=Male):", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')  # 性别选择框

# Chest pain type input
cp = st.sidebar.selectbox("Chest Pain Type:", options=list(cp_options.keys()), format_func=lambda x: cp_options[x])  # 胸痛类型选择框

# Resting blood pressure input
trestbps = st.sidebar.number_input("Resting Blood Pressure (trestbps):", min_value=50, max_value=200, value=120)  # 静息血压输入框

# Serum cholesterol input
chol = st.sidebar.number_input("Serum Cholesterol (chol):", min_value=100, max_value=600, value=200)  # 血清胆固醇输入框

# Fasting blood sugar > 120 mg/dl input
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')  # 空腹血糖输入框

# Resting electrocardiographic results input
restecg = st.sidebar.selectbox("Resting Electrocardiographic Results:", options=list(restecg_options.keys()), format_func=lambda x: restecg_options[x])  # 静息心电图结果输入框

# Max heart rate input
thalach = st.sidebar.number_input("Max Heart Rate (thalach):", min_value=50, max_value=250, value=150)  # 最大心率输入框

# Exercise induced angina input
exang = st.sidebar.selectbox("Exercise Induced Angina (exang):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')  # 运动诱发的心绞痛输入框

# ST depression input
oldpeak = st.sidebar.number_input("ST Depression Relative to Rest (oldpeak):", min_value=0.0, max_value=10.0, value=1.0)  # ST段相对于静息期的压低输入框

# Slope of the peak exercise ST segment input
slope = st.sidebar.selectbox("Slope of the Peak Exercise ST Segment (slope):", options=list(slope_options.keys()), format_func=lambda x: slope_options[x])  # 峰值运动ST段的坡度输入框

# Number of major vessels input
ca = st.sidebar.number_input("Number of Major Vessels Colored by Fluoroscopy (ca):", min_value=0, max_value=4, value=0)  # 冯光学显影下着色的主要血管数量输入框

# Thal type input
thal = st.sidebar.selectbox("Thal Type:", options=list(thal_options.keys()), format_func=lambda x: thal_options[x])  # Thal类型选择框

# Process the input and make a prediction
feature_values = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]  # 收集所有输入的特征
features = np.array([feature_values])  # 转换为NumPy数组

if st.button("Make Prediction"):  # 如果点击了预测按钮
    # Predict the class and probabilities
    predicted_class = model.predict(features)[0]  # 预测心脏病类别
    predicted_proba = model.predict_proba(features)[0]  # 预测各类别的概率

    # Display the prediction results
    st.write(f"**Predicted Class:** {predicted_class}")  # 显示预测的类别
    st.write(f"**Prediction Probabilities:** {predicted_proba}")  # 显示各类别的预测概率

    # Generate advice based on the prediction result
    probability = predicted_proba[predicted_class] * 100  # 根据预测类别获取对应的概率，并转化为百分比

    if predicted_class == 1:  # 如果预测为心脏病
        advice = (
            f"According to our model, your risk of heart disease is high. "
            f"The probability of you having heart disease is {probability:.1f}%. "
            "Although this is just a probability estimate, it suggests that you might have a higher risk of heart disease. "
            "I recommend that you contact a cardiologist for further examination and assessment, "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )  # 如果预测为心脏病，给出相关建议
    else:  # 如果预测为无心脏病
        advice = (
            f"According to our model, your risk of heart disease is low. "
            f"The probability of you not having heart disease is {probability:.1f}%. "
            "Nevertheless, maintaining a healthy lifestyle is still very important. "
            "I suggest that you have regular health check-ups to monitor your heart health, "
            "and seek medical attention if you experience any discomfort."
        )  # 如果预测为无心脏病，给出相关建议

    st.write(advice)  # 显示建议

    # Visualize the prediction probabilities
    sample_prob = {
        'Class_0': predicted_proba[0],  # 类别0的概率
        'Class_1': predicted_proba[1]  # 类别1的概率
    }

    # Set figure size
    plt.figure(figsize=(10, 3))  # 设置图形大小

    # Create bar chart
    bars = plt.barh(['Not Sick', 'Sick'], 
                    [sample_prob['Class_0'], sample_prob['Class_1']], 
                    color=['#512b58', '#fe346e'])  # 绘制水平条形图

    # Add title and labels, set font bold and increase font size
    plt.title("Prediction Probability for Patient", fontsize=20, fontweight='bold')  # 添加图表标题，并设置字体大小和加粗
    plt.xlabel("Probability", fontsize=14, fontweight='bold')  # 添加X轴标签，并设置字体大小和加粗
    plt.ylabel("Classes", fontsize=14, fontweight='bold')  # 添加Y轴标签，并设置字体大小和加粗

    # Add probability text labels, adjust position to avoid overlap, set font bold
    for i, v in enumerate([sample_prob['Class_0'], sample_prob['Class_1']]):  # 为每个条形图添加概率文本标签
        plt.text(v + 0.0001, i, f"{v:.2f}", va='center', fontsize=14, color='black', fontweight='bold')  # 设置标签位置、字体加粗

    # Hide other axes (top, right, bottom)
    plt.gca().spines['top'].set_visible(False)  # 隐藏顶部边框
    plt.gca().spines['right'].set_visible(False)  # 隐藏右边框

    # Show the plot
    st.pyplot(plt)  # 显示图表