import streamlit as st
import pandas as pd
import pickle
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.metrics import confusion_matrix

with open('best_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

st.title('Stroke Prediction App')

st.sidebar.header('User Input')

with st.sidebar.form(key='user_input_form'):
    age = st.number_input('Age', min_value=0)
    hypertension = st.radio('Hypertension', ['No', 'Yes'])
    heart_disease = st.radio('Heart Disease', ['No', 'Yes'])
    ever_married = st.radio('Ever Married', ['No', 'Yes'])
    residence_type = st.radio('Residence Type', ['Rural', 'Urban'])
    avg_glucose_level = st.number_input('Average Glucose Level', min_value=0.0)
    bmi = st.number_input('BMI', min_value=0.0)
    gender = st.radio('Gender', ['Female', 'Male', 'Other'])
    work_type = st.selectbox('Work Type', ['Govt_job', 'Never_worked', 'Private', 'Self-employed', 'children'])
    smoking_status = st.selectbox('Smoking Status', ['Unknown', 'formerly smoked', 'never smoked', 'smokes'])

    submitted = st.form_submit_button('Predict')

if submitted:
    residence_type_encoded = 1 if residence_type == 'Urban' else 0

    user_input_df = pd.DataFrame({
        'age': [age],
        'hypertension': [1 if hypertension == 'Yes' else 0],
        'heart_disease': [1 if heart_disease == 'Yes' else 0],
        'ever_married': [1 if ever_married == 'Yes' else 0],
        'Residence_type': [residence_type_encoded],  
        'avg_glucose_level': [avg_glucose_level],
        'bmi': [bmi],
        'gender_Female': [1 if gender == 'Female' else 0],
        'gender_Male': [1 if gender == 'Male' else 0],
        'gender_Other': [1 if gender == 'Other' else 0],
        'work_type_Govt_job': [1 if work_type == 'Govt_job' else 0],
        'work_type_Never_worked': [1 if work_type == 'Never_worked' else 0],
        'work_type_Private': [1 if work_type == 'Private' else 0],
        'work_type_Self-employed': [1 if work_type == 'Self-employed' else 0],
        'work_type_children': [1 if work_type == 'children' else 0],
        'smoking_Unknown': [1 if smoking_status == 'Unknown' else 0],
        'smoking_formerly smoked': [1 if smoking_status == 'formerly smoked' else 0],
        'smoking_never smoked': [1 if smoking_status == 'never smoked' else 0],
        'smoking_smokes': [1 if smoking_status == 'smokes' else 0]
    })

    prediction = model.predict(user_input_df)

    if prediction[0] == 1:
        st.write('Prediction: You are at risk of having a stroke.')
    else:
        st.write('Prediction: You are not at risk of having a stroke.')

#Charts for model performance


    TP = 285
    FP = 34
    TN = 257
    FN = 8

    confusion_matrix_values = [[TN, FP], [FN, TP]]
    confusion_matrix_labels = ['Not Stroke', 'Stroke']
    fig_cm = ff.create_annotated_heatmap(
        z=confusion_matrix_values,
        x=confusion_matrix_labels,
        y=confusion_matrix_labels,
        colorscale='Viridis',
        showscale=True
    )

    fig_cm.update_layout(
        title="Confusion Matrix",
        xaxis=dict(title="Predicted", side="bottom"),
        yaxis=dict(title="Actual"),
    )

    st.header('Confusion Matrix')
    st.plotly_chart(fig_cm)
    light_colors = {
    'light green': 'rgb(144, 238, 144)',  
    'blue': 'rgb(173, 216, 230)',        
    'red': 'rgb(255, 182, 193)',          
    'purple': 'rgb(200, 162, 200)'       
}

    model_performance_data = {
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'Score': [0.92, 0.89, 0.97, 0.93],
        'Color': ['green', 'blue', 'red', 'purple']  
    }

    df_model_performance = pd.DataFrame(model_performance_data)

    fig = px.bar(df_model_performance, x='Metric', y='Score', text='Score')
    fig.update_traces(texttemplate='%{text:.2f}', textposition='outside', marker_color=df_model_performance['Color'])

    st.header('Model Performance')
    st.plotly_chart(fig)
