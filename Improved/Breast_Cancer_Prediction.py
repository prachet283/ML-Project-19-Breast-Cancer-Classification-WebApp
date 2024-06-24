# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:16:02 2024

@author: prachet
"""

import json
import pickle
#import numpy as np
import streamlit as st
import pandas as pd

#loading. the saved model
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/columns.pkl", 'rb') as f:
    all_features = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/scaler.pkl", 'rb') as f:
    scalers = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/best_features_lr.json", 'r') as file:
    best_features_lr = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/best_features_xgb.json", 'r') as file:
    best_features_xgb = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/best_features_knn.json", 'r') as file:
    best_features_knn = json.load(file)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/parkinsons_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/parkinsons_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb = pickle.load(f)
with open("C:/Users/prachet/OneDrive - Vidyalankar Institute of Technology/Desktop/Coding/Machine Learning/ML-Project-19-Breast Cancer Classification using Machine Learning/Updated/parkinsons_disease_trained_knn_model.sav", 'rb') as f:
    loaded_model_knn = pickle.load(f)


def breast_cancer_prediction(input_data):

    df = pd.DataFrame([input_data], columns=all_features)

    df[all_features] = scalers.transform(df[all_features])
    
    df_best_features_lr = df[best_features_lr]
    df_best_features_xgb = df[best_features_xgb]
    df_best_features_knn = df[best_features_knn]
    
    prediction1 = loaded_model_lr.predict(df_best_features_lr)
    prediction2 = loaded_model_xgb.predict(df_best_features_xgb)
    prediction3 = loaded_model_knn.predict(df_best_features_knn)
    
    return prediction1 , prediction2, prediction3


def main():
    
    #page title
    st.title('Breast Cancer Prediction using ML')
    
    #columns for input fields

    col1 , col2 , col3 , col4, col5 = st.columns(5)

    with col1:
        mean_radius = st.number_input("mean radius",format="%.6f")
    with col2:
        mean_texture = st.number_input("mean texture",format="%.6f")
    with col3:
        mean_perimeter = st.number_input("mean_perimeter",format="%.6f")
    with col4:
        mean_area = st.number_input("mean_area",format="%.6f")
    with col5:
        mean_smoothness = st.number_input("mean_smoothness",format="%.6f")
    with col1:
        mean_compactness = st.number_input("mean_compactness",format="%.6f")
    with col2:
        mean_concavity = st.number_input("mean_concavity",format="%.6f")
    with col3:
        mean_concave_points = st.number_input("mean_concavepoints",format="%.6f")
    with col4:
        mean_symmetry = st.number_input("mean_symmetry",format="%.6f")
    with col5:
        mean_fractal_dimension = st.number_input("mean_fractal_dim",format="%.6f")
    with col1:
        radius_error = st.number_input("radius_error",format="%.6f")
    with col2:
        texture_error  = st.number_input("texture_error",format="%.6f")
    with col3:
        perimeter_error = st.number_input("perimeter_error",format="%.6f")
    with col4:
        area_error  = st.number_input("area_error",format="%.6f")
    with col5:
        smoothness_error = st.number_input("smoothness_error",format="%.6f")
    with col1:
        compactness_error = st.number_input("compactness_error",format="%.6f")
    with col2:
        concavity_error = st.number_input("concavity_error",format="%.6f")
    with col3:
        concave_points_error  = st.number_input("concave_points_error",format="%.6f")
    with col4:
        symmetry_error = st.number_input("symmetry_error",format="%.6f")
    with col5:
        fractal_dimension_error = st.number_input("fractal_dim_error",format="%.6f")
    with col1:
        worst_radius = st.number_input("worst_radius",format="%.6f")
    with col2:
        worst_texture = st.number_input("worst_texture",format="%.6f")
    with col3:
        worst_perimeter = st.number_input("worst_perimeter",format="%.6f")
    with col4:
        worst_area  = st.number_input("worst_area",format="%.6f")
    with col5:
        worst_smoothness = st.number_input("worst_smoothness",format="%.6f")
    with col1:
        worst_compactness = st.number_input("worst_compactness",format="%.6f")
    with col2:
        worst_concavity = st.number_input("worst_concavity",format="%.6f")
    with col3:
        worst_concave_points = st.number_input("worst_concavepoints",format="%.6f")
    with col4:
        worst_symmetry = st.number_input("worst_symmetry",format="%.6f")
    with col5:
        worst_fractal_dimension = st.number_input("worst_fractal_dim",format="%.6f")

    # code for prediction
    breast_cancer_diagnosis_lr = ''
    breast_cancer_diagnosis_knn = ''
    breast_cancer_diagnosis_xgb = ''
    
    breast_cancer_diagnosis_lr,breast_cancer_diagnosis_knn,breast_cancer_diagnosis_xgb = breast_cancer_prediction([mean_radius,mean_texture,mean_perimeter,
                                         mean_area,mean_smoothness,mean_compactness,
                                         mean_concavity,mean_concave_points,mean_symmetry,
                                         mean_fractal_dimension,radius_error,
                                         texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension])
    #creating a button for Prediction
    if st.button('Breast Cancer Test Result(LR)'):
        if(breast_cancer_diagnosis_lr[0]==0):
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant' 
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign'
        st.success(breast_cancer_diagnosis)
    if st.button('Breast Cancer Test Result(XGB)'):
        if(breast_cancer_diagnosis_xgb[0]==0):
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant' 
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign'
        st.success(breast_cancer_diagnosis)
    if st.button('Breast Cancer Test Result(KNN)'):
        if(breast_cancer_diagnosis_knn[0]==0):
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant' 
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign'
        st.success(breast_cancer_diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    


