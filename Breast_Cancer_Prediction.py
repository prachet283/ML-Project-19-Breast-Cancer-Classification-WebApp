# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 18:16:02 2024

@author: prachet
"""

import numpy as np
import pickle
import streamlit as st

#loading. the saved model
loaded_model = pickle.load(open("breast_cancer_trained_model.sav",'rb'))


def breast_cancer_prediction(input_data):

    #changing the input data to numpy
    input_data_as_numpy_array = np.asarray(input_data,dtype=np.float64)

    #reshape the array as we are predicting on 1 instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    
    return prediction


def main():
    
    #page title
    st.title('Breast Cancer Prediction using ML')
    
    #columns for input fields

    col1 , col2 , col3 , col4, col5 = st.columns(5)

    with col1:
        mean_radius = st.text_input("mean radius")
    with col2:
        mean_texture = st.text_input("mean texture")
    with col3:
        mean_perimeter = st.text_input("mean_perimeter")
    with col4:
        mean_area = st.text_input("mean_area")
    with col5:
        mean_smoothness = st.text_input("mean_smoothness")
    with col1:
        mean_compactness = st.text_input("mean_compactness")
    with col2:
        mean_concavity = st.text_input("mean_concavity")
    with col3:
        mean_concave_points = st.text_input("mean_concavepoints")
    with col4:
        mean_symmetry = st.text_input("mean_symmetry")
    with col5:
        mean_fractal_dimension = st.text_input("mean_fractal_dim")
    with col1:
        radius_error = st.text_input("radius_error")
    with col2:
        texture_error  = st.text_input("texture_error")
    with col3:
        perimeter_error = st.text_input("perimeter_error")
    with col4:
        area_error  = st.text_input("area_error")
    with col5:
        smoothness_error = st.text_input("smoothness_error")
    with col1:
        compactness_error = st.text_input("compactness_error")
    with col2:
        concavity_error = st.text_input("concavity_error")
    with col3:
        concave_points_error  = st.text_input("concave_points_error")
    with col4:
        symmetry_error = st.text_input("symmetry_error")
    with col5:
        fractal_dimension_error = st.text_input("fractal_dim_error")
    with col1:
        worst_radius = st.text_input("worst_radius")
    with col2:
        worst_texture = st.text_input("worst_texture")
    with col3:
        worst_perimeter = st.text_input("worst_perimeter")
    with col4:
        worst_area  = st.text_input("worst_area ")
    with col5:
        worst_smoothness = st.text_input("worst_smoothness")
    with col1:
        worst_compactness = st.text_input("worst_compactness")
    with col2:
        worst_concavity = st.text_input("worst_concavity")
    with col3:
        worst_concave_points = st.text_input("worst_concavepoints")
    with col4:
        worst_symmetry = st.text_input("worst_symmetry")
    with col5:
        worst_fractal_dimension = st.text_input("worst_fractal_dim")

    # code for prediction
    breast_cancer_diagnosis = ''

    #creating a button for Prediction
    if st.button('Breast Cancer Test Result'):
        breast_cancer_diagnosis=breast_cancer_prediction([[mean_radius,mean_texture,mean_perimeter,
                                             mean_area,mean_smoothness,mean_compactness,
                                             mean_concavity,mean_concave_points,mean_symmetry,
                                             mean_fractal_dimension,radius_error,
                                             texture_error,perimeter_error,area_error,smoothness_error,compactness_error,concavity_error,concave_points_error,symmetry_error,fractal_dimension_error,worst_radius,worst_texture,worst_perimeter,worst_area,worst_smoothness,worst_compactness,worst_concavity,worst_concave_points,worst_symmetry,worst_fractal_dimension]])
        if(breast_cancer_diagnosis[0]==0):
            breast_cancer_diagnosis = 'The Breast Cancer is Malignant' 
        else:
            breast_cancer_diagnosis = 'The Breast Cancer is Benign'
        st.success(breast_cancer_diagnosis)
    
    
    
if __name__ == '__main__':
    main()
    
    
    


