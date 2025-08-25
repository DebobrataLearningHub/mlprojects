import os
import sys
import streamlit as st
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline
from src.exception import CustomException
from src.logger import logging

def main():
    try:
        st.title("Provide student data to predic the Math Score")
        gender = st.radio(
            "Choose your gender:",
            ("male", "female"),
            horizontal=True,
            index=0)
        
        race_ethnicity = st.radio(
            "Choose your race:",
            ("group A", "group B","group C","group D","group E"),
            horizontal=True,
            index=0)
        parental_level_of_education_options = ["bachelor's degree","some college","master's degree","associate's degree","high school","some high school","bachelor's degree"]
        parental_level_of_education = st.selectbox("Select parental education:",parental_level_of_education_options)

        lunch_options=["standard","free/reduced"]
        lunch=st.selectbox("Select lunch:",lunch_options)

        test_preparation_course_options=["none","completed"]
        test_preparation_course=st.selectbox("Select test preparation:",test_preparation_course_options)
        reading_score = st.number_input("Enter reading score:", min_value=0, max_value=100, value=30)
        writing_score = st.number_input("Enter writing score:", min_value=0, max_value=100, value=30)

        if st.button("Predic the Score"):
            custom_data = CustomData(
                gender,
                race_ethnicity,
                parental_level_of_education,
                lunch,
                test_preparation_course,
                reading_score,
                writing_score)
            
            pd_custom_data=custom_data.create_dataframe()
            
            st.write(pd_custom_data)
            predic_pipeline = PredictionPipeline()
            results = predic_pipeline.initiate_prediction_pipeline(pd_custom_data)
            results=results[0]
            st.success(results)

    except Exception as e:
        st.error(e)


if __name__ == "__main__":
    main()
