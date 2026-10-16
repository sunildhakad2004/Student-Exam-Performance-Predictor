import streamlit as st
import pandas as pd
import joblib


@st.cache_resource
def load_model():
    return joblib.load("student_model.joblib")


model = load_model()
feature_columns = list(model.feature_names_in_)

st.title("Student Exam Performance Predictor")
st.write("Enter the student's details to estimate the final exam score.")

with st.form("student_score_form"):
    st.subheader("Academic profile")
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Age", min_value=12, max_value=30, value=18, step=1)
        gender = st.selectbox("Gender", ["Female", "Male", "Other"])
        education_level = st.selectbox("Education Level", ["High School", "Undergraduate"])
        school_type = st.selectbox("School Type", ["Charter", "Private", "Public"])
        family_income = st.selectbox("Family Income", ["High", "Low", "Lower-Middle", "Middle", "Upper-Middle"])
        parent_education = st.selectbox("Parent Education", ["Associate", "Bachelor", "Doctorate", "High School", "Master"])
        urban_rural = st.selectbox("Location", ["Rural", "Suburban", "Urban"])
        previous_exam_score = st.slider("Previous Exam Score (%)", 0, 100, 75)
        previous_gpa = st.slider("Previous GPA", 0.0, 4.0, 3.0, step=0.01)
        attendance_percentage = st.slider("Attendance (%)", 0, 100, 85)
        assignment_completion_rate = st.slider("Assignment Completion (%)", 0, 100, 80)
        class_participation = st.selectbox("Class Participation", ["High", "Low", "Medium"])
        study_hours_per_day = st.slider("Study Hours per Day", 0.0, 12.0, 4.0, step=0.1)
        self_study_hours = st.slider("Self-Study Hours", 0.0, 8.0, 2.0, step=0.1)
        private_tuition = st.number_input("Private Tuition (0 = No, 1 = Yes)", min_value=0, max_value=1, value=0, step=1)
        online_learning_hours = st.slider("Online Learning Hours", 0.0, 8.0, 2.0, step=0.1)
        study_consistency = st.selectbox("Study Consistency", ["High", "Low", "Medium"])
        study_environment = st.selectbox("Study Environment", ["Moderate", "Noisy", "Quiet"])
        study_method = st.selectbox("Study Method", ["Flashcards", "Group Study", "Practice Tests", "Self-Reading", "Summarizing"])
        revision_frequency = st.selectbox("Revision Frequency", ["Daily", "Rarely", "Weekly"])
        practice_tests_completed = st.number_input("Practice Tests Completed", min_value=0, max_value=20, value=8, step=1)
        notes_quality = st.selectbox("Notes Quality", ["Average", "Excellent", "Poor"])

    with col2:
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, step=0.1)
        sleep_quality = st.selectbox("Sleep Quality", ["Excellent", "Fair", "Good", "Poor"])
        daily_screen_time = st.slider("Daily Screen Time (hours)", 0.0, 12.0, 3.0, step=0.1)
        physical_activity_hours = st.slider("Physical Activity (hours)", 0.0, 8.0, 1.0, step=0.1)
        break_frequency = st.selectbox("Break Frequency", ["Frequently", "Occasionally", "Rarely"])
        stress_level = st.slider("Stress Level", 1, 10, 3)
        motivation_level = st.selectbox("Motivation Level", ["High", "Low", "Medium"])
        internet_access = st.number_input("Internet Access (0 = No, 1 = Yes)", min_value=0, max_value=1, value=1, step=1)
        device_availability = st.selectbox("Device Availability", ["Dedicated", "Shared"])
        educational_app_usage = st.selectbox("Educational App Usage", ["High", "Low", "Moderate"])
        online_course_hours = st.slider("Online Course Hours", 0.0, 8.0, 2.0, step=0.1)
        exam_difficulty = st.selectbox("Exam Difficulty", ["Easy", "Hard", "Medium"])
        exam_preparation_days = st.number_input("Exam Preparation Days", min_value=0, max_value=60, value=20, step=1)
        time_management_score = st.slider("Time Management Score", 0, 100, 75)
        exam_anxiety_level = st.slider("Exam Anxiety Level", 1, 10, 3)

    submitted = st.form_submit_button("Predict Score")

if submitted:
    student_data = {
        "age": age,
        "gender": gender,
        "education_level": education_level,
        "school_type": school_type,
        "family_income": family_income,
        "parent_education": parent_education,
        "urban_rural": urban_rural,
        "previous_exam_score": previous_exam_score,
        "previous_gpa": previous_gpa,
        "attendance_percentage": attendance_percentage,
        "assignment_completion_rate": assignment_completion_rate,
        "class_participation": class_participation,
        "study_hours_per_day": study_hours_per_day,
        "self_study_hours": self_study_hours,
        "private_tuition": private_tuition,
        "online_learning_hours": online_learning_hours,
        "study_consistency": study_consistency,
        "study_environment": study_environment,
        "study_method": study_method,
        "revision_frequency": revision_frequency,
        "practice_tests_completed": practice_tests_completed,
        "notes_quality": notes_quality,
        "sleep_hours": sleep_hours,
        "sleep_quality": sleep_quality,
        "daily_screen_time": daily_screen_time,
        "physical_activity_hours": physical_activity_hours,
        "break_frequency": break_frequency,
        "stress_level": stress_level,
        "motivation_level": motivation_level,
        "internet_access": internet_access,
        "device_availability": device_availability,
        "educational_app_usage": educational_app_usage,
        "online_course_hours": online_course_hours,
        "exam_difficulty": exam_difficulty,
        "exam_preparation_days": exam_preparation_days,
        "time_management_score": time_management_score,
        "exam_anxiety_level": exam_anxiety_level,
    }

    prediction_df = pd.DataFrame([student_data], columns=feature_columns)
    predicted_score = model.predict(prediction_df)[0]
    st.success(f"Predicted Exam Score: {predicted_score:.2f}/100")
    st.caption("This estimate is based on the trained student performance model.")