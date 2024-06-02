import streamlit as st
import pickle
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
def load_model():
    with open('trained_model.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# Define the features
feature_names = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education',
    'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate',
    'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction',
    'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Initialize encoders
encoders = {
    'BusinessTravel': LabelEncoder(),
    'Department': LabelEncoder(),
    'EducationField': LabelEncoder(),
    'Gender': LabelEncoder(),
    'JobRole': LabelEncoder(),
    'MaritalStatus': LabelEncoder(),
    'Over18': LabelEncoder(),
    'OverTime': LabelEncoder()
}

# Fit the encoders with some assumed categories (this should match your training data)
encoders['BusinessTravel'].fit(['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
encoders['Department'].fit(['Sales', 'Research & Development', 'Human Resources'])
encoders['EducationField'].fit(['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
encoders['Gender'].fit(['Male', 'Female'])
encoders['JobRole'].fit(['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                         'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
encoders['MaritalStatus'].fit(['Single', 'Married', 'Divorced'])
encoders['Over18'].fit(['Y'])
encoders['OverTime'].fit(['Yes', 'No'])

# User input features
def user_input_features():
    st.sidebar.header('User Input Parameters')

    # Collect user input for each feature
    age = st.sidebar.number_input('Age', min_value=0, max_value=100, value=25)
    business_travel = st.sidebar.selectbox('Business Travel', ['Travel_Rarely', 'Travel_Frequently', 'Non-Travel'])
    daily_rate = st.sidebar.number_input('Daily Rate', min_value=0, max_value=2000, value=800)
    department = st.sidebar.selectbox('Department', ['Sales', 'Research & Development', 'Human Resources'])
    distance_from_home = st.sidebar.number_input('Distance From Home (km)', min_value=0, max_value=100, value=10)
    education = st.sidebar.selectbox('Education', [1, 2, 3, 4, 5])
    education_field = st.sidebar.selectbox('Education Field', ['Life Sciences', 'Other', 'Medical', 'Marketing', 'Technical Degree', 'Human Resources'])
    employee_count = st.sidebar.number_input('Employee Count', min_value=0, max_value=10, value=1)
    employee_number = st.sidebar.number_input('Employee Number', min_value=1, max_value=10000, value=1)
    environment_satisfaction = st.sidebar.selectbox('Environment Satisfaction', [1, 2, 3, 4])
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    hourly_rate = st.sidebar.number_input('Hourly Rate', min_value=0, max_value=100, value=60)
    job_involvement = st.sidebar.selectbox('Job Involvement', [1, 2, 3, 4])
    job_level = st.sidebar.number_input('Job Level', min_value=1, max_value=5, value=1)
    job_role = st.sidebar.selectbox('Job Role', ['Sales Executive', 'Research Scientist', 'Laboratory Technician', 'Manufacturing Director',
                                                 'Healthcare Representative', 'Manager', 'Sales Representative', 'Research Director', 'Human Resources'])
    job_satisfaction = st.sidebar.selectbox('Job Satisfaction', [1, 2, 3, 4])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    monthly_income = st.sidebar.number_input('Monthly Income', min_value=0, max_value=20000, value=5000)
    monthly_rate = st.sidebar.number_input('Monthly Rate', min_value=0, max_value=30000, value=10000)
    num_companies_worked = st.sidebar.number_input('Num Companies Worked', min_value=0, max_value=10, value=3)
    over18 = st.sidebar.selectbox('Over18', ['Y'])
    over_time = st.sidebar.selectbox('Over Time', ['Yes', 'No'])
    percent_salary_hike = st.sidebar.number_input('Percent Salary Hike', min_value=0, max_value=100, value=15)
    performance_rating = st.sidebar.selectbox('Performance Rating', [1, 2, 3, 4])
    relationship_satisfaction = st.sidebar.selectbox('Relationship Satisfaction', [1, 2, 3, 4])
    standard_hours = st.sidebar.number_input('Standard Hours', min_value=0, max_value=24, value=8)
    stock_option_level = st.sidebar.number_input('Stock Option Level', min_value=0, max_value=3, value=1)
    total_working_years = st.sidebar.number_input('Total Working Years', min_value=0, max_value=50, value=10)
    training_times_last_year = st.sidebar.number_input('Training Times Last Year', min_value=0, max_value=10, value=3)
    work_life_balance = st.sidebar.selectbox('Work Life Balance', [1, 2, 3, 4])
    years_at_company = st.sidebar.number_input('Years At Company', min_value=0, max_value=50, value=5)
    years_in_current_role = st.sidebar.number_input('Years In Current Role', min_value=0, max_value=50, value=5)
    years_since_last_promotion = st.sidebar.number_input('Years Since Last Promotion', min_value=0, max_value=50, value=5)
    years_with_curr_manager = st.sidebar.number_input('Years With Current Manager', min_value=0, max_value=50, value=5)

    # Create a DataFrame with user input
    data = {
        'Age': age,
        'BusinessTravel': business_travel,
        'DailyRate': daily_rate,
        'Department': department,
        'DistanceFromHome': distance_from_home,
        'Education': education,
        'EducationField': education_field,
        'EmployeeCount': employee_count,
        'EmployeeNumber': employee_number,
        'EnvironmentSatisfaction': environment_satisfaction,
        'Gender': gender,
        'HourlyRate': hourly_rate,
        'JobInvolvement': job_involvement,
        'JobLevel': job_level,
        'JobRole': job_role,
        'JobSatisfaction': job_satisfaction,
        'MaritalStatus': marital_status,
        'MonthlyIncome': monthly_income,
        'MonthlyRate': monthly_rate,
        'NumCompaniesWorked': num_companies_worked,
        'Over18': over18,
        'OverTime': over_time,
        'PercentSalaryHike': percent_salary_hike,
        'PerformanceRating': performance_rating,
        'RelationshipSatisfaction': relationship_satisfaction,
        'StandardHours': standard_hours,
        'StockOptionLevel': stock_option_level,
        'TotalWorkingYears': total_working_years,
        'TrainingTimesLastYear': training_times_last_year,
        'WorkLifeBalance': work_life_balance,
        'YearsAtCompany': years_at_company,
        'YearsInCurrentRole': years_in_current_role,
        'YearsSinceLastPromotion': years_since_last_promotion,
        'YearsWithCurrManager': years_with_curr_manager
    }

    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Apply encoders to the appropriate columns
for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

# Ensure the columns are in the same order as during training
input_df = input_df[feature_names]

# Display the input data for verification
st.subheader('User Input parameters')
st.write(input_df)

# Make prediction
prediction = model.predict(input_df)

st.subheader('Prediction')
if prediction == 1:
    st.write('<span style="color:red; font-size:24px;"><b>Attrition</b></span>', unsafe_allow_html=True)
else:
    st.write('<span style="color:green; font-size:24px;"><b>No Attrition</b></span>', unsafe_allow_html=True)