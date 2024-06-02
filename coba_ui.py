import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.under_sampling import RandomUnderSampler

# Import data
data = pd.read_csv('hr_data.csv',
                   names=['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome', 'Education', 'EducationField', 'EmployeeCount', 'EmployeeNumber', 'EnvironmentSatisfaction', 'Gender', 'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole', 'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'Over18', 'OverTime', 'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'])

# Buat DataFrame dari data yang sudah diformat
df_formatted = data



numerical_cols = [
    'Age', 'DailyRate', 'DistanceFromHome', 'Education', 'EmployeeCount', 
    'EmployeeNumber', 'EnvironmentSatisfaction', 'HourlyRate', 'JobInvolvement', 
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 
    'PercentSalaryHike', 'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours', 
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance', 
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

for col in numerical_cols:
    df_formatted[col] = pd.to_numeric(df_formatted[col], errors='coerce')

# Drop rows with any NaN values (in case there were non-numeric entries)
df_formatted = df_formatted.dropna()

# Label Encoding
encoder = LabelEncoder()
df_formatted['Attrition'] = encoder.fit_transform(df_formatted['Attrition'])
df_formatted['BusinessTravel'] = encoder.fit_transform(df_formatted['BusinessTravel'])
df_formatted['Department'] = encoder.fit_transform(df_formatted['Department'])
df_formatted['EducationField'] = encoder.fit_transform(df_formatted['EducationField'])
df_formatted['Gender'] = encoder.fit_transform(df_formatted['Gender'])
df_formatted['JobRole'] = encoder.fit_transform(df_formatted['JobRole'])
df_formatted['MaritalStatus'] = encoder.fit_transform(df_formatted['MaritalStatus'])
df_formatted['Over18'] = encoder.fit_transform(df_formatted['Over18'])
df_formatted['OverTime'] = encoder.fit_transform(df_formatted['OverTime'])



# Menghitung korelasi dengan 'Attrition'
correlation_with_attrition = df_formatted.corrwith(df_formatted['Attrition'], method='spearman').abs()
correlation_with_attrition = correlation_with_attrition.sort_values(ascending=False)

# Undersample the majority class to achieve a 2:1 ratio


# Membuat DataFrame berdasarkan fitur yang paling berkorelasi
top_5_features = df_formatted[correlation_with_attrition.head(6).index]
top_10_features = df_formatted[correlation_with_attrition.head(11).index]
top_15_features = df_formatted[correlation_with_attrition.head(16).index]



# Input untuk memilih jumlah fitur
num_features = st.selectbox('Pilih jumlah fitur', [5, 10, 15, 27])

# Memilih dataset berdasarkan jumlah fitur
if num_features == 5:
    data = top_5_features
if num_features == 10:
    data = top_10_features
if num_features == 15:
    data = top_15_features    
if num_features == 27:
    data = df_formatted

# Input untuk memilih hyperparameter
n_estimators = st.selectbox('Pilih n_estimators', [10, 50, 100])
max_depth = st.selectbox('Pilih max_depth', [1, 5, 10])
max_features = st.selectbox('Pilih max_features', [2, 5, 10])
size_test = st.selectbox('Pilih size test', [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
jumlah_data = st.selectbox('Pilih jumlah data', [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300])
undersample_data = st.selectbox('Apakah ingin undersample data?', ['Yes', 'No'])

if jumlah_data == 100:
    data = data.head(100)
if jumlah_data == 200:
    data = data.head(200)
if jumlah_data == 300:
    data = data.head(300)
if jumlah_data == 400:
    data = data.head(400)
if jumlah_data == 500:
    data = data.head(500)
if jumlah_data == 600:
    data = data.head(600)
if jumlah_data == 700:
    data = data.head(700)
if jumlah_data == 800:
    data = data.head(800)
if jumlah_data == 900:
    data = data.head(900)
if jumlah_data == 1000:
    data = data.head(1000)
if jumlah_data == 1100:
    data = data.head(1100)
if jumlah_data == 1200: 
    data = data.head(1200)
if jumlah_data == 1300:
    data = data.head(1300)



trained_model = False
model_hasil = RandomForestClassifier()

# Tombol untuk melakukan training ulang
if st.button('Train Model'):
    # Memisahkan fitur dan target

    if undersample_data == 'Yes':
        rus = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
        X_res, y_res = rus.fit_resample(data.drop(columns=['Attrition']), data['Attrition'])
        data = pd.concat([X_res, y_res], axis=1)
    else:
        data = data
        
    X = data.drop(columns=['Attrition'])
    y = data['Attrition']

    # Membagi dataset menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size_test, random_state=42)

    # Membuat dan melatih model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=max_features, random_state=42)
    model.fit(X_train, y_train)
    model_hasil = model


    # Memprediksi dan menghitung akurasi
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Menampilkan hasil akurasi
    st.write(f'Akurasi model: {accuracy * 100:.2f}%')

    # Membuat confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Menampilkan confusion matrix menggunakan seaborn
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title('Confusion Matrix')

    # Menampilkan grafik di Streamlit
    st.pyplot(fig)
    trained_model = True


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


# Apply encoders to the appropriate columns
for column, encoder in encoders.items():
    input_df[column] = encoder.transform(input_df[column])

top_5_features = top_5_features.drop(columns=['Attrition'])
top_10_features = top_10_features.drop(columns=['Attrition'])
top_15_features = top_15_features.drop(columns=['Attrition'])
datainput = df_formatted.drop(columns=['Attrition'])

if num_features == 5:
    input_df = input_df[top_5_features.columns]
if num_features == 10:
    input_df = input_df[top_10_features.columns]
if num_features == 15:
    input_df = input_df[top_15_features.columns]
if num_features == 27:
    input_df = input_df[datainput.columns]




# Display the input data for verification
st.subheader('User Input parameters')
st.write(input_df)

if trained_model == False:
    st.write('Please train the model first')
    st.stop()
else:
    # Make prediction
    prediction = model_hasil.predict(input_df)

    st.subheader('Prediction')
    if prediction == 1:
        st.write('<span style="color:red; font-size:24px;"><b>Attrition</b></span>', unsafe_allow_html=True)
    else:
        st.write('<span style="color:green; font-size:24px;"><b>No Attrition</b></span>', unsafe_allow_html=True)


