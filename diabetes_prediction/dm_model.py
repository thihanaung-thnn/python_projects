import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pickle

rf = pickle.load(open('diabetes_prediction/rf_model.pkl', 'rb'))


def user_inputs():
    age = st.number_input('Age', max_value=100)

    gender = st.radio('Gender', ('Male', 'Female'))

    # calculate bmi
    st.caption('Enter your height')
    col1, col2 = st.columns(2)
    foot = col1.number_input('Feet', max_value=8, min_value=1)
    inches = col2.number_input('inches', min_value=0, max_value=11)
    lbs = st.number_input('Enter your weights in pound', step=5, min_value=10)
    height_in = foot*12 + inches
    bmi = (703*lbs)/height_in**2

    physically_active = st.radio(
        'ကိုယ်လက်လှုပ်ရှား လေ့ကျင့်ခန်း တရက် ဘယ်နနာရီ လုပ်ဖြစ်ပါသလဲ။',
        ('one hour or more', 'more than half an hour',
         'less than half an hour', 'none')
    )

    junk_food = st.radio(
        'အာဟာရနည်းပြီး ကိုလက်စရောများသော မုန့်ပဲသရေစာများ (Junk Food)စားပါသလား။ ',
        ('occasionally', 'often', 'very often', 'always')
    )

    stress = st.radio(
        'စိတ်ဖိစီးမှုများ တတ်ပါသလား။',
        ('not at all', 'sometimes', 'very often', 'always')
    )

    uriation_freq = st.radio(
        'ဆီးသွားတာ ပုံမှန်လား။ ပုံမှန်ထက် ပိုများပါသလား။ ',
        ('not much', 'quite often')
    )

    bp_level = st.radio(
        'သွေးပေါင်ချိန် ပုံမှန်ပဲလား။',
        ('low', 'normal', 'high')
    )

    highBP = st.radio(
        'သွေးပေါင်ချိန် အများကြီး တက်ဖူးတာမျိုး ရှိပါသလား။',
        ('yes', 'no')
    )

    family_diabetes = st.radio(
        'မိသားစုတွင် ဆီးချိုရှိသူ ရှိပါသလား။ ', ('yes', 'no'))

    regular_medicine = st.radio(
        'ပုံမှန်သောက်ရသော ဆေးများရှိပါသလား။ ', ('yes', 'no'))

    alcohol = st.radio(
        'အရက်ကို စွဲစွဲမြဲမြဲသောက်ပါသလား။', ('yes', 'no')
    )

    smoking = st.radio(
        'ဆေးလိပ်ကို စွဲစွဲမြဲမြဲ  သောက်ပါသလား။ ', ('yes', 'no')
    )

    sleep = st.number_input('ညတွင် ဘယ်နှနာရီ အိပ်ပျော်ပါသလဲ။', max_value=24)
    soundsleep = st.number_input(
        'ဘယ်နနာရီ နှစ်နှစ်ခြိုက်ခြိုက်အိပ်ပျော်ပါသလဲ။ ', max_value=24)

    pregancies = st.number_input(
        'အမျိုးသမီးဖြစ်က ဘယ်နယောက်ကိုယ်ဝန်ဆောင်ဖူးပါသလဲ။', min_value=0, step=1)
    pdiabetes = 0

    data_dict = {
        'bmi': [bmi], 'sleep': [sleep], 'soundsleep': [soundsleep], 'pregancies': [pregancies], 'family_diabetes': [family_diabetes],
        'smoking': [smoking], 'gender': [gender], 'bp_level': [bp_level], 'stress': [stress], 'physically_active': [physically_active],
        'uriation_freq': [uriation_freq], 'age': [age], 'alcohol': [alcohol], 'regular_medicine': [regular_medicine], 'junk_food': [junk_food],
        'pdiabetes': [pdiabetes], 'highBP': [highBP]
    }
    df = pd.DataFrame(data_dict)

    def age_cat(x):
        if x < 40:
            return 0
        elif x >= 40 and x < 50:
            return 1
        elif x >= 50 and x < 60:
            return 2
        else:
            return 3
    df['age'] = df['age'].apply(age_cat)
    category_mapping = {
        'age': {'less than 40': 0, '40-49': 1, '50-59': 2, '60 or older': 3},
        'family_diabetes': {'no': 0, 'yes': 1},
        'gender': {'Female': 0, 'Male': 1},
        'smoking': {'no': 0, 'yes': 1},
        'pdiabetes': {'no': 0, 'yes': 1},
        'regular_medicine': {'no': 0, 'yes': 1},
        'physically_active': {'one hour or more': 0, 'more than half an hour': 1, 'less than half an hour': 2, 'none': 3},
        'junk_food': {'occasionally': 0, 'often': 1, 'very often': 2, 'always': 3},
        'bp_level': {'low': 0, 'normal': 1, 'high': 2},
        'highBP': {'no': 0, 'yes': 1},
        'alcohol': {'no': 0, 'yes': 1},
        'uriation_freq': {'not much': 0, 'quite often': 1},
        'stress': {'not at all': 0, 'sometimes': 1, 'very often': 2, 'always': 3}}
    df = df.replace(category_mapping)
    return df


st.title('Diabetes Prediction')
st.markdown("""
Prediction model used here is **Random Forest Classifier** with accuracy score - 97% with recall score - 90 % and roc-auc score - 99.7%. 
You can check the complete analysis [here](https://nbviewer.org/github/thihanaung-thnn/python_projects/blob/main/diabetes_prediction/pre_diabetes.ipynb).

You can get the original dataset [here](https://www.kaggle.com/tigganeha4/diabetes-dataset-2019)
""")
data = user_inputs()
result = rf.predict(data)
if st.button('CHECK'):
    if result == 0:
        st.title('ဝမ်းသာပါတယ်။ သင့်မှာ ဆီးချိုလက္ခဏာများ မရှိပါ။')
    if result == 1:
        st.title('ဆီးချိုဖြစ်နိုင်သည်။ နီးစပ်ရာတွင် သွေးဖောက်စစ်ရန် လိုအပ်သည်။ ')
