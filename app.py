import streamlit as st
import pandas as pd
import pickle

# ================================

# LOAD FILES

# ================================

model = pickle.load(open('model.pkl', 'rb'))
encoders = pickle.load(open('encoders.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.set_page_config(page_title="Customer Travel Prediction")

st.title("✈️ Customer Travel Prediction App")

input_data = {}

# ================================

# INPUTS

# ================================

for col in columns:
    if col in encoders:
        # categorical
        encoder = encoders[col]
        options = list(encoder.classes_)
        selected = st.selectbox(f"Select {col}", options)
        input_data[col] = encoder.transform([selected])[0]
    else:
        # numeric
        value = st.number_input(f"Enter {col}", value=0)
        input_data[col] = value

# ================================

# PREDICTION

# ================================

if st.button("Predict"):
    try:
        input_df = pd.DataFrame([input_data])
        input_df = input_df[columns]
        prediction = model.predict(input_df)[0]
        if prediction == 1:
            st.success("✅ Customer is likely to Travel")
        else:
            st.error("❌ Customer is NOT likely to Travel")
    except Exception as e:
        st.error(f"Error: {e}")
