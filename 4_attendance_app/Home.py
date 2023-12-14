import streamlit as st
# import face_rec

st.set_page_config(page_title='Attendance System',layout='wide')
st.header("Attendance System using Facial Recognition")

with st.spinner("Loading Models and Connecting to Redis"):
    import face_rec

st.success("Model loaded successfully!!")
st.success("Database connected successfully!!")
