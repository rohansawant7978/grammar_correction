import streamlit as st
from inference import function1,function2
st.title('Grammar Error Correction')
st.header('Using Attention Mechanism')
st.markdown('created by: **Rohan Sawant**')

st.text_area("Enter Text", '')
st.button('Check Your Text')
