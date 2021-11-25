import streamlit as st
from inference import function1,function2
st.title('Grammar Error Correction')
st.header('Using Attention Mechanism')
st.markdown('created by: **Rohan Sawant**')

inp = st.text_area("Enter Text", '')
submit = st.button('Check Your Text')

if submit:
    actual = function1(inp)
    st.text(f'last refreshed on {datetime.now(timezone("Asia/Kolkata")).strftime("%Y-%m-%d %I:%M:%S %p")}')
