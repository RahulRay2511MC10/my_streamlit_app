from langchain_openai import OpenAI
from dotenv import load_dotenv
import streamlit as st;
st.header("Response Tool")
user_input=st.text_input("Enter your question")
load_dotenv()
llm=OpenAI(model="gpt-3.5-turbo-instruct")
if st.button('Get answer'):
    with st.spinner("Getting responses.😴😴..."):
        result=llm.invoke(user_input)

    
    st.write(result)
st.markdown(
    """
    <hr>
    <p style="text-align:center; color:#666;">
        🚀 Developed by <b>Rahul</b>
    </p>
    """,
    unsafe_allow_html=True
)




