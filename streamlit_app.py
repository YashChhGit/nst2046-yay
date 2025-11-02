import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get API key and verify it's loaded
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

st.title("Bomboclatt App")

# Check if API key is loaded
if not GOOGLE_API_KEY:
    st.error("⚠️ API key not found! Please set GOOGLE_API_KEY in your .env file or environment variables.")
    st.stop()

def generate_response(prompt):
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)
        result = llm.invoke(prompt)
        return result.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "Mi bomba",
    )
    uploaded_file = st.file_uploader("Choose a file")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        if text:
            with st.spinner("Generating response..."):
                response = generate_response(text)
                st.write("Response:")
                st.write(response)
        else:
            st.warning("Please enter some text.")