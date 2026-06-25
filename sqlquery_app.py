import streamlit as st
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

GOOGLE_API_KEY = "GOOGLE API KEY"

MODEL = "gemini-2.5-flash"

genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(MODEL)

st.set_page_config(page_title="GenAI SQL Helper", page_icon="🧠", layout="centered")

st.markdown(
    """
    <div style="text-align:center; padding: 1rem 0;">
        <h1>GenAI SQL Helper</h1>
        <p>Generate a SQL query from plain English with safer error handling.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

text_input = st.text_area("Enter your request in plain English", height=160)
submit_button = st.button("Generate SQL")

template = """
Create a SQL query using the text below.
Return the result in this format:

SQL Query:
<the SQL query>

Explanation:
<plain-English explanation>

Example:
<one example input or output usage>

Text:
{text_input}
"""

if submit_button:
    if not text_input.strip():
        st.warning("Please enter a request before generating SQL.")
    else:
        with st.spinner("Generating..."):
            try:
                response = model.generate_content(template.format(text_input=text_input.strip()))
                result_text = getattr(response, "text", "").strip()

                if not result_text:
                    st.error("The model returned an empty response.")
                else:
                    st.success("SQL generated successfully")
                    st.markdown(result_text)
            except GoogleAPIError as exc:
                st.error(f"Google API error: {exc}")
            except Exception as exc:
                st.error(f"Unexpected error: {exc}")


