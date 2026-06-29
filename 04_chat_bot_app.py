import os

import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY not found.")
    st.stop()

client = genai.Client(api_key=API_KEY)

SYSTEM_PROMPT = """
You are a professional AI assistant.

Instructions:
- Give accurate answers.
- Use Markdown formatting.
- Format code properly.
- Use bullet points when appropriate.
- If unsure, clearly say you don't know.
"""

st.set_page_config(
    page_title="AI Chatbot",
    layout="wide"
)

st.title("AI Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:

    st.header("Settings")

    model = st.selectbox(
        "Model",
        [
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ],
    )

    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.7,
        0.1,
    )

    st.write(f"Messages: {len(st.session_state.messages)}")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

chat_text = ""

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

    chat_text += f"{message['role'].title()}: {message['content']}\n\n"

st.sidebar.download_button(
    "Download Chat",
    data=chat_text,
    file_name="chat_history.txt",
)

prompt = st.chat_input("Type your message...")

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    contents = [
        {
            "role": "user",
            "parts": [{"text": SYSTEM_PROMPT}],
        }
    ]

    for msg in st.session_state.messages:

        role = "model" if msg["role"] == "assistant" else "user"

        contents.append(
            {
                "role": role,
                "parts": [
                    {
                        "text": msg["content"],
                    }
                ],
            }
        )

    with st.chat_message("assistant"):

        placeholder = st.empty()

        full_response = ""

        try:

            stream = client.models.generate_content_stream(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                ),
            )

            for chunk in stream:

                if chunk.text:

                    full_response += chunk.text

                    placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response)

        except Exception as e:

            full_response = f"Error: {e}"

            placeholder.error(full_response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
        }
    )