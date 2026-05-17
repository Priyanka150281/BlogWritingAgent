import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL")

st.set_page_config(page_title="Blog Writing Agent", page_icon="✍️")

st.title("✍️ Blog Writing Agent")

if "messages" not in st.session_state:
    st.session_state.messages = []

# show chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input box
topic = st.chat_input("Enter blog topic...")

if topic:
    # user message
    st.session_state.messages.append({"role": "user", "content": topic})

    with st.chat_message("user"):
        st.markdown(topic)

    try:
        response = requests.post(
            API_URL,
            json={"topic": topic},  
            headers={
                "accept": "application/json",
                "Content-Type": "application/json"
            }
        )

        data = response.json()

        # adjust key depending on backend response
        bot_reply = data.get("response", str(data))

    except Exception as e:
        bot_reply = f"Error: {e}"

    # assistant message
    st.session_state.messages.append(
        {"role": "assistant", "content": bot_reply}
    )

    with st.chat_message("assistant"):
        st.markdown(bot_reply)