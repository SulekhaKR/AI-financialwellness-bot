from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import os
import sys
import pandas as pd
from PyPDF2 import PdfReader
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from utils.web_search import perform_web_search  # <-- SerpApi

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from models.llm import get_chatgroq_model

# --------------------- System Prompt ---------------------
def build_system_prompt(mode):
    if mode == "Concise":
        return """You are a helpful financial assistant.

Answer the user's personal finance questions briefly and clearly. Give only the most important insights."""
    else:
        return """You are a detailed and thoughtful financial wellness advisor.

Provide in-depth answers, with step-by-step reasoning, relevant examples, and actionable advice tailored to the user's situation."""

# --------------------- Get Chat Response ---------------------
def get_chat_response(chat_model, messages, system_prompt):
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            if msg["role"] == "user":
                formatted_messages.append(HumanMessage(content=msg["content"]))
            else:
                formatted_messages.append(AIMessage(content=msg["content"]))
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

# --------------------- Instructions Page ---------------------
def instructions_page():
    st.title("The Chatbot Blueprint")
    st.markdown("Welcome! Follow these instructions to set up and use the chatbot.")

    st.markdown("""
## 🤖 Welcome to the AI Financial Chatbot!

This assistant helps you analyze your financial documents such as:

- 💳 Credit card statements
- 📄 Income tax summaries
- 💼 Salary slips

### 📂 How to Use:
1. **Go to the "Chat" tab**
2. **Upload** a `.pdf` or `.csv` file with your financial data
3. **Ask questions** like:
   - "Summarize this credit card statement"
   - "What are my top expenses?"
   - "Detect any duplicate transactions"
4. Use the **Concise / Detailed** toggle to control how much explanation you want.

### 🌐 Need real-time info?
Just type:
> current gold price  
> latest income tax rules  
> search: best credit card offers

This triggers a **live web search** powered by SerpApi 🌐

---

Feel free to explore — the bot understands natural language and financial terms.
""")

# --------------------- Chat Page ---------------------
def chat_page():
    st.title("🤖 AI Financial Chatbot")

    chat_model = get_chatgroq_model()
    uploaded_file = st.file_uploader("Upload your financial document (CSV or PDF)", type=["csv", "pdf"])
    uploaded_text = ""

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            uploaded_text = df.to_string(index=False)
            st.success("✅ CSV uploaded.")
            st.dataframe(df)
        elif uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            if reader.is_encrypted:
                password = st.text_input("🔒 This PDF is encrypted. Enter password to proceed:", type="password")
                if password:
                    try:
                        result = reader.decrypt(password)
                        if result == 0:
                            st.error("❌ Incorrect password. Please try again.")
                            return
                    except Exception as e:
                        st.error(f"❌ Decryption failed: {e}")
                        return
                else:
                    st.warning("Please enter the password to read the PDF.")
                    return

            try:
                num_pages = len(reader.pages)
                if num_pages == 0:
                    st.warning("The PDF appears to be empty or not readable.")
                    return

                extracted_pages = []
                for i in range(num_pages):
                    text = ""
                    try:
                        text = reader.pages[i].extract_text()
                    except Exception as e:
                        st.caption(f"⚠️ Page {i + 1} could not be read — it may not exist or is corrupted.")
                        continue  # Skip to next page
                    if text:
                        extracted_pages.append(text)
                    else:
                        st.caption(f"ℹ️ Page {i + 1} had no extractable text.")

                if not extracted_pages:
                    st.caption("No readable text found in the PDF.")
                    return

                uploaded_text = "\n".join(extracted_pages)
                st.success("✅ PDF uploaded and text extracted successfully.")
            except Exception as e:
                st.error(f"❌ Failed to process PDF: {e}")
                return

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Type your message here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Getting response..."):
                combined_prompt = build_system_prompt(st.session_state.mode)
                if uploaded_text:
                    combined_prompt += f"\n\nUser's financial data:\n{uploaded_text[:3000]}"
                response = get_chat_response(chat_model, st.session_state.messages, combined_prompt)

                if not response or "I don't know" in response.lower():
                    st.warning("🤖 Local model unsure. Searching the web...")
                    response = perform_web_search(prompt)

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

# --------------------- Main App ---------------------
def main():
    st.set_page_config(
        page_title="LangChain Multi-Provider ChatBot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    with st.sidebar:
        st.title("Navigation")
        page = st.radio("Go to:", ["Chat", "Instructions"], index=0)
        st.divider()
        st.session_state.mode = st.radio("Select Response Mode:", ["Concise", "Detailed"], index=0)
        if page == "Chat":
            st.divider()
            if st.button("🗑️ Clear Chat History", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

    if page == "Instructions":
        instructions_page()
    elif page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
