from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

def get_chatgroq_model():
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set")
        return ChatGroq(api_key=api_key, model="llama3-8b-8192")
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
