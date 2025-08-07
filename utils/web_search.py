# utils/web_search.py

import os
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv()

def perform_web_search(query):
    try:
        params = {
            "q": query,
            "api_key": os.getenv("SERPAPI_API_KEY"),
            "engine": "google"
        }

        search = GoogleSearch(params)
        results = search.get_dict()

        top_results = results.get("organic_results", [])
        if not top_results:
            return "No search results found."

        output = ""
        for i, result in enumerate(top_results[:3]):  # Return top 3 results
            title = result.get("title")
            link = result.get("link")
            snippet = result.get("snippet")
            output += f"🔹 **{title}**\n{snippet}\n[Link]({link})\n\n"

        return output.strip()

    except Exception as e:
        return f"❌ Web search failed: {str(e)}"
