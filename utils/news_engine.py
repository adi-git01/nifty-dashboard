
from duckduckgo_search import DDGS
import streamlit as st

@st.cache_data(ttl=3600)
def fetch_latest_news(query_term, max_results=5):
    """
    Searches DuckDuckGo for the top news related to the query.
    Returns a list of dictionaries with title, link, date, and source/snippet.
    """
    results = []
    try:
        query = f"{query_term} share news India"
        with DDGS() as ddgs:
            # ddgs.news returns a generator
            news_gen = ddgs.news(query, region="in-en", safesearch="off", max_results=max_results)
            for r in news_gen:
                results.append({
                    "title": r.get("title"),
                    "link": r.get("url"),
                    "date": r.get("date"),
                    "source": r.get("source"),
                    "snippet": r.get("body")
                })
    except Exception as e:
        print(f"Error fetching news for {query_term}: {e}")
        # Return a dummy error item so UI shows something
        results.append({
            "title": "Could not fetch specific news at this moment.",
            "link": "#",
            "date": "",
            "source": "System",
            "snippet": f"Error detail: {str(e)}"
        })
        
    return results
