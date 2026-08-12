import streamlit as st
import requests
import os

# Configure the Streamlit page
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="centered"
)

# Read API URL from environment, default to docker-compose service name if not set
API_URL = os.getenv("API_URL", "http://api:8000")
PREDICT_ENDPOINT = f"{API_URL}/predict"

def main():
    st.title("🐦 Twitter Sentiment Analysis")
    st.markdown("""
    Enter text below to analyze its sentiment using our fine-tuned BERT model.
    """)
    
    # Text input area
    text_input = st.text_area("Input Text", height=150, placeholder="Type a tweet or review here...")
    
    if st.button("Analyze Sentiment", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
            
        with st.spinner("Analyzing..."):
            try:
                # Make request to FastAPI backend
                response = requests.post(
                    PREDICT_ENDPOINT,
                    json={"text": text_input},
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    sentiment = result.get("sentiment", "unknown")
                    confidence = result.get("confidence", 0.0)
                    
                    # Display results with nice formatting
                    st.divider()
                    st.subheader("Results")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if sentiment.lower() == "positive":
                            st.success(f"**Sentiment:** {sentiment.capitalize()} 🟢")
                        elif sentiment.lower() == "negative":
                            st.error(f"**Sentiment:** {sentiment.capitalize()} 🔴")
                        else:
                            st.info(f"**Sentiment:** {sentiment.capitalize()} ⚪")
                            
                    with col2:
                        st.metric("Confidence Score", f"{confidence:.2%}")
                        
                else:
                    error_msg = response.json().get('detail', 'Unknown error')
                    st.error(f"Error {response.status_code}: {error_msg}")
                    
            except requests.exceptions.ConnectionError:
                st.error(f"Failed to connect to the backend API at {API_URL}. Is the service running?")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
