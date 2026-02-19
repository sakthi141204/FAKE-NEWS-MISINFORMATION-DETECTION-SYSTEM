import webbrowser
import threading
import streamlit as st

# Auto open browser
def open_browser():
    webbrowser.open_new("http://localhost:8501")

threading.Timer(1, open_browser).start()

st.title("Fake News Detection App")
st.write("Your Streamlit app is running successfully!")
