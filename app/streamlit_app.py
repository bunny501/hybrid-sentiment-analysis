
import streamlit as st

st.title("Hybrid Sentiment Analysis Framework")
review = st.text_area("Enter a review")

if st.button("Predict"):
    st.write("Connect trained model here.")
