import streamlit as st
import classifier as cl

st.header("Fake News Article Classifier")

st.subheader("Send in a news article to see if it is REAL or FAKE!")

title = st.text_input("Enter the title of your news article")

if st.button("Classify"):
    vectorizer, model = cl.classify()

    prediction = cl.predict(title, vectorizer, model)

    st.write(f"Your news article is {prediction}")
