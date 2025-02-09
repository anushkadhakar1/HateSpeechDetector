import streamlit as st
from helper import *

st.title("Hate speech Detector")


st.markdown("""
This app uses a machine learning model to classify text as:
- **Hate Speech**
- **Offensive Language**
- **No hate and offensive speech**
""")
st.subheader("Enter Text for Analysis:")
test_data = st.text_area("Type or paste the text here:", placeholder="Write your text here...")


if st.button("Analyze"):
    if test_data.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        # Preprocess and predict
        user_data = cv.transform([test_data]).toarray()
        prediction = clf.predict(user_data)
        st.write(f"### Prediction: {prediction[0]}")