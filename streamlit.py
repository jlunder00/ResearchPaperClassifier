import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

def load_data():
    data = pd.read_json("gans/data/data_subset.json")
    return st.dataframe(data)


st.title("Research Paper Classification")

st.header("The Data")
data = load_data()


st.header("Discriminator Model")

message = st.text_area("Input a made up or real abstract for a research paper", height=100)
prediction = requests.post("http://73.254.3.61:5002/predict", data=message)
st.title('Real!' if prediction.json()['prediction'] == 0 else 'Not Real!')
