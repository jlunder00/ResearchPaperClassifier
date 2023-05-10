import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

@st.cache_data
def load_data():
    with open("gans/data/with_titles_and_abstract.json", 'r') as f:
        data = json.load(f)
        df = pd.DataFrame(data)
        columns_to_drop = [col for col in df.columns if col not in ['title', 'abstract']]
        df = df.drop(columns=columns_to_drop, axis=1)
        df = df.sample(frac=0.01)
        return st.dataframe(df)


st.title("Research Paper Classification")
st.heading("The Data")
data = load_data()

