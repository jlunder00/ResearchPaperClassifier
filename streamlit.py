import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

@st.cache_data
def load_data():
    with open("gans/data/with_titles_and_abstract.json", 'r') as f:
        data = pd.read_json(f)
        return st.dataframe(data)


st.title("Research Paper Classification")
st.header("The Data")
data = load_data()

