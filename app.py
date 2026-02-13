# app.py

import os
import requests
import streamlit as st

st.set_page_config(page_title="MiniTune", page_icon="🧪")

HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
MODEL_ID = "arpitamishra27/minitune-merged-model"


API_URL = f"https://router.huggingface.co/hf-inference/models/{MODEL_ID}"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}


def query_model(question):
    payload = {
        "inputs": f"Question: {question}\nAnswer:",
        "parameters": {
            "max_new_tokens": 100
        }
    }

    response = requests.post(
        API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    # DEBUG INFO
    return f"""
Status Code: {response.status_code}

Response Headers:
{response.headers}

Raw Response Text:
{response.text}
"""
