# dashboard/components/ml_client.py
import requests
import streamlit as st

def trigger_ml_processing(data_id):
    """Trigger ML processing in Container Apps"""
    try:
        # In Container Apps, services communicate by name
        response = requests.post(
            'http://ml-model:5000/process',
            json={'data_id': data_id},
            timeout=60  # Account for cold starts
        )
        return response.json()
    except requests.Timeout:
        st.warning("Processing is taking longer than expected...")
        # Implement polling logic here