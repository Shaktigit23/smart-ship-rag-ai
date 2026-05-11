import streamlit as st
import pandas as pd
import boto3
import json
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from prepare_data import load_and_prepare_data


# Page Config
st.set_page_config(
    page_title="Smart Ship AI",
    page_icon="🚢",
    layout="wide"
)

# Title and Description
st.title("🚢 Smart Ship AI Copilot")

st.subheader("RAG + LLM Predictive Maintenance System")

# Load Data
df = load_and_prepare_data()

# Show Data
st.write("### Enhanced Vessel Dataset")
st.dataframe(df.head(10))

# Filter Issues
issue_df = df[
    (df["Severity"] == "Normal") |
    (df["Severity"] == "Warning") |
    (df["Severity"] == "High") |
    (df["Severity"] == "Critical")
]

# ============================================
# METRICS
# ============================================

critical_count = len(
    issue_df[issue_df["Severity"] == "Critical"]
)

high_count = len(
    issue_df[issue_df["Severity"] == "High"]
)

warning_count = len(
    issue_df[issue_df["Severity"] == "Warning"]
)

Normal_count = len(
    issue_df[issue_df["Severity"] == "Normal"]
)

total_issues = len(issue_df)


col1, col2, col3, col4 = st.columns(4)

col1.metric("Critical Issues", critical_count)

col2.metric("High Issues", high_count)

col3.metric("Warning Issues", warning_count)

col4.metric("Normal Issues", Normal_count)

# Issue Table
st.write("## Detected Vessel Issues")

st.dataframe(issue_df.head(10))

# Load Bedrock Embeddings.
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)

# Load FAISS Vector Database
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# ============================================
# USER QUESTION
# ============================================

question = st.text_input(
    "Ask AI About Vessel Issues"
)

# AI Question Processing

if question:
    with st.spinner("Analyzing vessel telemetry..."):
        # Retrive Relevent Chunks
        docs = vectorstore.similarity_search(question, k=5)
        context = "\n".join([doc.page_content for doc in docs])
        st.write("### Retrieved Context")
        
# Create Prompt
        prompt = f"""
        You are a Expert Marine Vessel AI Assistant.
        Your responsibilities:
        - Root Cause Analysis
        - Defect Analysis
        - Predictive Maintenance
        - Troubleshooting Guidance
        - Remediation Recommendations
        
        Important:
        Use ONLY the provided vessel context.

        If no answer exists, say:
        'No relevant vessel issue found.'

        Return response ONLY in valid JSON format.

        JSON format:

        {{
            "Vessel_ID": "",
            "Defect_ID": "",
            "Root_Cause": "",
            "Severity": "",
            "Troubleshooting_Steps": "",
            "Remediation_Actions": ""
        }}

        
        Context:
        {context}

        Question:
        {question}
        """
        st.write("### Generated Prompt")
        
# Connect Bedrock
        client = boto3.client('bedrock-runtime', region_name='us-east-1')
        
        # Nova Request Body
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt
                        }
                    ]
                }
            ],
            "inferenceConfig": {
                "max_new_tokens": 300
            }
        }
        
# Invoke Model
        response = client.invoke_model(
            modelId="amazon.nova-lite-v1:0",
            body=json.dumps(body)
        )
        #  Parse Response
        response_body = json.load(
            response["body"]
            
        )
        ai_answer = response_body["output"]["message"]["content"][0]["text"]
        
        # Show AI Answer
               
        st.write("### AI RCA Analysis")
        
        st.success(ai_answer)

# ============================================

            