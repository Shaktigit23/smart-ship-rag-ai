# Requirements - 
""" A RAG+LLM application that consumes high-frequency telemetry (examples given: five-minute intervals, suggested sample 15 days–one month) 
and produces defect records containing defect ID, root cause, troubleshooting steps, and remediation actions (example: engine temp expected ~120°C, observed 180°C → infer thermostat 
failure, recommend replacement). The RAG component must provide vessel-specific context to the LLM; vector database choice is flexible. The developer confirmed willingness 
to implement the prototype and demonstrate it. Unresolved items include exact sample dataset selection and final vector DB choice; implementation specifics (production deployment, 
scalability, and validation metrics) were not defined.""" 


# Smart Ship AI – RAG + LLM Application
# Author - Shaktiprasad Rudrawar

# AI Engineer Project – Smart Ship AI

## Overview

Smart Ship AI is an industrial AI application built using:

- Amazon Bedrock
- Titan Embeddings
- FAISS Vector Database
- Retrieval-Augmented Generation (RAG)
- Amazon Nova / Claude LLM

The system analyzes vessel operational data and provides:

- Root Cause Analysis (RCA)
- Fault Detection
- Severity Analysis
- Troubleshooting Guidance
- Remediation Recommendations

---

# Architecture

Raw CSV Data
↓
Data Engineering (Pandas)
↓
Issue Filtering
↓
Text Chunking
↓
Titan Embeddings
↓
FAISS Vector Database
↓
Semantic Retrieval
↓
LLM Reasoning
↓
AI Generated Answers

---

# Technologies Used

- Python
- Pandas
- LangChain
- Amazon Bedrock
- Titan Embeddings
- FAISS
- boto3

---

# Features

- Vessel anomaly detection
- Predictive maintenance support
- Root cause analysis
- Semantic search
- AI-powered troubleshooting
- Retrieval-Augmented Generation (RAG)

---

# Project Structure

project/

├── app.py  
├── prepare_data.py  
├── sample.csv  
├── enhanced_ship_data.csv  
├── requirements.txt  
└── README.md  

---

# Installation

## Create Virtual Environment

```bash
python -m venv .venv
```

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / Mac

```bash
source .venv/bin/activate
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# AWS Configuration

Configure AWS CLI:

```bash
aws configure
```

Required:
- AWS Access Key
- AWS Secret Key
- Region: us-east-1

---

# Run Application

```bash
python app.py
```

---

# Example Questions

- What is the root cause of overheating?
- Which records contain critical severity?
- What remediation actions were recommended?
- Which operational mode caused abnormal vibration?

---

# Future Improvements

- Streamlit dashboard
- Real-time sensor ingestion
- Predictive maintenance ML models
- LangGraph agents
- Cloud deployment
- Multi-vessel monitoring

---

