from annotated_types import doc
import boto3
import json
import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_aws import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from prepare_data import load_and_prepare_data

# Step 1 - Call the function to load and prepare the data
df = load_and_prepare_data()
print(df.head())

df = df[
    (df["Severity"] == "Warning") |
    (df["Severity"] == "High") |
    (df["Severity"] == "Critical")
]

# Step 2 - Preprocess the data 
# (e.g., handle missing values, encode categorical variables, etc.)
documents = []
for _, row in df.iterrows():
    sentence = f"""
    At time {row['Time_Stamp']},

    vessel was operating in {row['Operational_Mode']} mode.

    Engine temperature was {row['Temperature (°C)']} degree Celsius.

    Engine RPM was {row['RPM']}.

    Fuel efficiency was {row['Fuel_Efficiency']}.

    Vibration X was {row['Vibration_X']}.

    Vibration Y was {row['Vibration_Y']}.

    Vibration Z was {row['Vibration_Z']}.

    Engine torque was {row['Torque']}.

    Power output was {row['Power_Output (kW)']} kilowatts.

    Fault condition status was {row['Fault_Condition']}.
    
    Vessel type was {row['Vessel_Type']}.
    
    Vessel ID was {row['Vessel_ID']}.
    
    Engine model was {row['Engine_Model']}.
    
    Expected temperature was {row['Expected_Temp']} degree Celsius. 
    
    Defect ID associated with this issue was {row['Defect_ID']}.

    Root cause identified was {row['Root_Cause']}.

    Severity level of the issue was {row['Severity']}.

    Recommended troubleshooting steps were:
    {row['Troubleshooting_Steps']}.

    Recommended remediation actions were:
    {row['Remediation_Actions']}.
    
    """

    documents.append(sentence)
    
print("Documents prepared successfully")
# Convert all rows into one text block
print(f"Total Documents: {len(documents)}")

text = "\n".join(documents)
print("\n========GENERATED TEXT========\n")
 
# Convert the DataFrame to a text data.
# text = df.to_string()

# Step 3 - Chunking step
# Create an instance of the RecursiveCharacterTextSplitter class
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

# Split the text into chunks
chunks = text_splitter.split_text(text)

print("Chunks created successfully")

print(f"Total Chunks: {len(chunks)}")
# print(chunks[0])

 
 # Step 4 - Create an instance of the BedrockEmbeddings class used Titan Embeddings. 
embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    region_name="us-east-1"
)
# Generate embeddings for the chunks of text
# vector = embeddings.embed_query("Hello")


# Step 5 - Create a FAISS vector store from the chunks and embeddings
vectorstore = FAISS.from_texts(chunks, embedding = embeddings)
vectorstore.save_local("faiss_index")
print("FAISS vector store database created and saved successfully.")


# Step 6 - User question 
query = input("\nAsk your question:")

# Step 7 - Retrieve relevant Chunks
docs = vectorstore.similarity_search(query, k=5)

context = "\n".join([doc.page_content for doc in docs])

print("\n=======Retrived Context========\n")
print("Successfully Retrived Relevent Context.")



#Step 8 - Create Better Prompt 
prompt = f"""
You are a Expert Marine Vessel AI Assistant.

Your responsibilities:
- identify vessel anomalies
- analyze engine failures
- explain root causes
- suggest corrective actions
- analyze operational issues

IMPORTANT:
Use ONLY the provided context.
If information is unavailable, say:
'No relevant vessel data found.'

Context:
{context}

Give:
- Defect_ID
- main issue
- Troubleshooting_Steps
- severity
- recommendation

Maximum 5 lines.

Question:
{query}
"""
print("\n=======Generated Prompt========\n")

print("Successfully Generated the Prompt for Nova LLM.")


# Step 9 - Connect Nova LLM 
client = boto3.client(
    "bedrock-runtime", 
    region_name = "us-east-1"
)    

# Step 10 - Create the Nova request body.
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


# Step 11 - Invoke Nova Model
response = client.invoke_model(
    modelId = "amazon.nova-lite-v1:0",
    body = json.dumps(body)
     
)



# Step 12 - Print the response
response_body = json.loads(response["body"].read())

answer = response_body["output"]["message"]["content"][0]["text"]



# Step 13 - Print the Final answer
print("\n======AI Answer=======\n")
print("Successfully Generated the answer from Nova LLM.")


# Step 14 - Filter only issues based on severity level.
issue_df = df[df["Severity"] != "Normal"]


# Step 15 - Create a new table with only shows Issues table.
issue_table = issue_df[
    [       
            "Time_Stamp",
            "Temperature (°C)",
            "RPM",
            "Fault_Condition",
            "Vessel_ID",
            "Vessel_Type",
            "Engine_Model",
            "Expected_Temp",
            "Defect_ID",
            "Root_Cause",
            "Troubleshooting_Steps",
            "Severity",
            "Remediation_Actions"
    ]
]

print("\n========== ISSUE TABLE ==========\n")

print(issue_table.head())
print('Successfully Created the Issues table.')






