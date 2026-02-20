from mem0 import Memory
import os
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()

OPEN_API_KEY=os.getenv("OPEN_API_KEY")

client = OpenAI()

config = {
    "version":"v1.1",
    "embedder":{
        "provider":"openai",
        "config":{
            "api_key":OPEN_API_KEY,
            "model":"text-embedding-3-small"
        }
    },
    "llm":{
        "provider":"openai",
        "config":{
            "api_key":OPEN_API_KEY, 
            "model":"gpt-4.1"
        }
    },
    "vector_store":{
        "provider":"qdrant",
        "config":{
            "host":"localhost",
            "port":6333
        }
    }
}

memClient = Memory.from_config(config)

#get user input
userInput = input("Please ask your question...\n")

#search the memory for user inputs:
search_results = memClient.search(query= userInput,user_id="shakshy")

#search result string
userDetails = [f"ID: {result.get("id")} Memory: {result.get("memory")}" for result in search_results.get("results")]

print(userDetails)
SYSTEM_PROMPT = f"""
You are an AI assistant, who could help with uer details by storing them.

Here is the context of the user: {json.dumps(userDetails)}
"""

#call llm
response = client.chat.completions.create(
    model ="gpt-4o",
    messages =[
        {
            "role":"user", "content":userInput
        }
    ]
)

memClient.add(
    user_id="shakshy",messages=[{"role":"user","content":userInput},{"role":"assistant","content":response.choices[0].message.content}]
)
