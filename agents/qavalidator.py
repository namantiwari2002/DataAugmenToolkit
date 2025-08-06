import os
from langgraph.prebuilt import create_react_agent
from utils.utils import load_config
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict
import json
from dotenv import load_dotenv

load_dotenv(".env")

# 🔹 4. QAValidator Agent Output Schema
class QAValidatorOutput(BaseModel):
    is_valid: bool = Field(..., description="True if the question is relevant, the answer is correct, and it aligns with the passage.")

# Load the config file for the agent
config = load_config('./config.yaml')

agent_state = config["agents"].get("qa_validator", {}).get("state", "False")

system_prompt = config["agents"].get("qa_validator", {}).get("system_prompt", "Default system prompt")

client = OpenAI(
     api_key = os.environ['LLM_API_KEY'],
     base_url = os.environ['LLM_BASE_URL']
)

def validate_qa(question, answer, chunk):   
    
    prompt = [""]
    prompt[0] = {
        "role": "system",
        "content": system_prompt
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", 
                   "content": f'''Now, do the task for the following chunk: 
                                  Question: {question}
                                  Answer: {answer}
                                  Chunk: {chunk}
                                  Don't deviate from the instructions otherwise you will be penalized heavily.'''
                  })
    
    
    response = client.beta.chat.completions.parse(
        model=os.environ['LLM_MODEL_NAME'],
        messages=prompt,
        temperature=0,
        response_format=QAValidatorOutput,
        extra_body=dict(guided_decoding_backend="outlines"),
    )
    
    data = json.loads(response.choices[0].message.content)

    
    return data["is_valid"]