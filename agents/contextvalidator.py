import os
from langgraph.prebuilt import create_react_agent
from utils.utils import load_config
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict
import json
from dotenv import load_dotenv

load_dotenv(".env")

class ContextValidatorOutput(BaseModel):
    is_relevant: bool = Field(..., description="True if the chunk contains important information, otherwise False.")


# Load the config file for the agent
config = load_config('./config.yaml')

agent_state = config["agents"].get("context_validator", {}).get("state", "False")

system_prompt = config["agents"].get("context_validator", {}).get("system_prompt", "Default system prompt")

client = OpenAI(
     api_key = os.environ['LLM_API_KEY'],
     base_url = os.environ['LLM_BASE_URL']
)

def validate_context(chunk):   
    
    prompt = [""]
    prompt[0] = {
        "role": "system",
        "content": system_prompt
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", 
                   "content": f'''Now, do the task for the following chunk: {chunk}
                                  Don't deviate from the instructions otherwise you will be penalized heavily.'''
                  })
    
    
    response = client.beta.chat.completions.parse(
        model=os.environ['LLM_MODEL_NAME'],
        messages=prompt,
        temperature=0,
        response_format=ContextValidatorOutput,
        extra_body=dict(guided_decoding_backend="outlines"),
    )
    
    data = json.loads(response.choices[0].message.content)

    
    return data["is_relevant"]