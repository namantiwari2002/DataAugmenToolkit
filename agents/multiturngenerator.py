import os
from utils.utils import load_config
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv

load_dotenv(".env")


class QAEntry(BaseModel):
    question: str = Field(..., description="User questions")
    answer: str = Field(..., description="AI answer")

class ConversationOutput(BaseModel):
    conversation: List[QAEntry] = Field(..., description="A list of conversation questions and answers")


# Load the config file for the agent
config = load_config('./config.yaml')

agent_state = config["agents"].get("multi_turn_generator", {}).get("state", "False")

system_prompt = config["agents"].get("multi_turn_generator", {}).get("system_prompt", "Default system prompt")

client = OpenAI(
     api_key = os.environ['LLM_API_KEY'],
     base_url = os.environ['LLM_BASE_URL']
)

def generate_multi_turn_conversation(context):   
    
    prompt = [""]
    prompt[0] = {
        "role": "system",
        "content": system_prompt
    }

    # Add the user's question to the prompt and display it
    prompt.append({"role": "user", 
                   "content": f'''Now, do the task for the following.
                                  Context: {context}
                                  Don't deviate from the instructions otherwise you will be penalized heavily.'''
                  })
    
    response = client.beta.chat.completions.parse(
        model=os.environ['LLM_MODEL_NAME'],
        messages=prompt,
        temperature=0.7,
        response_format= ConversationOutput,
        extra_body=dict(guided_decoding_backend="outlines"),
    )
    
    data = json.loads(response.choices[0].message.content)

    
    return data["conversation"]