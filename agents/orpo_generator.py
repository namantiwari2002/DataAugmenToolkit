import os
from utils.utils import load_config
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List
import json
from dotenv import load_dotenv
import random

load_dotenv(".env")

client = OpenAI(
     api_key = os.environ['LLM_API_KEY'],
     base_url = os.environ['LLM_BASE_URL']
)

config = load_config('./config.yaml')

agents = ["rlhf_irrelevant_content_generator","rlhf_incorrect_facts_generator","rlhf_offensive_tone_generator"]

# agent_state = config["agents"].get(agents[2], {}).get("state", "False")


def generate_orpo_data(chunk, id):
    
    prefixL = len(chunk['conversations'])-id-1 
    if prefixL < 0:
       return None
    
    system_prompt = config["agents"].get(agents[id], {}).get("prompt", "Default system prompt")
    
    if len(chunk['conversations']) == 1:
        system_prompt = config["agents"].get(agents[random.randint(0,2)], {}).get("prompt", "Default system prompt")


    prompt = [""]
    prompt[0] = ({
        "role": "system",
        "content": system_prompt
    })
    
    conversations = [{"from":"system", "value":"You are a helpful AI assistant. Please answer questions in the same language as of the question."}]

    

    for i in range(prefixL):
        conversations.append({"from":"human", "value":chunk['conversations'][i]['question']})
        conversations.append({"from":"gpt", "value":chunk['conversations'][i]['answer']})



    # Add the user's question to the prompt and display it
    prompt.append({"role": "user",
                   "content": f'''Now, do the task for the following: 
                                  Context: {chunk['context']}
                                  {conversations}
                                  Question: {chunk['conversations'][prefixL]['question']}
                                  Correct Answer: {chunk['conversations'][prefixL]['answer']}

                                  Don't deviate from the instructions otherwise you will be penalized heavily.'''
                  })

    response = client.chat.completions.create(
        model=os.environ['LLM_MODEL_NAME'],
        messages=prompt,
        temperature=0
    )
    
    conversations.append({"from":"human", "value": chunk['conversations'][prefixL]['question']})

    result = {  
                "conversations":conversations,
                "chosen": {"from":"gpt", "value": chunk['conversations'][prefixL]['answer']},
                "rejected": {"from":"gpt", "value": response.choices[0].message.content}
            }
                

    return result