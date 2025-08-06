import yaml
from openai import OpenAI

def load_config(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
