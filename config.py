from pathlib import Path
from pydantic import BaseModel, Field, validator

class Config(BaseModel):
    """Runtime options – filled by CLI flags or Streamlit form."""
    # LLM / API
    model_name: str = Field(..., description="HF model or local alias")
    api_key: str
    base_url: str = "http://localhost:8001/v1"

    # generation mode
    mode: str = Field(..., description="single-sft | multi-sft | single-align | multi-align")

    # files / paths
    input_file: Path
    output_dir: Path = Path("output")
    max_workers: int = 8

    @validator("mode")
    def _check_mode(cls, v: str) -> str:
        modes = {"single-sft", "multi-sft", "single-align", "multi-align"}
        if v not in modes:
            raise ValueError(f"`mode` must be one of {modes}")
        return v
