from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
import os
import pandas as pd
import sys

load_dotenv()
HF_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def fetch_imgs() -> None:
    """Loads and saves the AID dataset from HuggingFace"""
    ds = load_dataset("blanchon/AID", token=HF_KEY)
    ds.save_to_disk("./data")

def create_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds prompts to image dataset for workflow eval"""
    #TODO:add LLM-generated prompts for eval dataset
    return

if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "fetch":
        fetch_imgs()
    ds = load_from_disk("./data")
    ds = ds.with_format("pandas")
    df = ds["train"][:]
