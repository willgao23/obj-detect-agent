from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import sys
import asyncio
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
import yaml
from obj_detect_agent.helpers import convert_b64, convert_msg
import ast

load_dotenv()
HF_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def fetch_imgs() -> None:
    """Loads and saves the AID dataset from HuggingFace"""
    ds = load_dataset("blanchon/AID", token=HF_KEY)
    ds.save_to_disk("./data")

def get_dota_classes() -> str:
    """Loads the DOTA classes in dota.yaml as a comma separated string"""
    with open("./data/dota.yaml") as fp:
        dict = yaml.safe_load(fp)
    return ",".join([k for k, _ in dict["names"].items()])

def choose_prompt(prompt_type: str) -> str:
    """Varies the user prompt based on index"""
    prompt = ""
    if prompt_type == "general":
        prompt = f"""Write the user message with these constraints:
            <GENERAL>"""
    elif prompt_type == "dota":
        prompt = f"""Write the user message with these constraints:
            - Target: a single object class in the following: {get_dota_classes()}
            - Specificity: broad"""
    else:
        prompt = f"""Write the user message with these constraints:
            - Target: a single object class NOT in the following: {get_dota_classes()}
            - Specificity: precise"""
    return prompt

def generate_messages(path: str, prompt_type: str) -> list[SystemMessage, HumanMessage]:
    """Create messages to prompt the LLM with for mock user prompt creation"""
    im_prompt = convert_msg(choose_prompt(prompt_type), convert_b64(path)) if prompt_type != "general" else convert_msg(choose_prompt(prompt_type), None)
    with open("./data/prompts/sys_prompt.txt") as fp:
        sys_prompt = fp.read()
    sys_prompt = SystemMessage(sys_prompt)
    return [sys_prompt, im_prompt]

async def create_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds prompts to image dataset for workflow eval"""
    shuffled_df = df.sample(frac=1, random_state=42)
    shuffled_df = shuffled_df[:1000]
    paths = shuffled_df["image_path"].to_numpy()
    general, dota, vocab_free = np.array_split(paths, 3)
    general_msgs = [generate_messages(path, "general") for path in general]
    dota_msgs = [generate_messages(path, "dota") for path in dota]
    open_vocab_msgs = [generate_messages(path, "open_vocab") for path in vocab_free]
    msgs = general_msgs + dota_msgs + open_vocab_msgs
    llm = ChatOllama(model="qwen3-vl:8b-instruct", temperature=0.5, num_predict=128)
    responses = await asyncio.gather(*[asyncio.create_task(llm.ainvoke(msg)) for msg in msgs])
    prompts = [ast.literal_eval(res.content_blocks[0]["text"])["message"] for res in responses]
    classes = [ast.literal_eval(res.content_blocks[0]["text"])["classes"] for res in responses]
    prompt_type = ["general"] * len(general_msgs) + ["dota"] * len(dota_msgs) + ["open_vocab"] * len(open_vocab_msgs) 
    output_df = pd.DataFrame({"Path": paths, "Prompt": prompts, "Classes": classes, "Type": prompt_type})
    return output_df

if __name__ == "__main__":
    """Construct the evaluation dataset with images and mock user prompts.  
    Only fetches images and saves them if script is called with 'fetch' as an arg"""
    mode = sys.argv[1]
    if mode == "fetch":
        fetch_imgs()
        ds = load_from_disk("./data")
        ds = ds.with_format("pandas")
        df = ds["train"][:]
        im_paths = []
        for index, row in df.iterrows():
            path = f"./data/images/{index}.jpg"
            row["image"].save(path)
            im_paths.append(path)
        df.insert(len(df.columns), "image_path", im_paths)
        no_img_df = df.drop(["image"], axis=1)
        no_img_df.to_csv("./data/image_paths.csv", index=False)
    img_path_df = pd.read_csv("./data/image_paths.csv")
    df_final = asyncio.run(create_prompts(img_path_df))
    df_final.to_csv("./data/data.csv", index=False)
    print("Dataset creation complete.")
