from datasets import load_dataset, load_from_disk
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
import sys
from io import BytesIO
import base64
import asyncio
from langchain_ollama import ChatOllama
from langchain.messages import HumanMessage, SystemMessage
from PIL import Image
import yaml

load_dotenv()
HF_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

def fetch_imgs() -> None:
    """Loads and saves the AID dataset from HuggingFace"""
    ds = load_dataset("blanchon/AID", token=HF_KEY)
    ds.save_to_disk("./data")

def get_coco_classes() -> str:
    """Loads the COCO classes in coco.yaml as a comma separated string"""
    with open("./data/coco.yaml") as fp:
        dict = yaml.safe_load(fp)
    return ",".join([v for _, v in dict["names"].items()])

def convert_b64(path: str) -> str:
    """A helper function to convert an image in a path to base64"""
    with Image.open(path) as img:
        buf = BytesIO()
        img.save(buf, format="jpeg")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

def choose_prompt(prompt_type: str) -> str:
    """Varies the user prompt based on index"""
    prompt = ""
    if prompt_type == "all":
        prompt = f"""Write the user message with these constraints:
            - Target: all objects, do NOT specify a particular object or object class
            - Specificity: broad"""
    elif prompt_type == "coco":
        prompt = f"""Write the user message with these constraints:
            - Target: a single object class in the following: {get_coco_classes()}
            - Specificity: broad"""
    else:
        prompt = f"""Write the user message with these constraints:
            - Target: a single object class NOT in the following: {get_coco_classes()}
            - Specificity: precise"""
    return prompt

def generate_messages(path: str, prompt_type: str) -> list[SystemMessage, HumanMessage]:
    """Create messages to prompt the LLM with for mock user prompt creation"""
    img_b64 = convert_b64(path)
    text_content = {"type": "text", "text": f"{choose_prompt(prompt_type)}"}
    img_content = {"type": "image_url", "image_url": f"data:image/jpeg;base64,{img_b64}"}
    im_prompt = HumanMessage(content=[text_content, img_content]) if prompt_type != "all" else HumanMessage(content=[text_content])
    with open("./data/sys_prompt.txt") as fp:
        sys_prompt = fp.read()
    sys_prompt = SystemMessage(sys_prompt)
    return [sys_prompt, im_prompt]

async def create_prompts(df: pd.DataFrame) -> pd.DataFrame:
    """Adds prompts to image dataset for workflow eval"""
    shuffled_df = df.sample(frac=1, random_state=42)
    shuffled_df = shuffled_df[:1000]
    paths = shuffled_df["image_path"].to_numpy()
    all, coco, non_coco = np.array_split(paths, 3)
    all_msgs = [generate_messages(path, "all") for path in all]
    coco_msgs = [generate_messages(path, "coco") for path in coco]
    non_coco_msgs = [generate_messages(path, "non_coco") for path in non_coco]
    msgs = all_msgs + coco_msgs + non_coco_msgs
    llm = ChatOllama(model="qwen3-vl:8b-instruct", temperature=0.5, num_predict=128)
    responses = await asyncio.gather(*[asyncio.create_task(llm.ainvoke(msg)) for msg in msgs])
    content = [res.content_blocks[0]["text"] for res in responses]
    prompt_type = ["all"] * len(all_msgs) + ["coco"] * len(coco_msgs) + ["non_coco"] * len(non_coco_msgs)
    output_df = pd.DataFrame({"Path": paths, "Prompt": content, "Type": prompt_type})
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
