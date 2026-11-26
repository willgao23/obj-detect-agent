from .agent import ObjDetectAgent
from .tools import yolo_obb, grounding_dino
import pandas as pd
import numpy as np 
import asyncio
from .helpers import convert_msg, map_tool_type, find_object, all_dota_class

async def vlm_agent_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Agent workflow for object detection routing task"""
    res_df = df
    with open("./data/prompts/sys_prompt_VLM.txt") as fp:
        sys_prompt = fp.read()
    config = {
            "model": "qwen3-vl:8b-instruct",
            "tools": [yolo_obb, grounding_dino],
            "system_prompt": sys_prompt
        }
    agent = ObjDetectAgent(config)
    prompts = res_df["Prompt"]
    text_msgs = np.asarray(prompts.apply(convert_msg))
    sem = asyncio.Semaphore(2)
    with open("./data/results_log.jsonl", "a") as out:
        results = await asyncio.gather(*[asyncio.create_task(agent.ainvoke_tool_eval(text_msg, sem, out)) for text_msg in text_msgs])
    tool_names = [result.tool_calls[0]["name"] if result.tool_calls else "none" for result in results]
    args = [result.tool_calls[0]["args"]["classes"] if result.tool_calls else "none" for result in results]
    called_type = [map_tool_type(tool_name, arg) for tool_name, arg in zip(tool_names, args)]
    res_df["Agent Tool"] = tool_names
    res_df["Agent Args"] = args
    res_df["Agent Called Type"] = called_type
    return res_df

def deterministic_routing(objs: list[str]) -> tuple[str, str]:
    """Determines the type of tool call to make based on the objects passed"""
    tool = "yolo_obb" if len(objs) == 0 or all_dota_class(objs) else "grounding_dino"
    called_type = map_tool_type(tool, objs)
    return (tool, called_type)

def deterministic_workflow(df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic baseline for object detection routing task"""
    res_df = df
    prompts = res_df["Prompt"]
    res_df["Det Args"] = prompts.apply(find_object)
    results = res_df["Det Args"].apply(deterministic_routing)
    res_df["Det Tools"] = [result[0] for result in results]
    res_df["Det Called Type"] = [result[1] for result in results]
    return res_df

def called_type_metrics(called_type: str, df: pd.DataFrame, isAgent:bool) -> tuple[float, float, float]:
    """Calculates the class-specific precision, recall, and f1 scores"""
    tp = sum(df.apply(lambda x: x["Type"] == called_type and x["Agent Called Type"] == called_type)
          if isAgent else
          df.apply(lambda x: x["Type"] == called_type and x["Det Called Type"] == called_type))
    fp = sum(df.apply(lambda x: x["Type"] != called_type and x["Agent Called Type"] == called_type)
              if isAgent else
              df.apply(lambda x: x["Type"] != called_type and x["Det Called Type"] == called_type))
    tn = sum(df.apply(lambda x: x["Type"] != called_type and x["Agent Called Type"] != called_type)
              if isAgent else
              df.apply(lambda x: x["Type"] != called_type and x["Det Called Type"] != called_type))
    fn = sum(df.apply(lambda x: x["Type"] == called_type and x["Agent Called Type"] != called_type)
              if isAgent else
              df.apply(lambda x: x["Type"] == called_type and x["Det Called Type"] != called_type))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = (2 * precision * recall) / (precision + recall)
    return (precision, recall, f1)

def eval_workflows() -> None:
    """Runs both the agentic and deterministic workflows to compare them"""
    df = pd.read_csv("./data/data.csv")
    agent_df = asyncio.run(vlm_agent_workflow(df))
    agent_df.to_csv("./data/data_agent_run.csv")
    final_df = deterministic_workflow(agent_df)
    final_df.to_csv("./data/data_final_eval.csv")
    type_accuracy_agent = np.sum(final_df["Agent Called Type"] == final_df["Type"]) / (len(final_df))
    arg_accuracy_agent = np.sum(final_df["Agent Args"] == final_df["Classes"]) / (len(final_df))
    type_accuracy_det = np.sum(final_df["Det Called Type"] == final_df["Type"]) / (len(final_df))
    arg_accuracy_det =  np.sum(final_df["Det Args"] == final_df["Classes"]) / (len(final_df))
    with open("./data/eval_metrics.txt", "w") as fp:
        fp.write(f"Agent Accuracies:\nTool Type: {type_accuracy_agent:.4f}\nArgs: {arg_accuracy_agent:.4f}")
        fp.write(f"\n------\n")
        fp.write(f"Deterministic Accuracies:\nTool Type: {type_accuracy_det:.4f}\nArgs: {arg_accuracy_det:.4f}")
        fp.write(f"\n------\n")
        metric_df = pd.DataFrame([final_df["Type"], final_df["Agent Called Type"], final_df["Det Called Type"]])
        for called_type in np.unique(np.asarray(final_df["Type"])):
            agent_prec, agent_rec, agent_f1 = called_type_metrics(called_type, metric_df, True)
            det_prec, det_rec, det_f1 = called_type_metrics(called_type, metric_df, False)
            fp.write(f"Agent Metrics for Class {called_type}:\nPrecision: {agent_prec:.4f}\nRecall: {agent_rec:.4f}\nF1: {agent_f1:.4f}")
            fp.write(f"\n------\n")
            fp.write(f"Deterministic Metrics for Class {called_type}:\nPrecision: {det_prec:.4f}\nRecall: {det_rec:.4f}\nF1: {det_f1:.4f}")
            fp.write(f"\n------\n")
