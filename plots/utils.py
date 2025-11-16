import json
import os
from typing import Tuple

RES_DIR = os.environ.get("RES_DIR", "../results/thesis/final")

def get_coco_data(exp_name: str, shot: int, seed: int) -> Tuple[float, float]:
    dir = f"{RES_DIR}/coco/{exp_name}/defrcn_gfsod_r101_novel/tfa-like"
    return read_exp_metrics(metrics_file_name=f"{dir}/{shot}shot_seed{seed}/metrics.json")


def get_voc_data(exp_name: str, split: int, shot: int, seed: int) -> Tuple[float, float]:
    dir = f"{RES_DIR}/voc/{exp_name}/defrcn_gfsod_r101_novel{split}/tfa-like"
    return read_exp_metrics(metrics_file_name=f"{dir}/{shot}shot_seed{seed}/metrics.json")


def read_exp_metrics(metrics_file_name: str) -> Tuple[float, float]:
    with open(metrics_file_name, 'r') as f:
        # Last line is always the final metrics
        results_json: str = f.readlines()[-1]
    results_dict: dict = json.loads(results_json)
    return results_dict['bbox/bAP'], results_dict['bbox/nAP']
