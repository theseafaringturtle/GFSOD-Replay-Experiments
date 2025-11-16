import os
import sys

import pandas
import numpy as np
import matplotlib.pyplot as plt

from utils import get_voc_data, get_coco_data

plt.rcParams.update({'font.size': 13})

# Get dataset name
if len(sys.argv) < 3:
    raise ValueError("Usage: python plots/bar_plot.py <dataset_name:COCO|VOC Split x> <exp_name:sampling|grad")

# "COCO", "VOC Split 1"
dataset_name = sys.argv[1]
exp_set = sys.argv[2]

if not os.path.exists(f'plots/outputs/{dataset_name}/{exp_set}'):
    os.mkdir(f'plots/outputs/{dataset_name}/{exp_set}')

if "voc" in dataset_name.lower():
    split_number = int(dataset_name[-1])

# Get experiment names
grad_voc_exp_names = {"agem": "A-GEM", "mega2": "MEGA-II",
                      "cag": "CAG", "cfa": "CFA", "cfaloss": "CFAL", "avgonly": "Averaging"}  # , "vanilla_er"]
grad_coco_exp_names = {"agem": "A-GEM", "mega2": "MEGA-II",
                       "cag": "CAG", "cfa": "CFA", "cfaloss": "CFAL", "avgonly": "Averaging"}
sam_coco_exp_names = {"sampled_ablation_cap_100": "Random",
                      "sampled_proto_cap_100": "ProtoDist",
                      # Note to self: accidentally  renamed, use _nomin for VOC, _cap_100 for COCO
                      "sampled_protosim_nomin": "ProtoDist Ratio",
                      "sampled_protosim_nomin_er": "ProtoDist Ratio (ER)"
                      }
sam_voc_exp_names = sam_coco_exp_names

def get_official_name(name: str):
    if name == "vanilla":
        return "Original"
    return exp_names[name]

exp_names = sam_coco_exp_names
if "VOC" in dataset_name:
    if exp_set == "sampling":
        start_seed, end_seed = 30, 39
        exp_names = sam_voc_exp_names
    elif exp_set == "grad":
        start_seed, end_seed = 0, 9
        exp_names = grad_voc_exp_names
    else:
        raise NotImplementedError("Change this snippet to add experiments your own dataset")
elif "COCO" in dataset_name:
    if exp_set == "sampling":
        start_seed, end_seed = 10, 19
        exp_names = sam_coco_exp_names
    elif exp_set == "grad":
        start_seed, end_seed = 0, 9
        exp_names = grad_coco_exp_names
    else:
        raise NotImplementedError("Change this snippet to add experiments your own dataset")
else:
    raise NotImplementedError("Change this snippet to add experiments your own dataset")

# Load experiment data (default implementation uses each metric.json)
shots = [1, 5, 10]

color_pool = ['#D81B51', '#1E88E5', '#FD9C14', '#116758', '#48FF00', '#D0737B', '#98FBB7', '#791C98', '#E6BD6E']

exp_rows = []
for exp_name in exp_names.keys():
    for shot in shots:
        for seed in range(start_seed, end_seed + 1):
            if "VOC" in dataset_name:
                bAP, nAP = get_voc_data(exp_name, split_number, shot, seed)
            elif "COCO" in dataset_name:
                bAP, nAP = get_coco_data(exp_name, shot, seed)
            else:
                raise NotImplementedError("Change this snippet to handle your own dataset")
            row = {"name": exp_name, "shot": shot, "bAP": bAP, "nAP": nAP, "seed": seed}
            exp_rows.append(row)

# Add vanilla separately since the old "data seed" is different from sampling experiments
for shot in shots:
    for seed in range(0, 10):
        if "VOC" in dataset_name:
            bAP, nAP = get_voc_data("vanilla", split_number, shot, seed)
        elif "COCO" in dataset_name:
            bAP, nAP = get_coco_data("vanilla", shot, seed)
        else:
            raise NotImplementedError("Change this snippet to handle your own dataset")
        row = {"name": "vanilla", "shot": shot, "bAP": bAP, "nAP": nAP, "seed": seed}
        exp_rows.append(row)

df = pandas.DataFrame(exp_rows)

af = df.groupby(["name", "shot"], sort=False)[["bAP", "nAP"]].agg([np.mean, np.std]).reset_index()

metrics = ["bAP", "nAP"]

for shot in shots:
    for metric in metrics:
        exp_af = af[af['shot'] == shot][['name', metric]]
        names = [get_official_name(unofficial_name) for unofficial_name in exp_af['name'].to_numpy()] #
        means = exp_af[metric]['mean'].to_numpy()
        stds = exp_af[metric]['std'].to_numpy()
        cis = np.array([s * 1.96 / np.sqrt(len(range(start_seed, end_seed))) for s in stds])
        del stds
        # Viz
        if exp_set == "grad":
            plt.figure(figsize=(8, 5))
        else:
            plt.figure(figsize=(8.5, 6)) # 6.5
        for i, m in enumerate(means):
            plt.bar(i+1, m, yerr=cis[i], width=0.6, alpha=0.95, color=color_pool[i], zorder=5, label=names[i])
        # plt.bar(range(len(means)), means, yerr=cis, align='center', width=0.5, alpha=0.95, color=color_pool, zorder=5)
        # Legends
        handles = [plt.Rectangle((0, 0), 1, 1, color=color_pool[i]) for i, name in enumerate(names)]
        plt.legend(handles, names, loc="upper left", bbox_to_anchor=(1.1, 1))

        plt.grid(axis='y', zorder=0)
        plt.xticks(range(len(names)), ['' for n in names], rotation=45, ha='right') # Dummy X labels since we're using the legend
        plt.ylabel(metric)
        plt.title(f'{metric} for {shot}-shot {dataset_name}')
        # Truncate lower portion if too high, so we can see differences better
        min_index = np.argmin(means)
        min_mean = means[min_index]
        min_ci = cis[min_index]
        y_start, y_end = plt.ylim()
        if min_mean - np.round(min_ci) < 10:
            y_start = 0
        elif min_mean - np.round(min_ci) < 20:
            y_start = 10
        elif min_mean - np.round(min_ci) < 30:
            y_start = 20
        elif min_mean - np.round(min_ci) < 40:
            y_start = 30
        else:
            y_start = 40
        if y_end < 10:
            y_end = 10
        plt.ylim(bottom=y_start, top=y_end)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f'plots/outputs/{dataset_name}/{exp_set}/{shot}shot_{metric}_{dataset_name}.png')
        plt.close()

print(f"Figures generated for {dataset_name}")

# convert +append 1shot*.png VOC1_1shot.png
# convert +append 5shot*.png VOC1_5shot.png
# convert +append 10shot*.png VOC1_10shot.png
# convert -append VOC1_1shot.png VOC1_5shot.png VOC1_10shot.png VOC1.png
