#!/usr/bin/env python3
import os
import json
import re
import numpy as np
import argparse
from tabulate import tabulate

datasets = [
    "ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211",
    "OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA",
    "VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS",
    "MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"
]

datasets_class = {
	"Classification": ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397", "Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211"],
	"VQA": ["OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W", "ScienceQA", "VizWiz", "GQA", "TextVQA"],
	"Retrieval": ["VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS"],
	"Visual_Grounding": ["MSCOCO", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"],
    "IND": ["ImageNet-1K", "N24News", "HatefulMemes", "VOC2007", "SUN397","OK-VQA", "A-OKVQA", "DocVQA", "InfographicsVQA", "ChartQA", "Visual7W","VisDial", "CIRR", "VisualNews_t2i", "VisualNews_i2t", "MSCOCO_t2i", "MSCOCO_i2t", "NIGHTS", "WebQA", "MSCOCO"],
    "OOD": ["Place365", "ImageNet-A", "ImageNet-R", "ObjectNet", "Country211", "ScienceQA", "VizWiz", "GQA", "TextVQA", "FashionIQ", "Wiki-SS-NQ", "OVEN", "EDIS", "RefCOCO", "RefCOCO-Matching", "Visual7W-Pointing"]
}


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--checkpoint_path", type=str, required=True)
	parser.add_argument("--output_path", type=str, required=True)
	args = parser.parse_args()
	os.makedirs(args.output_path, exist_ok=True)

	checkpoint_paths = [ 
		args.checkpoint_path
	]

	def extract_step(checkpoint_name):
		match = re.search(r'checkpoint-(\d+)', checkpoint_name)
		return int(match.group(1)) if match else float('inf')


	gathered_scores_by_exp = {}

	for checkpoint_path in checkpoint_paths:
		step = extract_step(checkpoint_path)
		experiment_dir = checkpoint_path.split("/")[-3]

		if str.isdigit(str(step)):
			checkpoint_scores = {"experiment": experiment_dir, "checkpoint": str(step)}
		else:
			checkpoint_scores = {"experiment": experiment_dir, "checkpoint": "default"}

		for dataset in datasets:
			score_file = os.path.join(checkpoint_path, f"{dataset}_score.json") 
			if not os.path.exists(score_file): print(f"Score file {score_file} does not exist")
			if os.path.isfile(score_file):
				with open(score_file, "r") as f:
					score_data = json.load(f) 
					checkpoint_scores[dataset] = score_data.get("acc", "N/A")
			else:
				checkpoint_scores[dataset] = "N/A" 

		gathered_scores_by_exp[experiment_dir] = checkpoint_scores

	header = ["dataset"] + list(gathered_scores_by_exp.keys())
	table = []
	table_final = []
	check_list = {}

	for dataset in datasets:
		row = [dataset]
		for experiment, scores in gathered_scores_by_exp.items():
			row.append(scores[dataset])
		table.append(row)
		check_list[dataset] = row[1]


	for class_name, datasets in datasets_class.items():
		row = [class_name, f"{np.mean([check_list[dataset] for dataset in datasets]):.3f}"]
		table_final.append(row)

	table_final.append(["Average", f"{np.mean([check_list[each] for _, datasets in datasets_class.items() for each in datasets ]):.3f}"])

	table_str = tabulate(table, headers=header, tablefmt="grid")
	table_final_str = tabulate(table_final, headers=header, tablefmt="grid")

	print(table_str)
	print(table_final_str)

	with open(os.path.join(args.output_path, "MMEB_eval_conclude.txt"), "w") as f:
		f.write(table_str)
		f.write(table_final_str)