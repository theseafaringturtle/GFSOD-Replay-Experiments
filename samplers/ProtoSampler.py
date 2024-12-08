from __future__ import annotations

from typing import Dict, Union, Any, Tuple

import torch
from detectron2.data import MetadataCatalog

import os
import cv2
import torch
import logging

from detectron2.structures import Boxes
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .BaseFeatureSampler import BaseFeatureSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class ProtoSampler(BaseFeatureSampler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sample_roi_features = {}  # file_name to feature tensor
        self.sample_roi_labels = {}  # file_name to feature labels
        self.all_features = []
        self.all_labels = []

    def process_image_entry(self, input):
        # Load support images and gt-boxes. Same as PCB.
        file_name = input['file_name']
        gt_classes = input['instances'].get("gt_classes")

        img = cv2.imread(file_name)  # BGR
        img_h, img_w = img.shape[0], img.shape[1]
        ratio = img_h / input['instances'].image_size[0]
        input['instances'].gt_boxes.tensor = input['instances'].gt_boxes.tensor * ratio
        boxes = input["instances"].gt_boxes.clone().to(self.device)

        # extract roi features
        _features = self.extract_roi_features(img, [boxes])  # use list since it expects a batch
        avg_features, avg_labels = self.average_roi_features(_features, gt_classes)
        self.sample_roi_features[file_name] = avg_features.cpu().clone()
        self.all_features.append(avg_features.cpu().clone().data)

        self.sample_roi_labels[file_name] = avg_labels.cpu().clone()
        self.all_labels.append(avg_labels.cpu().clone().data)

    @time_perf(logger)
    def process_post(self):
        # concat
        self.all_features = torch.cat(self.all_features, dim=0)
        self.all_labels = torch.cat(self.all_labels, dim=0)
        assert self.all_features.shape[0] == self.all_labels.shape[0]

        # calculate prototype
        features_dict = {}
        for i, label in enumerate(self.all_labels):
            label = int(label)
            if label not in features_dict:
                features_dict[label] = []
            features_dict[label].append(self.all_features[i].unsqueeze(0))

        prototypes_dict = {}
        for label in features_dict:
            print(f"Creating prototype for class {label} ({self.base_class_id_to_name(label)})")
            features = torch.cat(features_dict[label], dim=0)
            prototypes_dict[label] = torch.mean(features, dim=0, keepdim=True)
        self.prototypes = prototypes_dict

    @time_perf(logger)
    def select_samples(self, instances_needed) -> Dict:
        samples_per_class = {cls_id: [] for cls_id in self.class_samples.keys()}
        instances_per_class = {cls_id: 0 for cls_id in self.class_samples.keys()}
        for class_name in self.class_samples.keys():
            # same_class_dist = []
            # other_class_dist = []
            sim_scores = []
            for file_name in self.class_samples[class_name]:
                sample_features = self.sample_roi_features[file_name]
                # When there are multiple RoI box features in an image, can't average them since they might have different labels
                # Average the ones with the same label instead.
                sample_labels = self.sample_roi_labels[file_name]
                sample_feature_means = {}
                for label in torch.unique(sample_labels):
                    sample_feature_means[label.item()] = sample_features[sample_labels == label].mean(axis=0)
                sample_distances = {}
                for label in sample_feature_means:
                    dist = euclidean_distances(self.prototypes[label], sample_feature_means[label].unsqueeze(0))
                    sample_distances[label] = dist
                # Create a collated similarity score from different labels
                sim_score = 0.
                for label in sample_distances:
                    if label == class_name:
                        sim_score += sample_distances[label]
                sim_tuple = (file_name, sim_score)
                sim_scores.append(sim_tuple)
            sim_scores.sort(key=lambda tup: tup[1])
            # print(f"Distances: {sim_scores}")
            for file_name, dist in sim_scores:
                # Have at least one sample, but if other instances already contain that class ignore it
                samples_per_class[class_name].append(file_name)
                for label in self.sample_labels[file_name]:
                    instances_per_class[label] += 1
                if instances_per_class[class_name] >= instances_needed:
                    break
            # samples_per_class[class_name] = [file_name for file_name, dist in sim_scores[:samples_needed]]
        logger.info("Samples have been ranked!")
        return samples_per_class
