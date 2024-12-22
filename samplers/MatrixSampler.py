import logging
from typing import Dict, List

import cv2
import numpy as np
import numpy.linalg.linalg
import torch
from detectron2.structures import Boxes
from torch import Tensor

from .BaseFeatureSampler import BaseFeatureSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class MatrixSampler(BaseFeatureSampler):

    def __init__(self, cfg):
        super(BaseFeatureSampler, self).__init__(cfg)
        self.sample_roi_features = {}  # file_name to feature tensor
        self.sample_roi_labels = {}  # file_name to feature labels
        self.NUM_BINS = 10

    def process_image_entry(self, entry):
        # Load support images and gt-boxes. Same as PCB.
        file_name = entry['file_name']
        gt_classes = torch.tensor([anno["category_id"] for anno in entry["annotations"]])

        img = cv2.imread(file_name)  # BGR
        boxes = Boxes(torch.tensor([anno["bbox"] for anno in entry["annotations"]]))

        # extract roi features
        _features = self.extract_roi_features(img, [boxes])  # use list since it expects a batch
        avg_features, avg_labels = self.average_roi_features(_features, gt_classes)
        self.sample_roi_features[file_name] = avg_features.cpu().clone()
        self.sample_roi_labels[file_name] = avg_labels.cpu().clone()

    @time_perf(logger)
    def process_post(self):
        """Euclidean distance between each pair of images"""
        # Dimensions: num_filenames x num_filenames
        self.file_features: np.ndarray[(str, Tensor)] = np.array(list(self.sample_roi_features.items()), dtype=object)
        dist_matrix = np.tile(np.inf, (len(self.file_features), len(self.file_features)))
        for i, (file_name, feature_tensors) in enumerate(self.file_features):
            entries = []
            for j, (other_file_name, other_feature_tensors) in enumerate(self.file_features):
                if file_name == other_file_name:
                    entries.append(0.)
                    continue
                # Copy half of matrix since it's symmetric
                if dist_matrix[j][i] != np.inf:
                    dist_matrix[i][j] = dist_matrix[j][i]
                    continue
                # non-weighted average
                distance_scores = [np.linalg.linalg.norm(feature_t - other_feature_t)
                                   for feature_t in feature_tensors for other_feature_t in other_feature_tensors]
                dist_matrix[i][j] = sum(distance_scores) / len(distance_scores)
        self.dist_matrix = dist_matrix

    @time_perf(logger)
    def select_samples(self, instances_needed) -> Dict:
        # Rank each row file_name based on their distance to all other sample features.
        # This means summing up distance entries in that row
        rank = np.zeros(len(self.dist_matrix[0]))
        for i, _ in enumerate(self.file_features):
            rank[i] = np.sum(self.dist_matrix[i])
        rank_indices = np.argsort(rank)
        self.file_features = self.file_features[rank_indices]
        # Make sure hard and easy samples are included by using histogram method
        samples_per_class = {cls_id: [] for cls_id in self.class_samples.keys()}
        instances_per_class = {cls_id: 0 for cls_id in self.class_samples.keys()}
        bin_size = len(rank_indices) // self.NUM_BINS
        histogram_bins = [self.file_features[i * bin_size: (i + 1) * bin_size].tolist() for i in range(self.NUM_BINS)]
        for class_id in self.class_samples.keys():
            while instances_per_class[class_id] < instances_needed:
                for i in range(self.NUM_BINS):
                    assert len(histogram_bins[i]) >= 1, \
                        f"Splitting a pool of {len(self.class_samples[class_id])} into {self.NUM_BINS} bins left bin {i} without samples to draw from"
                    file_name = histogram_bins[i].pop(0)[0]
                    samples_per_class[class_id].append(file_name)
                    for label in self.sample_labels[file_name]:
                        instances_per_class[label] += 1
                    if instances_per_class[class_id] >= instances_needed:
                        break
        return samples_per_class
