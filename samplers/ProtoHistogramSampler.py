from __future__ import annotations

from typing import Dict, Union, Any, Tuple
import cv2
import torch
import logging

from .ProtoSampler import ProtoSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class ProtoHistogramSampler(ProtoSampler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.NUM_BINS = 10

    @time_perf(logger)
    def select_samples(self, instances_needed) -> Dict:
        samples_per_class = {cls_id: [] for cls_id in self.class_samples.keys()}
        instances_per_class = {cls_id: 0 for cls_id in self.class_samples.keys()}
        for class_id in self.class_samples.keys():
            sim_scores = []
            for file_name in self.class_samples[class_id]:
                sample_features = self.sample_roi_features[file_name]
                # When there are multiple RoI box features in an image, can't average them since they might have different labels
                # Average the ones with the same label instead.
                sample_labels = self.sample_roi_labels[file_name]
                sample_feature_means = {}
                for label in torch.unique(sample_labels):
                    sample_feature_means[label.item()] = sample_features[sample_labels == label].mean(axis=0)
                sample_distances = {}
                for label in sample_feature_means:
                    dist = torch.linalg.norm(self.prototypes[label] - sample_feature_means[label]).item()
                    sample_distances[label] = dist
                # Create a collated similarity score from different labels
                sim_score = 0.
                for label in sample_distances:
                    if label == class_id:
                        sim_score += sample_distances[label]
                sim_tuple = (file_name, sim_score)
                sim_scores.append(sim_tuple)
            sim_scores.sort(key=lambda tup: tup[1])
            # Split ranked array into histogram bins
            bin_size = len(sim_scores) // self.NUM_BINS
            histogram_bins = [sim_scores[i * bin_size: (i + 1) * bin_size] for i in range(self.NUM_BINS)]
            # Sample uniformly from histograms, having at least one for each category (even if it appeared somewhere else)
            added_first = False
            while instances_per_class[class_id] < instances_needed or not added_first:
                # Pick from each bin, starting from closest one, so that 1-shot is equivalent to proto, but this will add further samples as it scales
                for i in range(self.NUM_BINS):
                    assert len(histogram_bins[i]) >= 1, \
                        f"Splitting a pool of {len(self.class_samples[class_id])} into {self.NUM_BINS} bins left bin {i} without samples to draw from"
                    file_name = histogram_bins[i].pop(0)[0]
                    samples_per_class[class_id].append(file_name)
                    added_first = True
                    for label in self.sample_labels[file_name]:
                        instances_per_class[label] += 1
                    if instances_per_class[class_id] >= instances_needed:
                        break
        logger.info("Samples have been ranked!")
        print(samples_per_class)
        return samples_per_class
