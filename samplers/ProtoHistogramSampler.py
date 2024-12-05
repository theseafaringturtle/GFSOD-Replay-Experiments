from __future__ import annotations

from typing import Dict, Union, Any, Tuple
import cv2
import torch
import logging
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from .ProtoSampler import ProtoSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class ProtoHistogramSampler(ProtoSampler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.NUM_BINS = 10

    @time_perf(logger)
    def select_samples(self, samples_needed) -> Dict:
        samples_per_class = {}
        for class_name in self.class_samples.keys():
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
            # Split ranked array into histogram bins
            bin_size = len(sim_scores) // self.NUM_BINS
            histogram_bins = [sim_scores[i * bin_size: (i + 1) * bin_size] for i in range(self.NUM_BINS)]
            # Sample uniformly from histograms
            selected_samples = []
            while len(selected_samples) < samples_needed:
                for i in range(self.NUM_BINS):
                    assert len(histogram_bins[i]) > 1, \
                        f"Splitting a pool of {len(self.class_samples[class_name])} into {self.NUM_BINS} bins left bin {i} without samples to draw from"
                    selected_samples.append(histogram_bins[i].pop(0)[0])
                    if len(selected_samples) == samples_needed:
                        break
            samples_per_class[class_name] = selected_samples
        logger.info("Samples have been ranked!")
        return samples_per_class
