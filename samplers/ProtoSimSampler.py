from __future__ import annotations

from typing import Dict
import torch
import logging

from .ProtoSampler import ProtoSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class ProtoSimSampler(ProtoSampler):

    def __init__(self, cfg):
        super().__init__(cfg)

    @time_perf(logger)
    def select_samples(self, instances_needed) -> Dict:
        # Class to Filename output and instance number checker
        samples_per_class = {cls_id: [] for cls_id in self.class_samples.keys()}
        instances_per_class = {cls_id: 0 for cls_id in self.class_samples.keys()}
        # Prototypical distances
        sample_sim_scores = {}  # filename to {id: int} scores of distance to each centoid
        for class_id in self.class_samples.keys():
            for file_name in self.class_samples[class_id]:
                sample_features = self.sample_roi_features[file_name]
                # When there are multiple RoI box features in an image, can't average them since they might have different labels
                # Average the ones with the same label instead.
                sample_labels = self.sample_roi_labels[file_name]
                sample_feature_means = {}
                for label in torch.unique(sample_labels):
                    sample_feature_means[label.item()] = sample_features[sample_labels == label].mean(axis=0)
                distances = {}
                for label in sample_feature_means:
                    if label == class_id:
                        for proto_label, proto_vec in self.prototypes.items():
                            dist = torch.linalg.norm(self.prototypes[proto_label] - sample_feature_means[label]).item()
                            distances[proto_label] = dist
                        sample_sim_scores[file_name] = distances
                        break
            # Create a ranking similarity score for closeness to correct centroid versus other ones
            # Chosen: ratio between distance to correct centroid and next closest one.
            # This means mis-classifications will have ratio > 1.
            sim_scores = []
            for file_name in self.class_samples[class_id]:
                distances = sample_sim_scores[file_name]
                dist_correct = distances[class_id]
                sorted_distances = sorted([(proto_cid, dist) for proto_cid, dist in distances.items()],
                                          key=lambda x: x[1])
                proto_next, dist_next = next(filter(lambda el: el[0] != class_id, sorted_distances))
                try:
                    ratio = dist_correct / dist_next
                except ZeroDivisionError:
                    ratio = 10e5
                sim_scores.append((file_name, ratio))
            sim_scores.sort(key=lambda proto_score_tup: proto_score_tup[1])
            # Collect samples
            for file_name, dist in sim_scores:
                # Have at least one sample, but if other instances already contain that class ignore it
                samples_per_class[class_id].append(file_name)
                for label in self.sample_labels[file_name]:
                    instances_per_class[label] += 1
                if instances_per_class[class_id] >= instances_needed:
                    break
        logger.info("Samples have been ranked!")
        return samples_per_class