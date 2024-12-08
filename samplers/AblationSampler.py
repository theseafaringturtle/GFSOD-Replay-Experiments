from __future__ import annotations

import logging
from typing import Dict

from .BaseSampler import BaseSampler
from .utils import time_perf

logger = logging.getLogger("defrcn").getChild(__name__)


class AblationSampler(BaseSampler):
    """An ablation class that does not create protos and picks samples at random without ranking"""

    def process_image_entry(self, input):
        pass

    def process_post(self):
        pass

    @time_perf(logger)
    def select_samples(self, instances_needed: int) -> Dict:
        samples_per_class = {cls_id: [] for cls_id in self.class_samples.keys()}
        instances_per_class = {cls_id: 0 for cls_id in self.class_samples.keys()}
        for class_name in self.class_samples.keys():
            for file_name in self.class_samples[class_name]:
                # Have at least one sample, but if other instances already contain that class ignore it
                samples_per_class[class_name].append(file_name)
                for label in self.sample_labels[file_name]:
                    instances_per_class[label] += 1
                if instances_per_class[class_name] >= instances_needed:
                    break
            samples_per_class[class_name] = list(self.class_samples[class_name])[:instances_needed]
        logger.info(f"Random {instances_needed} samples returned (ablation)")
        return samples_per_class
