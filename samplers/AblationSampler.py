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
    def select_samples(self, samples_needed: int) -> Dict:
        samples_per_class = {}
        for class_name in self.class_samples.keys():
            samples_per_class[class_name] = list(self.class_samples[class_name])[:samples_needed]
        logger.info(f"Random {samples_needed} samples returned (ablation)")
        return samples_per_class
