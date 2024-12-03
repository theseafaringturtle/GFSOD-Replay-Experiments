from typing import Tuple

import torch
from detectron2.structures import ImageList

from .BaseSampler import BaseSampler


class BaseFeatureSampler(BaseSampler):

    def __init__(self, cfg):
        super().__init__(cfg)
        self.sample_roi_features = {}  # file_name to feature tensor
        self.sample_roi_labels = {}  # file_name to feature labels

    def extract_roi_features(self, img, boxes):
        mean = torch.tensor([0.406, 0.456, 0.485]).reshape((3, 1, 1)).to(self.device)
        std = torch.tensor([[0.225, 0.224, 0.229]]).reshape((3, 1, 1)).to(self.device)

        img = img.transpose((2, 0, 1))
        img = torch.from_numpy(img).to(self.device)
        images = [(img / 255. - mean) / std]
        images = ImageList.from_tensors(images, 0)
        conv_feature = self.imagenet_model(images.tensor[:, [2, 1, 0]])[1]  # size: BxCxHxW

        box_features = self.roi_pooler([conv_feature], boxes).squeeze(2).squeeze(2)

        activation_vectors = self.imagenet_model.fc(box_features)

        return activation_vectors.detach()

    def average_roi_features(self, features, labels) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average features for instances of the same class appearing in the same image.
           This makes images with multiple object instances less over-represented in final prototype"""
        # Get unique labels and their indices
        uniq_labels, indices = torch.unique(labels, return_inverse=True)
        uniq_labels = uniq_labels.to(features.device)
        indices = indices.to(features.device)

        # Use scatter_add to sum features for each unique label
        avg_features = torch.zeros(uniq_labels.size(0), features.size(1), dtype=features.dtype, device=features.device)
        avg_features.scatter_add_(0, indices.unsqueeze(1).expand(-1, features.size(1)), features)

        # Count occurrences of each label
        label_counts = torch.bincount(indices)

        # Divide summed features by label counts to get averages
        avg_features /= label_counts.float().unsqueeze(1)

        return avg_features, uniq_labels
