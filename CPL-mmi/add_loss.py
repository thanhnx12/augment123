from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import Tensor, nn
import numpy as np 
from enum import Enum
from torch.nn import functional as F
from collections import defaultdict
# from sentence_transformers import util
# from sentence_transformers.SentenceTransformer import SentenceTransformer


def _convert_to_tensor(a: list | np.ndarray | Tensor):
    """
    Converts the input `a` to a PyTorch tensor if it is not already a tensor.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input array or tensor.

    Returns:
        Tensor: The converted tensor.
    """
    if not isinstance(a, Tensor):
        a = torch.tensor(a)
    return a

def _convert_to_batch(a: Tensor):
    """
    If the tensor `a` is 1-dimensional, it is unsqueezed to add a batch dimension.

    Args:
        a (Tensor): The input tensor.

    Returns:
        Tensor: The tensor with a batch dimension.
    """
    if a.dim() == 1:
        a = a.unsqueeze(0)
    return a


def _convert_to_batch_tensor(a: list | np.ndarray | Tensor):
    """
    Converts the input data to a tensor with a batch dimension.

    Args:
        a (Union[list, np.ndarray, Tensor]): The input data to be converted.

    Returns:
        Tensor: The converted tensor with a batch dimension.
    """
    a = _convert_to_tensor(a)
    a = _convert_to_batch(a)
    return a

def normalize_embeddings(embeddings: Tensor):
    """
    Normalizes the embeddings matrix, so that each sentence embedding has unit length.

    Args:
        embeddings (Tensor): The input embeddings matrix.

    Returns:
        Tensor: The normalized embeddings matrix.
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)

def cos_sim(a: list | np.ndarray | Tensor, b: list | np.ndarray | Tensor):
    """
    Computes the cosine similarity between two tensors.

    Args:
        a (Union[list, np.ndarray, Tensor]): The first tensor.
        b (Union[list, np.ndarray, Tensor]): The second tensor.

    Returns:
        Tensor: Matrix with res[i][j] = cos_sim(a[i], b[j])
    """
    a = _convert_to_batch_tensor(a)
    b = _convert_to_batch_tensor(b)

    a_norm = normalize_embeddings(a)
    b_norm = normalize_embeddings(b)
    return torch.mm(a_norm, b_norm.transpose(0, 1))


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it requires more
              training time.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    # def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    #     """Returns token_embeddings, cls_token"""
    #     trans_features = {"input_ids": features["input_ids"], "attention_mask": features["attention_mask"]}


# khi load input vi du la 1 batch thi input dau vao phai la 1 cai dict giua input dau vao description tuong ung 

#     The embeddings for the anchor sentences are stored in embeddings_a, and the embeddings for the positive sentences are concatenated into embeddings_b.
# The similarity scores between embeddings_a and embeddings_b are computed using the provided similarity function and then scaled.
    def forward(self, embeddings_a, embeddings_b, labels) -> Tensor:
        # reps = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]
        # embeddings_a = reps[0]
        # embeddings_b = torch.cat(reps[1:])
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        # Example a[i] should match with b[i]

        #  print(scores.shape) b*b
        # range_labels.shape = b
        # range_labels = torch.arange(0, scores.size(0), device=scores.device)
        
        return self.cross_entropy_loss(scores, labels)



    
    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}



class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)



class BatchHardTripletLossDistanceFunction:
    """This class defines distance functions, that can be used with Batch[All/Hard/SemiHard]TripletLoss"""

    @staticmethod
    def cosine_distance(embeddings: Tensor) :
        """Compute the 2D matrix of cosine distances (1-cosine_similarity) between all embeddings."""
        return 1 - cos_sim(embeddings, embeddings)

    @staticmethod
    def eucledian_distance(embeddings: Tensor, squared=False):
        """
        Compute the 2D matrix of eucledian distances between all the embeddings.
        Args:
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            pairwise_distances: tensor of shape (batch_size, batch_size)
        """

        dot_product = torch.matmul(embeddings, embeddings.t())

        # Get squared L2 norm for each embedding. We can just take the diagonal of `dot_product`.
        # This also provides more numerical stability (the diagonal of the result will be exactly 0).
        # shape (batch_size,)
        square_norm = torch.diag(dot_product)

        # Compute the pairwise distance matrix as we have:
        # ||a - b||^2 = ||a||^2  - 2 <a, b> + ||b||^2
        # shape (batch_size, batch_size)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)

        # Because of computation errors, some distances might be negative so we put everything >= 0.0
        distances[distances < 0] = 0

        if not squared:
            # Because the gradient of sqrt is infinite when distances == 0.0 (ex: on the diagonal)
            # we need to add a small epsilon where distances == 0.0
            mask = distances.eq(0).float()
            distances = distances + mask * 1e-16

            distances = (1.0 - mask) * torch.sqrt(distances)

        return distances


class BatchHardTripletLoss(nn.Module):
    def __init__(
        self,
        distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance,
        margin: float = 5,
    ):
        """
        BatchHardTripletLoss takes a batch with (sentence, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. It then looks
        for the hardest positive and the hardest negatives.
        The labels must be integers, with same label indicating sentences from the same class. Your train dataset
        must contain at least 2 examples per label class.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used
            margin: Negative samples should be at least margin further
                apart from the anchor than the positive.

        Definitions:
            :Easy triplets: Triplets which have a loss of 0 because
                ``distance(anchor, positive) + margin < distance(anchor, negative)``.
            :Hard triplets: Triplets where the negative is closer to the anchor than the positive, i.e.,
                ``distance(anchor, negative) < distance(anchor, positive)``.
            :Semi-hard triplets: Triplets where the negative is not closer to the anchor than the positive, but which
                still have a positive loss, i.e., ``distance(anchor, positive) < distance(anchor, negative) + margin``.

        References:
            * Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
            * Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
            * Blog post: https://omoindrot.github.io/triplet-loss

        Requirements:
            1. Each sentence must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.
            3. Your dataset should contain hard positives and negatives.

        Inputs:
            +------------------+--------+
            | Texts            | Labels |
            +==================+========+
            | single sentences | class  |
            +------------------+--------+

        Relations:
            * :class:`BatchAllTripletLoss` uses all possible, valid triplets, rather than only the hardest positive and negative samples.
            * :class:`BatchSemiHardTripletLoss` uses only semi-hard triplets, valid triplets, rather than only the hardest positive and negative samples.
            * :class:`BatchHardSoftMarginTripletLoss` does not require setting a margin, while this loss does.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                # E.g. 0: sports, 1: economy, 2: politics
                train_dataset = Dataset.from_dict({
                    "sentence": [
                        "He played a great game.",
                        "The stock is up 20%",
                        "They won 2-1.",
                        "The last goal was amazing.",
                        "They all voted against the bill.",
                    ],
                    "label": [0, 1, 0, 0, 2],
                })
                loss = losses.BatchHardTripletLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.triplet_margin = margin
        self.distance_metric = distance_metric

    def forward(self, rep : Tensor, labels: Tensor):
        # rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_hard_triplet_loss(labels, rep)

    # Hard Triplet Loss
    # Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    # Blog post: https://omoindrot.github.io/triplet-loss
    def batch_hard_triplet_loss(self, labels: Tensor, embeddings: Tensor) :
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            margin: margin for triplet loss
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss
        tl = hardest_positive_dist - hardest_negative_dist + self.triplet_margin
        tl[tl < 0] = 0
        triplet_loss = tl.mean()

        return triplet_loss

    @staticmethod
    def get_triplet_mask(labels: Tensor) :
        """Return a 3D mask where mask[a, p, n] is True iff the triplet (a, p, n) is valid.
        A triplet (i, j, k) is valid if:
            - i, j, k are distinct
            - labels[i] == labels[j] and labels[i] != labels[k]
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        """
        # Check that i, j and k are distinct
        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal
        i_not_equal_j = indices_not_equal.unsqueeze(2)
        i_not_equal_k = indices_not_equal.unsqueeze(1)
        j_not_equal_k = indices_not_equal.unsqueeze(0)

        distinct_indices = (i_not_equal_j & i_not_equal_k) & j_not_equal_k

        label_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        i_equal_j = label_equal.unsqueeze(2)
        i_equal_k = label_equal.unsqueeze(1)

        valid_labels = ~i_equal_k & i_equal_j

        return valid_labels & distinct_indices

    @staticmethod
    def get_anchor_positive_triplet_mask(labels: Tensor) :
        """Return a 2D mask where mask[a, p] is True iff a and p are distinct and have same label.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check that i and j are distinct

        indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
        indices_not_equal = ~indices_equal

        # Check if labels[i] == labels[j]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)

        return labels_equal & indices_not_equal

    @staticmethod
    def get_anchor_negative_triplet_mask(labels: Tensor) :
        """Return a 2D mask where mask[a, n] is True iff a and n have distinct labels.
        Args:
            labels: tf.int32 `Tensor` with shape [batch_size]
        Returns:
            mask: tf.bool `Tensor` with shape [batch_size, batch_size]
        """
        # Check if labels[i] != labels[k]
        # Uses broadcasting where the 1st argument has shape (1, batch_size) and the 2nd (batch_size, 1)

        return ~(labels.unsqueeze(0) == labels.unsqueeze(1))

class BatchHardSoftMarginTripletLoss(BatchHardTripletLoss):
    def __init__(
        self, distance_metric=BatchHardTripletLossDistanceFunction.eucledian_distance
    ):
        """
        BatchHardSoftMarginTripletLoss takes a batch with (sentence, label) pairs and computes the loss for all possible, valid
        triplets, i.e., anchor and positive must have the same label, anchor and negative a different label. The labels
        must be integers, with same label indicating sentences from the same class. Your train dataset
        must contain at least 2 examples per label class. This soft-margin variant does not require setting a margin.

        Args:
            model: SentenceTransformer model
            distance_metric: Function that returns a distance between
                two embeddings. The class SiameseDistanceMetric contains
                pre-defined metrics that can be used.

        Definitions:
            :Easy triplets: Triplets which have a loss of 0 because
                ``distance(anchor, positive) + margin < distance(anchor, negative)``.
            :Hard triplets: Triplets where the negative is closer to the anchor than the positive, i.e.,
                ``distance(anchor, negative) < distance(anchor, positive)``.
            :Semi-hard triplets: Triplets where the negative is not closer to the anchor than the positive, but which
                still have a positive loss, i.e., ``distance(anchor, positive) < distance(anchor, negative) + margin``.

        References:
            * Source: https://github.com/NegatioN/OnlineMiningTripletLoss/blob/master/online_triplet_loss/losses.py
            * Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
            * Blog post: https://omoindrot.github.io/triplet-loss

        Requirements:
            1. Each sentence must be labeled with a class.
            2. Your dataset must contain at least 2 examples per labels class.
            3. Your dataset should contain hard positives and negatives.

        Relations:
            * :class:`BatchHardTripletLoss` uses a user-specified margin, while this loss does not require setting a margin.

        Inputs:
            +------------------+--------+
            | Texts            | Labels |
            +==================+========+
            | single sentences | class  |
            +------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                # E.g. 0: sports, 1: economy, 2: politics
                train_dataset = Dataset.from_dict({
                    "sentence": [
                        "He played a great game.",
                        "The stock is up 20%",
                        "They won 2-1.",
                        "The last goal was amazing.",
                        "They all voted against the bill.",
                    ],
                    "label": [0, 1, 0, 0, 2],
                })
                loss = losses.BatchHardSoftMarginTripletLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.sentence_embedder = model
        self.distance_metric = distance_metric

    def forward(self, rep : Tensor, labels: Tensor):
        # rep = self.sentence_embedder(sentence_features[0])["sentence_embedding"]
        return self.batch_hard_triplet_soft_margin_loss(labels, rep)

    # Hard Triplet Loss with Soft Margin
    # Paper: In Defense of the Triplet Loss for Person Re-Identification, https://arxiv.org/abs/1703.07737
    def batch_hard_triplet_soft_margin_loss(self, labels: Tensor, embeddings: Tensor):
        """Build the triplet loss over a batch of embeddings.
        For each anchor, we get the hardest positive and hardest negative to form a triplet.
        Args:
            labels: labels of the batch, of size (batch_size,)
            embeddings: tensor of shape (batch_size, embed_dim)
            squared: Boolean. If true, output is the pairwise squared euclidean distance matrix.
                     If false, output is the pairwise euclidean distance matrix.
        Returns:
            Label_Sentence_Triplet: scalar tensor containing the triplet loss
        """
        # Get the pairwise distance matrix
        pairwise_dist = self.distance_metric(embeddings)

        # For each anchor, get the hardest positive
        # First, we need to get a mask for every valid positive (they should have same label)
        mask_anchor_positive = BatchHardTripletLoss.get_anchor_positive_triplet_mask(labels).float()

        # We put to 0 any element where (a, p) is not valid (valid if a != p and label(a) == label(p))
        anchor_positive_dist = mask_anchor_positive * pairwise_dist

        # shape (batch_size, 1)
        hardest_positive_dist, _ = anchor_positive_dist.max(1, keepdim=True)

        # For each anchor, get the hardest negative
        # First, we need to get a mask for every valid negative (they should have different labels)
        mask_anchor_negative = BatchHardTripletLoss.get_anchor_negative_triplet_mask(labels).float()

        # We add the maximum value in each row to the invalid negatives (label(a) == label(n))
        max_anchor_negative_dist, _ = pairwise_dist.max(1, keepdim=True)
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (1.0 - mask_anchor_negative)

        # shape (batch_size,)
        hardest_negative_dist, _ = anchor_negative_dist.min(1, keepdim=True)

        # Combine biggest d(a, p) and smallest d(a, n) into final triplet loss with soft margin
        # tl = hardest_positive_dist - hardest_negative_dist + margin
        # tl[tl < 0] = 0
        tl = torch.log1p(torch.exp(hardest_positive_dist - hardest_negative_dist))
        triplet_loss = tl.mean()

        return triplet_loss


class OnlineContrastiveLoss(nn.Module):
    def __init__(
        self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, margin: float = 0.5
    ) :
        super().__init__()
        self.margin = margin
        self.distance_metric = distance_metric

    def forward(self, rep_des, hidden, labels: Tensor, size_average=False):

        distance_matrix = self.distance_metric(rep_des, hidden)
        negs = distance_matrix[labels == 0]
        poss = distance_matrix[labels == 1]

        # select hard positive and hard negative pairs
        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        positive_loss = positive_pairs.pow(2).sum()
        negative_loss = F.relu(self.margin - negative_pairs).pow(2).sum()
        loss = positive_loss + negative_loss
        # loss = torch.log1p(torch.exp(loss))
        return loss


class InClusterLoss(nn.Module):
    def __init__(
        self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, temperature: float = 0.5
    ) :
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric
    def forward(self, embeddings, labels, labels2cluster, temperature=0.5, same_cluster_weight=0.5):
        """
        Compute the InfoNCE loss with labeled data and cluster-based weighting for negatives.
        
        Args:
            embeddings: torch.Tensor of shape (N, D), where N is the batch size,
                        and D is the dimension of the embeddings.
            labels: torch.Tensor of shape (N,), where each element is the label
                    corresponding to the embedding.
            labels2cluster: dict that maps each label to a cluster.
            temperature: float, the temperature hyperparameter to scale the logits.
            same_cluster_weight: float, the additional weight for negative pairs from the same cluster.
        
        Returns:
            torch.Tensor: the computed InfoNCE loss.
        """
        # print(embeddings.shape, labels.shape, labels2cluster)
        # Normalize embeddings to unit vectors
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute cosine similarity between each pair of embeddings
        similarity_matrix = torch.matmul(embeddings, embeddings.T)
        
        # Scale the similarity matrix by the temperature
        logits = similarity_matrix / temperature
        
        # Mask out self-similarity by setting the diagonal to a large negative value
        logits = logits.fill_diagonal_(float('-inf'))
        
        # Compute a mask where positives have the same label
        labels = labels.unsqueeze(0)  # Shape: (1, N)
        positive_mask = torch.eq(labels, labels.T).float().to(embeddings.device)
        
        # Create cluster-based weights for negatives
        clusters = torch.tensor([labels2cluster[label.item()] for label in labels.squeeze(0)]).to(embeddings.device)
        cluster_mask = torch.eq(clusters.unsqueeze(0), clusters.T).float()
        
        # Weights for negatives in the same cluster
        negative_weights = torch.ones_like(cluster_mask) + (cluster_mask * same_cluster_weight)
        
        # Positive mask should not affect normalization
        # positive_mask /= positive_mask.sum(1, keepdim=True)
        # Compute weighted loss for negatives
        weighted_log_prob = negative_weights * F.log_softmax(logits, dim=1)
        # print(weighted_log_prob)
        # Compute the InfoNCE loss with positive pairs and weighted negatives
        mask = torch.eye(logits.shape[0], dtype=torch.bool).to(embeddings.device)
        positive_mask.masked_fill_(mask, 0)
        weighted_log_prob.masked_fill_(mask, 0)
        loss = - (positive_mask * weighted_log_prob).sum(1).mean()
        
        return loss
    
class ClusterLoss(nn.Module):
    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric

    def forward(self, embeddings, labels, labels2cluster, temperature=0.5, same_cluster_weight=0.5):
        """
        Compute the InfoNCE loss between clusters.
        
        Args:
            embeddings: torch.Tensor of shape (N, D), where N is the batch size,
                        and D is the dimension of the embeddings.
            labels: torch.Tensor of shape (N,), where each element is the label
                    corresponding to the embedding.
            labels2cluster: dict that maps each label to a cluster.
            temperature: float, the temperature hyperparameter to scale the logits.
        
        Returns:
            torch.Tensor: the computed InfoNCE loss.
        """
        if len(set(labels2cluster.values())) == 1:
            return torch.tensor(0.0).to(embeddings.device)
        
        # Group embeddings by their cluster
        cluster2embeddings = defaultdict(list)
        for i, label in enumerate(labels):
            cluster = labels2cluster[label.item()]
            cluster2embeddings[cluster].append(embeddings[i])
        
        # Compute the mean embedding for each cluster
        cluster_embeddings = []
        cluster_labels = []
        for cluster, emb_list in cluster2embeddings.items():
            # Compute the mean of all embeddings in the cluster
            cluster_embedding = torch.mean(torch.stack(emb_list), dim=0)
            cluster_embeddings.append(cluster_embedding)
            cluster_labels.append(cluster)
        
        # Convert to tensors
        cluster_embeddings = torch.stack(cluster_embeddings)  # Shape: (num_clusters, D)
        cluster_labels = torch.tensor(cluster_labels).to(embeddings.device)  # Shape: (num_clusters,)

        #  Normalize the cluster embeddings to unit vectors
        cluster_embeddings = F.normalize(cluster_embeddings, dim=1)

        #  Compute cosine similarity between all pairs of cluster embeddings
        similarity_matrix = torch.matmul(cluster_embeddings, cluster_embeddings.T)
        
        #  Scale the similarity matrix by the temperature
        logits = similarity_matrix / temperature
        
        cluster_labels = cluster_labels.unsqueeze(0)  # Shape: (1, num_clusters)
        positive_mask = torch.eq(cluster_labels, cluster_labels.T).float().to(embeddings.device) # --> Ok with case no positve pair 
        
        mask = torch.eye(logits.shape[0], dtype=torch.bool).to(embeddings.device)
        # logits = logits.masked_fill(mask, float('-inf')) 
    
        log_prob = F.log_softmax(logits, dim=1)

        loss = - ( positive_mask*log_prob).sum(1).mean()
        
        return loss

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric

    def forward(self, embeddings, labels, label_embeddings: dict, temperature=0.5):
        """
        Args:
            embeddings: torch.Tensor of shape (N, D), where N is the batch size,
                        and D is the dimension of the embeddings.
            labels: torch.Tensor of shape (N,), where each element is the label
                    corresponding to the embedding.
            label_embeddings: dict that maps each label to its embedding (e.g., cluster or semantic representations).
            temperature: float, the temperature hyperparameter to scale the logits.
            
        Returns:
            torch.Tensor: the computed weighted contrastive loss.
        """
        N = embeddings.shape[0]
        
        # Step 1: Normalize the embeddings
        embeddings = F.normalize(embeddings, dim=1)  # Normalize to unit vectors

        # Step 2: Compute cosine similarity between all pairs of embeddings (NxN matrix)
        embedding_similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

        # Step 3: Create label embedding matrix and normalize
        label_emb_list = [label_embeddings[label.item()] for label in labels]  # Get label embeddings
        label_emb_matrix = torch.stack(label_emb_list).to(embeddings.device)  # Shape: (N, D_label)
        label_emb_matrix = F.normalize(label_emb_matrix, dim=1)  # Normalize label embeddings

        # Step 4: Compute cosine similarity between all label embeddings (NxN matrix)
        label_similarity_matrix = torch.matmul(label_emb_matrix, label_emb_matrix.T)  # (N, N)

        # Step 5: Scale embedding similarity matrix by temperature
        logits = embedding_similarity_matrix / temperature

        # Step 6: Mask out self-similarity (diagonal)
        mask = torch.eye(N, dtype=torch.bool).to(embeddings.device)
        logits = logits.masked_fill(mask, float('-inf'))

        # Step 7: Create positive and negative masks
        positive_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(embeddings.device)  # Same labels
        positive_mask = positive_mask.masked_fill(mask, 0)  # Zero out self-similarities
        
        # Step 8: Compute the weighted loss
        # Negatives are weighted by the label similarity matrix
        weighted_negatives = (1 - positive_mask) * (1+ label_similarity_matrix)  # Weighting for negatives
        weighted_negatives.masked_fill(positive_mask.bool(), 1)  # 
        log_prob = F.log_softmax(logits, dim=1)  # Log probability over softmax scores
        log_prob_weighted = log_prob * weighted_negatives
        # Compute loss (sum of positives + weighted negatives)
        # loss = - (positive_mask * log_prob).sum(1) - (weighted_negatives * log_prob).sum(1)
        mask = torch.eye(logits.shape[0], dtype=torch.bool).to(embeddings.device)
        log_prob_weighted.masked_fill_(mask, 0)
        loss = - (positive_mask * log_prob_weighted).sum(1).mean()
        loss = loss.mean()  # Final loss

        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, distance_metric=SiameseDistanceMetric.COSINE_DISTANCE, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.distance_metric = distance_metric

    def forward(self, embeddings, labels, temperature=0.05):
        """
        Args:
            embeddings: torch.Tensor of shape (N, D), where N is the batch size,
                        and D is the dimension of the embeddings.
            labels: torch.Tensor of shape (N,N)
            temperature: float, the temperature hyperparameter to scale the logits.
            
        Returns:
            torch.Tensor: the computed contrastive loss.
        """
        N = embeddings.shape[0]
        
        # Step 1: Normalize the embeddings
        embeddings = F.normalize(embeddings, dim=1)  # Normalize to unit vectors

        # Step 2: Compute cosine similarity between all pairs of embeddings (NxN matrix)
        embedding_similarity_matrix = torch.matmul(embeddings, embeddings.T)  # (N, N)

        # Step 3: Scale embedding similarity matrix by temperature
        logits = embedding_similarity_matrix / temperature

        # Step 4: Mask out self-similarity (diagonal)
        mask = torch.eye(N, dtype=torch.bool).to(embeddings.device)
        logits = logits.masked_fill(mask, float('-inf'))

        # Step 5: Create positive and negative masks
        positive_mask = labels.to(embeddings.device)  # Same labels
        positive_mask = positive_mask.masked_fill(mask, 0)  # Zero out self-similarities
        
        # Step 6: Compute the loss
        log_prob = F.log_softmax(logits, dim=1)  # Log probability over softmax scores
        log_prob.masked_fill_(mask, 0)
        loss = - (positive_mask * log_prob).sum(1).mean()  # Compute loss (sum of positives)
        return loss

class MultipleNegativesRankingLoss(nn.Module):
    def __init__(self, scale: float = 20.0, similarity_fct=cos_sim):
        """
        This loss expects as input a batch consisting of sentence pairs ``(a_1, p_1), (a_2, p_2)..., (a_n, p_n)``
        where we assume that ``(a_i, p_i)`` are a positive pair and ``(a_i, p_j)`` for ``i != j`` a negative pair.

        For each ``a_i``, it uses all other ``p_j`` as negative samples, i.e., for ``a_i``, we have 1 positive example
        (``p_i``) and ``n-1`` negative examples (``p_j``). It then minimizes the negative log-likehood for softmax
        normalized scores.

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, relevant_doc)) as it will sample in each batch ``n-1`` negative docs randomly.

        The performance usually increases with increasing batch sizes.

        You can also provide one or multiple hard negatives per anchor-positive pair by structuring the data like this:
        ``(a_1, p_1, n_1), (a_2, p_2, n_2)``. Then, ``n_1`` is a hard negative for ``(a_1, p_1)``. The loss will use for
        the pair ``(a_i, p_i)`` all ``p_j`` for ``j != i`` and all ``n_j`` as negatives.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale
                value
            similarity_fct: similarity function between sentence
                embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://arxiv.org/pdf/1705.00652.pdf
            - `Training Examples > Natural Language Inference <../../examples/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../examples/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../examples/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../examples/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../examples/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../examples/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it requires more
              training time.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Inputs:
            +---------------------------------------+--------+
            | Texts                                 | Labels |
            +=======================================+========+
            | (anchor, positive) pairs              | none   |
            +---------------------------------------+--------+
            | (anchor, positive, negative) triplets | none   |
            +---------------------------------------+--------+

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        # self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, embeddings_a, embeddings_b, labels) -> Tensor:
        scores = self.similarity_fct(embeddings_a, embeddings_b) * self.scale
        return self.cross_entropy_loss(scores, labels)



    
    def get_config_dict(self) -> dict[str, Any]:
        return {"scale": self.scale, "similarity_fct": self.similarity_fct.__name__}



class SiameseDistanceMetric(Enum):
    """The metric for the contrastive loss"""

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    MANHATTAN = lambda x, y: F.pairwise_distance(x, y, p=1)
    COSINE_DISTANCE = lambda x, y: 1 - F.cosine_similarity(x, y)



class SupervisedSimCSELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        """
        Supervised SimCSE loss function.
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings1, embeddings2, labels):
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)
        
        similarity_matrix = torch.matmul(embeddings1, embeddings2.t()) / self.temperature
        
        labels = labels.view(-1, 1)
        positive_mask = (labels == labels.t()).float().to(embeddings1.device)
        
        exp_similarity = torch.exp(similarity_matrix)
        
        pos_sum = torch.sum(exp_similarity * positive_mask, dim=1)
        all_sum = torch.sum(exp_similarity, dim=1)
        
        loss = -torch.log(pos_sum / all_sum)
        return loss.mean()
    
class NegativeCosSimLoss(nn.Module):
    def __init__(self, temperature: float = 2):
        super().__init__()
        self.temperature = temperature
        

    def forward(self, ebd1, ebd2):
        
        # Normalize the embeddings to ensure they are unit vectors
        batch1 = F.normalize(ebd1, p=2, dim=-1)
        batch2 = F.normalize(ebd2, p=2, dim=-1)
        # add noise to the embeddings
        batch1 = batch1 + torch.randn_like(batch1) * 1e-5
        batch2 = batch2 + torch.randn_like(batch2) * 1e-5
        
        # Compute the cosine similarity for each corresponding pair in the batches
        cos_sim = torch.sum(batch1 * batch2, dim=-1) / self.temperature
        
        # Negate the cosine similarity to maximize the distance
        # loss = - torch.log(1 + exp(cos_sim))
        # loss = - torch.log(1 + torch.exp(cos_sim)).mean()
        exp_sim = torch.exp(cos_sim)
        loss = torch.log((1 + exp_sim)).mean()
        return loss