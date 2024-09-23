import random
import numpy as np
from itertools import combinations
from collections import defaultdict
random.seed(42)


# def mixup_data_augmentation(sample_list, sep_token_id=102, pad_token_id=0, max_len=512):
#     mixed_samples = []
    
#     # Iterate over all possible pairs of samples
#     for sample1, sample2 in combinations(sample_list, 2):
#         # Only mix if relations are different
#         if sample1['relation'] != sample2['relation']:
#             mixed_sample = mixup_samples(sample1, sample2, sep_token_id, pad_token_id, max_len)
#             if mixed_sample is not None:
#                 mixed_samples.append(mixed_sample)
    
#     return mixed_samples


def mixup_data_augmentation(sample_list, sep_token_id=102, pad_token_id=0, max_len=512):
    # random shuffle the sample_list
    random.shuffle(sample_list)
    mixed_samples = []
    sample_count = defaultdict(int)
    
    # Iterate over all possible pairs of samples
    for sample1, sample2 in combinations(sample_list, 2):
        # Only mix if relations are different and both samples have been used less than 2 times
        if (sample1['relation'] != sample2['relation'] and 
            sample_count[id(sample1)] < 4 and 
            sample_count[id(sample2)] < 4):
            
            mixed_sample = mixup_samples(sample1, sample2, sep_token_id, pad_token_id, max_len)
            if mixed_sample is not None:
                mixed_samples.append(mixed_sample)
                sample_count[id(sample1)] += 1
                sample_count[id(sample2)] += 1
    
    return mixed_samples

# Function to merge two samples, as discussed before
def mixup_samples(sample1, sample2, sep_token_id=102, pad_token_id=0, max_len=512):
    # Remove padding based on the mask from both samples
    ids1 = [idx for idx in sample1['ids'] if idx != 0]
    ids2 = [idx for idx in sample2['ids'] if idx != 0]
    
    # Merge ids with a [SEP] token between them
    # merged_ids = ids1[:-1] + [sep_token_id] + ids2[1:]  # Remove [SEP] from ids1 and first token from ids2
    merged_ids = ids1 + ids2  # Remove [SEP] from ids1 and first token from ids2
    
    # Create new mask (1s for actual tokens, 0s for padding)
    merged_mask = [1] * len(merged_ids)
    
    # Padding if necessary to max_len
    if len(merged_ids) < max_len:
        padding_length = max_len - len(merged_ids)
        merged_ids += [pad_token_id] * padding_length
        merged_mask += [0] * padding_length
    else:
        print("Truncated")
        merged_ids = merged_ids[:max_len]
        merged_mask = merged_mask[:max_len]
    
    # Create a new label as a list of both relations
    merged_label = [sample1['relation'], sample2['relation']]
    neg_labels = [sample1['neg_labels'], sample2['neg_labels']]
    return {
        'ids': merged_ids,
        'mask': merged_mask,
        'relation': merged_label,
        'neg_labels': neg_labels
    }

# Example usage with a list of samples
# sample_list = [
#     {'relation': 6, 'index': 1, 'ids': [101, 2132, 1997, 1996,103, 9593,102, 0, 0], 'mask': np.array([1, 1, 1, 1, 1,  1,1, 0, 0])},
#     {'relation': 4, 'index': 2, 'ids': [101, 2197, 2005, 2023,103, 1010, 102, 0, 0], 'mask': np.array([1, 1, 1, 1, 1,  1,1, 0, 0])},
#     {'relation': 5, 'index': 3, 'ids': [101, 2342, 5678, 1234,103, 4321, 102, 0, 0], 'mask': np.array([1, 1, 1, 1, 1,  1,1, 0, 0])},
#     {'relation': 3, 'index': 4, 'ids': [101, 6789, 8765, 4321,103, 1234, 102, 0, 0], 'mask': np.array([1, 1, 1, 1, 1,  1,1, 0, 0])}
# ]

# mixed_samples = mixup_data_augmentation(sample_list)
# for mixed_sample in mixed_samples:
#     print(mixed_sample)
