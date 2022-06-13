import copy
import random
from itertools import permutations


def get_matches_index_dict(matches_df, launches_df):
    index = launches_df["LAUNCHES_REC_ID"].apply(
        lambda x: matches_df[matches_df["LAUNCHES_REC_ID"] == x][
            "NOTAM_REC_ID"
        ].tolist()
    )
    index_dict = dict(
        (key, val) for (key, val) in index.to_dict().items() if len(val) != 0
    )
    return index_dict


def get_triplet_index_dict(matches_df, launches_df):
    anchor_index = []
    positive_index = []
    negative_index = []
    matches_dict = get_matches_index_dict(matches_df, launches_df)
    matches_dict_keys = list(matches_dict.keys())

    for key, values in matches_dict.items():
        temp_keys = copy.deepcopy(matches_dict_keys)
        temp_keys.remove(key)
        perms = permutations(values, 2)

        for pair in perms:
            rand_key = random.choice(temp_keys)
            temp_values = matches_dict[rand_key]
            anchor, positive = pair
            negative = random.choice(temp_values)
            anchor_index.append(anchor)
            positive_index.append(positive)
            negative_index.append(negative)
    return (anchor_index, positive_index, negative_index)
