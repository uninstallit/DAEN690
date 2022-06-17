import tensorflow as tf
import numpy as np

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from functions_.functions import get_triplet_index_dict


def main():
    (anchor_index, positive_index, negative_index) = get_triplet_index_dict()

    notams_data = np.load("./data/notams_data.npy", allow_pickle=True)[:, 1:]
    anchor_data = np.take(notams_data, anchor_index, axis=0).astype("float32")
    positive_data = np.take(notams_data, positive_index, axis=0).astype("float32")
    negative_data = np.take(notams_data, negative_index, axis=0).astype("float32")

    notams_embeddings = np.load("./data/notams_embeddings.npy", allow_pickle=True)
    anchor_embeddings = np.expand_dims(
        np.take(notams_embeddings, anchor_index, axis=0), -1
    )
    positive_embeddings = np.expand_dims(
        np.take(notams_embeddings, positive_index, axis=0), -1
    )
    negative_embeddings = np.expand_dims(
        np.take(notams_embeddings, negative_index, axis=0), -1
    )

    base_network = tf.keras.models.load_model(root + "/src/saved_models_/sm1_model")

    anch_index = 23
    pos_index = 32
    neg_index = 14

    anch_data = np.expand_dims(anchor_data[anch_index], 0)
    pos_data = np.expand_dims(positive_data[pos_index], 0)
    other_data = np.expand_dims(negative_data[neg_index], 0)

    anch_embd = np.expand_dims(anchor_embeddings[anch_index], 0)
    pos_embd = np.expand_dims(positive_embeddings[pos_index], 0)
    other_embd = np.expand_dims(negative_embeddings[neg_index], 0)

    anch_prediction = base_network.predict([anch_data, anch_embd])
    pos_prediction = base_network.predict([pos_data, pos_embd])
    other_prediction = base_network.predict([other_data, other_embd])

    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    positive_similarity = cosine_similarity(anch_prediction, pos_prediction)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anch_prediction, other_prediction)
    print("Negative similarity", negative_similarity.numpy())


if __name__ == "__main__":
    main()
