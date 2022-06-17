import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

current = os.path.dirname(os.path.realpath(__file__))

parent = os.path.dirname(current)
sys.path.append(parent)

root = os.path.dirname(parent)
sys.path.append(root)

from functions_.functions import get_triplet_index_dict


class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.
    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def _compute_loss(self, data):
        ap_distance, an_distance = self.siamese_network(data)
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        return loss

    def train_step(self, data):
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}


class DistanceLayer(tf.keras.layers.Layer):
    """
    This layer is responsible for computing the distance between the anchor
    embedding and the positive embedding, and the anchor embedding and the
    negative embedding.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, anchor, positive, negative):
        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_base_network(mixed_input_shape, embedding_input_shape):

    # mixed data
    mixed_inputs = tf.keras.Input(shape=mixed_input_shape, name="mixed")
    x = tf.keras.layers.Dense(128, activation="relu")(mixed_inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    mixed_outputs = tf.keras.layers.Dense(128, activation="relu")(x)

    # embeddings
    embedding_inputs = tf.keras.Input(shape=embedding_input_shape, name="text")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(
        embedding_inputs
    )
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    embedding_outputs = tf.keras.layers.Dense(128, activation="relu")(x)

    # combine branches
    concat = tf.keras.layers.concatenate([mixed_outputs, embedding_outputs])
    x = tf.keras.layers.Dense(128, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(128, activation="linear")(x)

    # mixed model
    base_model = tf.keras.Model(
        [mixed_inputs, embedding_inputs], outputs, name="base_model"
    )
    return base_model


def get_siamese_network(inputs_shape, base_model):
    mixed_data_shape, embeddings_shape = inputs_shape

    anchor_data_input = tf.keras.layers.Input(
        name="anchor_data", shape=mixed_data_shape
    )
    positive_data_input = tf.keras.layers.Input(
        name="positive_data", shape=mixed_data_shape
    )
    negative_data_input = tf.keras.layers.Input(
        name="negative_data", shape=mixed_data_shape
    )

    anchor_embd_input = tf.keras.layers.Input(
        name="anchor_embd", shape=embeddings_shape
    )
    positive_embd_input = tf.keras.layers.Input(
        name="positive_embd", shape=embeddings_shape
    )
    negative_embd_input = tf.keras.layers.Input(
        name="negative_embd", shape=embeddings_shape
    )

    distances = DistanceLayer()(
        base_model([anchor_data_input, anchor_embd_input]),
        base_model([positive_data_input, positive_embd_input]),
        base_model([negative_data_input, negative_embd_input]),
    )

    siamese_network = tf.keras.Model(
        inputs=[
            anchor_data_input,
            positive_data_input,
            negative_data_input,
            anchor_embd_input,
            positive_embd_input,
            negative_embd_input,
        ],
        outputs=distances,
        name="siamese",
    )
    return siamese_network


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

    anchor_data_dataset = tf.data.Dataset.from_tensor_slices(anchor_data)
    positive_data_dataset = tf.data.Dataset.from_tensor_slices(positive_data)
    negative_data_dataset = tf.data.Dataset.from_tensor_slices(negative_data)

    anchor_embd_dataset = tf.data.Dataset.from_tensor_slices(anchor_embeddings)
    positive_embd_dataset = tf.data.Dataset.from_tensor_slices(positive_embeddings)
    negative_embd_dataset = tf.data.Dataset.from_tensor_slices(negative_embeddings)

    dataset = tf.data.Dataset.zip(
        (
            anchor_data_dataset,
            positive_data_dataset,
            negative_data_dataset,
            anchor_embd_dataset,
            positive_embd_dataset,
            negative_embd_dataset,
        )
    )
    dataset = dataset.shuffle(buffer_size=4096)
    assert len(anchor_data) == len(anchor_embeddings)

    train_dataset = dataset.take(round(len(anchor_data) * 0.25))
    val_dataset = dataset.skip(round(len(anchor_data) * 0.75))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    mixed_data_input_shape = (6,)
    embeddings_input_shape = (384, 1)

    base_network = get_base_network(mixed_data_input_shape, embeddings_input_shape)
    siamese_network = get_siamese_network(
        [mixed_data_input_shape, embeddings_input_shape], base_network
    )

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
    history = siamese_model.fit(train_dataset, epochs=15, validation_data=val_dataset)

    base_network.save(root + "/src/saved_models_/test_model")

    # *** inference ***

    anch_data = np.expand_dims(positive_data[4], 0)
    other_data = np.expand_dims(negative_data[4], 0)

    anch_embd = np.expand_dims(positive_embeddings[4], 0)
    other_embd = np.expand_dims(negative_embeddings[4], 0)

    anch_prediction = base_network.predict([anch_data, anch_embd])
    other_prediction = base_network.predict([other_data, other_embd])

    cosine_similarity = tf.keras.metrics.CosineSimilarity()
    positive_similarity = cosine_similarity(anch_prediction, anch_prediction)
    print("Positive similarity:", positive_similarity.numpy())

    negative_similarity = cosine_similarity(anch_prediction, other_prediction)
    print("Negative similarity", negative_similarity.numpy())

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss History")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
