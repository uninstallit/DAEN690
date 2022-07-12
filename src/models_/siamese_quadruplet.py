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

from functions_.functions import fromBuffer


class SiameseModel(tf.keras.Model):
    """The Siamese Network model with a custom training and testing loops.
    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.
    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=1.0):
        super(SiameseModel, self).__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def _compute_loss(self, data):
        ap_distance, an_distance_one, an_distance_two = self.siamese_network(data)
        loss_one = tf.maximum(ap_distance - an_distance_one + 0.5 * self.margin, 0.0)
        loss_two = tf.maximum(ap_distance - an_distance_two + self.margin, 0.0)
        loss = loss_one + loss_two
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
        self.loss = tf.keras.losses.CosineSimilarity(axis=1)

    def call(self, anchor, positive, negative_one, negative_two):
        ap_distance = self.loss(anchor, positive)
        an_distance_one = self.loss(anchor, negative_one)
        an_distance_two = self.loss(anchor, negative_two)
        return (ap_distance, an_distance_one, an_distance_two)


def get_base_network(mixed_input_shape, embedding_input_shape):
    # mixed data
    mixed_inputs = tf.keras.Input(shape=mixed_input_shape, name="mixed")
    x = tf.keras.layers.Dense(64, activation="relu")(mixed_inputs)
    x = tf.keras.layers.Dropout(0.5)(x)
    mixed_outputs = tf.keras.layers.Dense(384, activation="relu")(x)

    # embeddings
    embedding_inputs = tf.keras.Input(shape=embedding_input_shape, name="text")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(
        embedding_inputs
    )
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    embedding_outputs = tf.keras.layers.Dense(384, activation="linear")(x)

    # combine branches
    concat = tf.keras.layers.concatenate([mixed_outputs, embedding_outputs])
    x = tf.keras.layers.Dense(384, activation="relu")(concat)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(384, activation="linear")(x)
    # outputs = tf.keras.layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)

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
    negative_one_data_input = tf.keras.layers.Input(
        name="negative_data_one", shape=mixed_data_shape
    )
    negative_two_data_input = tf.keras.layers.Input(
        name="negative_data_two", shape=mixed_data_shape
    )

    anchor_embd_input = tf.keras.layers.Input(
        name="anchor_embd", shape=embeddings_shape
    )
    positive_embd_input = tf.keras.layers.Input(
        name="positive_embd", shape=embeddings_shape
    )
    negative_one_embd_input = tf.keras.layers.Input(
        name="negative_embd_one", shape=embeddings_shape
    )
    negative_two_embd_input = tf.keras.layers.Input(
        name="negative_embd_two", shape=embeddings_shape
    )

    distances = DistanceLayer()(
        base_model([anchor_data_input, anchor_embd_input]),
        base_model([positive_data_input, positive_embd_input]),
        base_model([negative_one_data_input, negative_one_embd_input]),
        base_model([negative_two_data_input, negative_two_embd_input]),
    )

    siamese_network = tf.keras.Model(
        inputs=[
            anchor_data_input,
            positive_data_input,
            negative_one_data_input,
            negative_two_data_input,
            anchor_embd_input,
            positive_embd_input,
            negative_one_embd_input,
            negative_two_embd_input,
        ],
        outputs=distances,
        name="siamese",
    )
    return siamese_network


def main():

    anchor_data = np.load("./data/anchor_data.npy", allow_pickle=True)
    positive_data = np.load("./data/positive_data.npy", allow_pickle=True)
    negative_one_data = np.load("./data/negative_one_data.npy", allow_pickle=True)
    negative_two_data = np.load("./data/negative_two_data.npy", allow_pickle=True)

    anchor_embeddings = np.expand_dims(fromBuffer(anchor_data[:, 8]), -1)
    positive_embeddings = np.expand_dims(fromBuffer(positive_data[:, 8]), -1)
    negative_one_embeddings = np.expand_dims(fromBuffer(negative_one_data[:, 8]), -1)
    negative_two_embeddings = np.expand_dims(fromBuffer(negative_two_data[:, 8]), -1)

    # drop id, text, and byte array
    anchor_data = anchor_data[:, 4:-1].astype("float32")
    positive_data = positive_data[:, 4:-1].astype("float32")
    negative_one_data = negative_one_data[:, 4:-1].astype("float32")
    negative_two_data = negative_two_data[:, 4:-1].astype("float32")

    anchor_data_dataset = tf.data.Dataset.from_tensor_slices(anchor_data)
    positive_data_dataset = tf.data.Dataset.from_tensor_slices(positive_data)
    negative_one_data_dataset = tf.data.Dataset.from_tensor_slices(negative_one_data)
    negative_two_data_dataset = tf.data.Dataset.from_tensor_slices(negative_two_data)

    anchor_embd_dataset = tf.data.Dataset.from_tensor_slices(anchor_embeddings)
    positive_embd_dataset = tf.data.Dataset.from_tensor_slices(positive_embeddings)
    negative_one_embd_dataset = tf.data.Dataset.from_tensor_slices(
        negative_one_embeddings
    )
    negative_two_embd_dataset = tf.data.Dataset.from_tensor_slices(
        negative_two_embeddings
    )

    dataset = tf.data.Dataset.zip(
        (
            anchor_data_dataset,
            positive_data_dataset,
            negative_one_data_dataset,
            negative_two_data_dataset,
            anchor_embd_dataset,
            positive_embd_dataset,
            negative_one_embd_dataset,
            negative_two_embd_dataset,
        )
    )
    dataset = dataset.shuffle(buffer_size=4096)
    assert len(anchor_data) == len(anchor_embeddings)

    train_dataset = dataset.take(round(len(anchor_data) * 0.5))
    val_dataset = dataset.skip(round(len(anchor_data) * 0.5))

    train_dataset = train_dataset.batch(128, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(128, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    mixed_data_input_shape = (4,)
    embeddings_input_shape = (384, 1)

    base_network = get_base_network(mixed_data_input_shape, embeddings_input_shape)
    siamese_network = get_siamese_network(
        [mixed_data_input_shape, embeddings_input_shape], base_network
    )

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001))
    history = siamese_model.fit(train_dataset, epochs=125, validation_data=val_dataset)

    base_network.save(root + "/src/saved_models_/qsmy_model_125")

    # *** inference ***

    anch_data = np.expand_dims(positive_data[4], 0)
    other_data = np.expand_dims(negative_one_data[4], 0)

    anch_embd = np.expand_dims(positive_embeddings[4], 0)
    other_embd = np.expand_dims(negative_one_embeddings[4], 0)

    anch_prediction = base_network.predict([anch_data, anch_embd])
    other_prediction = base_network.predict([other_data, other_embd])

    cosine_similarity = tf.keras.metrics.CosineSimilarity()

    cosine_similarity.reset_state()
    cosine_similarity.update_state(anch_prediction, anch_prediction)
    print("Positive similarity:", cosine_similarity.result().numpy())

    cosine_similarity.reset_state()
    cosine_similarity.update_state(anch_prediction, other_prediction)
    print("Negative similarity", cosine_similarity.result().numpy())

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Loss History")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "val"], loc="upper left")
    plt.show()


if __name__ == "__main__":
    main()
