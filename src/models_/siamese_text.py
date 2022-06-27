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


# source: https://keras.io/examples/vision/siamese_network/


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


def get_base_network(input_shape):
    embedding_inputs = tf.keras.Input(shape=input_shape, name="text")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(
        embedding_inputs
    )
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64))(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    modifier_outputs = tf.keras.layers.Dense(128, activation="relu")(x)
    modifier_model = tf.keras.Model(embedding_inputs, modifier_outputs)
    return modifier_model


def get_siamese_network(input_shape, base_model):
    anchor_input = tf.keras.layers.Input(name="anchor", shape=input_shape)
    positive_input = tf.keras.layers.Input(name="positive", shape=input_shape)
    negative_input = tf.keras.layers.Input(name="negative", shape=input_shape)
    distances = DistanceLayer()(
        base_model(anchor_input),
        base_model(positive_input),
        base_model(negative_input),
    )
    siamese_network = tf.keras.Model(
        inputs=[anchor_input, positive_input, negative_input],
        outputs=distances,
        name="siamese",
    )
    return siamese_network


def main():

    anchor_data = np.load("./data/anchor_data.npy", allow_pickle=True)
    positive_data = np.load("./data/positive_data.npy", allow_pickle=True)
    negative_data = np.load("./data/negative_data.npy", allow_pickle=True)

    anchor_embeddings = np.expand_dims(fromBuffer(anchor_data[:, 8]), -1)
    positive_embeddings = np.expand_dims(fromBuffer(positive_data[:, 8]), -1)
    negative_embeddings = np.expand_dims(fromBuffer(negative_data[:, 8]), -1)

    anchor_embd_dataset = tf.data.Dataset.from_tensor_slices(anchor_embeddings)
    positive_embd_dataset = tf.data.Dataset.from_tensor_slices(positive_embeddings)
    negative_embd_dataset = tf.data.Dataset.from_tensor_slices(negative_embeddings)

    dataset = tf.data.Dataset.zip((anchor_embd_dataset, positive_embd_dataset, negative_embd_dataset))
    dataset = dataset.shuffle(buffer_size=4096)

    train_dataset = dataset.take(round(len(anchor_embeddings) * 0.8))
    val_dataset = dataset.skip(round(len(anchor_embeddings) * 0.8))

    train_dataset = train_dataset.batch(32, drop_remainder=False)
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)

    val_dataset = val_dataset.batch(32, drop_remainder=False)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)

    input_shape = (384, 1)
    base_network = get_base_network(input_shape)
    siamese_network = get_siamese_network(input_shape, base_network)

    siamese_model = SiameseModel(siamese_network)
    siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), weighted_metrics=[])
    history = siamese_model.fit(train_dataset, epochs=30, validation_data=val_dataset)

    # *** inference ***

    anch = np.expand_dims(positive_embeddings[4], 0)
    other = np.expand_dims(negative_embeddings[4], 0)

    anch_prediction = base_network.predict(anch)
    other_prediction = base_network.predict(other)

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
