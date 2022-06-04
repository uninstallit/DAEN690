import sqlite3
import torch
from sentence_transformers import SentenceTransformer, LoggingHandler
from sentence_transformers import models, util, datasets, evaluation, losses
from torch.utils.data import DataLoader

import nltk

# nltk.download("punkt")


def main():

    conn = sqlite3.Connection("./data/svo_db_20200901.db")
    cursor = conn.cursor()

    # model_name = "bert-base-uncased"
    # word_embedding_model = models.Transformer(model_name)
    # pooling_model = models.Pooling(
    #     word_embedding_model.get_word_embedding_dimension(), "cls"
    # )
    # model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # # Define a list with sentences (1k - 100k sentences)
    # train_sentences = [
    #     "Your set of sentences",
    #     "Model will automatically add the noise",
    #     "And re-construct it",
    #     "You should provide at least 1k sentences",
    # ]

    # # Create the special denoising dataset that adds noise on-the-fly
    # train_dataset = datasets.DenoisingAutoEncoderDataset(train_sentences)

    # # DataLoader to batch your data
    # train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # # Use the denoising auto-encoder loss
    # train_loss = losses.DenoisingAutoEncoderLoss(
    #     model, decoder_name_or_path=model_name, tie_encoder_decoder=True
    # )

    # # Call the fit method
    # model.fit(
    #     train_objectives=[(train_dataloader, train_loss)],
    #     epochs=10,
    #     weight_decay=0,
    #     scheduler="constantlr",
    #     optimizer_params={"lr": 3e-5},
    #     show_progress_bar=True,
    # )

    # # path = "./src/output/tsdae-model"
    # # model.save(path)

    # # # Model class must be defined somewhere
    # # model = torch.load(path)
    # # model.eval()

    # corpus = ['A man is eating food.']

    # corpus_embeddings = model.encode(corpus, convert_to_tensor=True)
    # print(corpus_embeddings.shape)

# sql = """ SELECT DISTINCT facility_id FROM tbl_facility_information WHERE facility_type == 3"""


if __name__ == "__main__":
    main()
