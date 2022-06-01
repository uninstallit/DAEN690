# from transformers import BertModel

# model = BertModel.from_pretrained("bert-base-uncased")
# embedding_matrix = model.embeddings.word_embeddings.weight

# print(model.embeddings.word_embeddings)

# good reference
# https://stackoverflow.com/questions/63461262/bert-sentence-embeddings-from-transformers

import tensorflow as tf
from transformers import BertTokenizer, TFBertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

input_ids = tf.constant(tokenizer.encode("Brenda is my friend"))[None, :]  # Batch size 1
outputs = model(input_ids)

last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
print(input_ids)
print(last_hidden_states)

# siamese
# https://towardsdatascience.com/siamese-nn-recipes-with-keras-72f6a26deb64

# https://github.com/MichelDeudon/variational-siamese-network