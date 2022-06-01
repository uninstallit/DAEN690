# from tokenizers import BertWordPieceTokenizer
# tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
from tokenizers import Tokenizer
# tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
tokenizer = Tokenizer.from_pretrained("albert-base-v2")

# *** Tokenizer algorithm ***
# ‘WLV’ - Word Level Algorithm
# ‘WPC’ - WordPiece Algorithm
# ‘BPE’ - Byte Pair Encoding
# ‘UNI’ - Unigram

# *** TODO ***
# Byte Pair Encoding
# <MASK>
# source: https://www.freecodecamp.org/news/train-algorithms-from-scratch-with-hugging-face/

output = tokenizer.encode("Hello, y'all! How are you 😁 ?")
print(output.tokens)
print(output.ids)
print(tokenizer.token_to_id("[SEP]"))

# *** pipeline ***
# - normalization
# - pre-tokenization
# - model
# - post-processing

# https://www.microsoft.com/en-us/research/blog/less-pain-more-gain-a-simple-method-for-vae-training-with-less-of-that-kl-vanishing-agony/