# TODO

# source: https://www.codegrepper.com/code-examples/python/sentence+transformers

"""
This is a simple application for sentence embeddings: semantic search

We have a corpus with various sentences. Then, for a given query sentence,
we want to find the most similar sentence in this corpus.

This script outputs for various queries the top 5 most similar sentences in the corpus.
"""
import sqlite3
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
root = os.path.dirname(parent)
sys.path.append(parent)  # parent = src
from pipelines_.pipelines import clean_column_text_pipeline

# 'bert-base-nli-mean-tokens'
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Find the closest 5 sentences of the corpus for each query sentence based on cosine similarity
def semantic_search_similar_notams(corpus_df, queries_df):
    print(f'semantic_search')

    corpus = corpus_df['TEXT'].to_numpy()
    notam_rec_ids = corpus_df['NOTAM_REC_ID'].to_numpy()
    corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

    top_k = min(10, len(corpus))
    for i in queries_df.index:
        q_notam_rec_id = queries_df['NOTAM_REC_ID'][i]
        q_notam_txt =  queries_df['INPUT_QUERY'][i]
        query_embedding = embedder.encode(q_notam_txt, convert_to_tensor=True)

        # We use cosine-similarity and torch.topk to find the highest 5 scores
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)

        print("\n======================")
        #print("Query:", query)
        print(f'\nGiven NOTAM_REC_ID:{q_notam_rec_id} - Query top {top_k} most similar sentences in corpus:')
        for score, idx in zip(top_results[0], top_results[1]):
            print(f'\n notam_rec_id:{notam_rec_ids[idx]}',  '%.110s...' % corpus[idx], "(Score: {:.4f})".format(score))

def main():
    conn = sqlite3.Connection("./data/svo_db_20201027.db")
    cursor = conn.cursor()

    # launch_id 391 time is 2018-04-02 20:30:38
    sql = """ SELECT NOTAM_REC_ID, E_CODE, TEXT FROM notams WHERE 
                DATETIME(notams.POSSIBLE_START_DATE) < '2018-04-02 20:30:38' and DATETIME(notams.POSSIBLE_END_DATE) = '2018-04-02 21:08:00' 
                and (E_CODE not null or TEXT not null) """
    data = cursor.execute(sql).fetchall()
    # take E code col if not take Text column
    notam_df = pd.DataFrame({ 'NOTAM_REC_ID': [d[0] for d in data], 'TEXT': [d[1] if d[1]  else d[2] for d in data]})
    notam_df = notam_df.dropna()
    print(f'FOUND notams: {len(notam_df)}')
    # corpus 
    corpus = clean_column_text_pipeline("TEXT").fit_transform(notam_df)
    corpus = np.squeeze(corpus, axis=1)
    corpus_df = pd.DataFrame( {'NOTAM_REC_ID': notam_df['NOTAM_REC_ID'].to_numpy(), 'TEXT': corpus})
    
    #query 
    # starting on a given found NOTAM that matched with launch_id = 391
    query_text = ['FL..AIRSPACE CAPE CANAVERAL FL..TEMPORARY FLT  RESTRICTION. PURSUANT TO 14 CFR SECTION 91.143, FLT LIMITATION IN  THE PROXIMITY OF SPACE FLT OPS, OPS BY FAA CERT PILOTS OR U.S.REG ACFT ARE PROHIBITED WI AN AREA DEFINED AS 285116N0804219W (OMN141034.4) TO 290730N0803000W (OMN108033.9) THEN CLOCKWISE VIA A  30 NM ARC CENTERED AT 283703N0803647W (OMN147048.7) TO 281330N0801600W (OMN145078.4) TO 282501N0803029W (OMN149061.9) TO 282501N0803759W (OMN155058.8) TO 282501N0804144W (OMN157057.4) TO 283121N0804349W (OMN157050.9) TO 283801N0804701W (OMN157043.7) TO 284910N0805044W (OMN154032.2) TO 285116N0804714W (OMN148031.8) TO POINT OF ORIGIN. SFC-FL180. MIAMI / ZMA / ARTCC, PHONE 305-716-1589, IS THE FAA COORDINATION FACILITY. THIS AREA ENCOMPASSES R2932, R2933, R2934, AND PORTIONS OF W137F, W137G, W497A. ADDITIONAL WARNING AND RESTRICTED AREAS WILL BE ACT. PILOTS MUST CONSULT ALL NOTAMS REGARDING THIS OP AND MAY CONTACT ZMA FOR CURRENT AIRSPACE STATUS.']
    query_df = pd.DataFrame({'NOTAM_REC_ID': 835580, 'INPUT_QUERY': query_text})
    clean_queries = clean_column_text_pipeline("INPUT_QUERY").fit_transform(query_df)
    clean_queries = np.squeeze(clean_queries, axis=1)
    query_df['INPUT_QUERY'] = clean_queries
    
    semantic_search_similar_notams(corpus_df, query_df)
  

if __name__ == "__main__":
    main()