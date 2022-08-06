# Matching Prior-Year NOTAM Entries to Space Launch Events with Machine Learning
#### DAEN 690 Summer 2022 
#### Team Bravo Members: Edvin Beqari, Beth Cao, Andrew Fisher, Joelle Thorpe, Brenda Tran
-----

## Project Description
The Federal Aviation Administration (FAA) is responsible for regulating and providing safe and efficient air navigation services in the United States. Notice to Air Missions (NOTAM) are required notices by both the International Civil Aviation Organization and FAA of unusual air space conditions found in the National Airspace System. The Systems Analysis and Modeling Division within the FAA have found it challenging to match rocket launch events with related NOTAMs. The current matching process is executed by manually filtering the NOTAMs.

This project attempts to improve the lexical search process with semantic search, by leveraging the pre-trained Bidirectional Encoder Representations from Transformers (BERT) Sentence Similarity Model, to understand context and find synonyms. We use transfer learning to build upon the BERT model to fine tune two additional training models, used to distinguish NOTAMS associated to either a launch or no-launch class. The first training model uses text embeddings as inputs and is trained with a triplet loss function with samples comprised of hand-selected launch or no-launch classes. Similarly, the second model is not only trained using text embeddings, but also with numerical and categorical features derived from the provided data. This model is trained with a quadruplet loss which differentiates between launch or no-launch classes as well as cross launch events provided by FAA. This approach offers a unique ability to train from a significantly small and imbalanced training set. The results from the three-model implementations show strong consistency on the top results as well as accurate and interpretable scores based on each model assumption. Our solution lays the foundation for an automated, versatile machine learning workflow to identify and match NOTAMs issued for space launch events and other correlated NOTAMs. 

<!--
## Presentation:
A full Presentation of the Project is found in [here]
-->

<!--(ADD URL HERE)
-->
