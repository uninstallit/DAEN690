# Matching Prior-Year NOTAM Entries to Space Launch Events with Machine Learning
#### DAEN 690 Summer 2022 
#### Team Bravo Members: Edvin Beqari, Beth Cao, Andrew Fisher, Joelle Thorpe, Brenda Tran
-----

## Project Description
The Systems Analysis and Modeling Division (ANG-B7) quantifies the benefits and tradeoffs of changes to the National Airspace System (NAS) to include, but not limited to, modernization plants and integration of new entrants. ANG-B7 performs their analyses using data-driven techniques and full day agent-based simulations. 

To prepare their simulations for incorporating traffic from space vehicle operations, they have collected a set of historical space launches and historical airspace closure data. The historical launch data was pulled from various online sources and the historical airspace closure were drawn from published Notices to Air Missions (NOTAMs) The Systems Analysis and Modeling Division have found it challenging to match the NOTAMs entries with the launch operations. The current process is done based on a manual filtering of the NOTAMs based on features drawn from them (e.g., data, location, facility). Using this manual process, they are only fully confident of 60 launches associated to NOTAMs (out of a total of more than 200 launches).  

This project attempts to transform the current manual matching process to an automated process using machine learning, using Siamese networks and NLP to match launches to associated NOTAM entries.  Modeling efforts include the use of Siamese Networks to train the model based on a “Good training dataset (AKA NOTAMs associated to a Launch) and a “Bad” training dataset (AKA NOTAMs not associated to a launch), and semantic search as a form of natural language processing (NLP) to find similarities of other NOTAMs entries based on the e-code text.
