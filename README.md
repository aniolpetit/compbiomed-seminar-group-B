# compbiomed-seminar
In this repository you can find all necessary files to train the final model used for both classifications tasks (transfer learning using HuBERT ECG model).
In order to do it you should have the data files: 'all_points_may_2024-001.pkl' and 'labels_FontiersUnsupervised.xlsx'.
1- Run `build_ecg_dataset.py` to create the dataframe that'll be used. 
2- Run `hubertPreProcess_Fast.py` to obtained the processed dataframe.
3- Run `train_hubert_side.py` and `train_hubert_simplified.py` to run both classifications and get the results.
