# speech-emotion-recognition

## Overview
Identifying emotion from speech is a non-trivial task pertaining to the ambiguous definition of emotion itself. In this work, we formalize compare performance of two categories of models. For both types of models, we extract some hand-crafted features from a given audio signal.

The hand-crafted feature vectors obtained are used to train two types of models:

1. ML-based: Logistic Regression, SVMs, Random Forest, eXtreme Gradient Boosting and Multinomial Naive-Bayes.
2. DL-based: Multi-Layer Perceptron, LSTM Classifier

This project was carried as a course project for the course CS 698 - Computational Audio taught by [Prof. Richard Mann](https://cs.uwaterloo.ca/~mannr/) at the University of Waterloo. For a more detailed explanation, please check the [report](https://arxiv.org/abs/1904.06022).

## Datasets
The [IEMOCAP](https://link.springer.com/content/pdf/10.1007%2Fs10579-008-9076-6.pdf) dataset was used for all the experiments in this work. Please refer to the [report](https://arxiv.org/abs/1904.06022) for a detailed explanation of pre-processing steps applied to the dataset.

## Requirements
All the experiments have been tested using the following libraries:
- xgboost==0.82
- torch==1.0.1.post2
- scikit-learn==0.20.3
- numpy==1.16.2
- jupyter==1.0.0
- pandas==0.24.1
- librosa==0.6.3

## Instructions
1. Run `1_extract_emotion_labels.ipynb` to extract labels from transriptions and compile other required data into a csv.
2. Run `2_build_audio_vectors.ipynb` to build vectors from the original wav files and save into a pickle file
3. Run `3_extract_audio_features.ipynb` to extract 8-dimensional audio feature vectors for the audio vectors
4. Run `4_prepare_data.ipynb` to preprocess and prepare audio + video data for experiments
5. It is recommended to train `LSTMClassifier` before running any other experiments for easy comparsion with other models later on:
  - Change `config.py` for any of the experiment settings (`s2e/t2e/combined`) if you want
  - Run `python lstm_classifier.py` from the corresponding experiment mode's folder inside `lstm_classifier` to train an LSTM classifier for that respective mode
6. Run `5_audio_classification.ipynb` to train ML classifiers for audio
7. Run `5.1_sentence_classification.ipynb` to train ML classifiers for text
8. Run `5.2_combined_classification.ipynb` to train ML classifiers for audio+text

**Note:** Make sure to include correct model paths in the notebooks as not everything is relative right now and it needs some refactoring

## Results
Accuracy, F-score, Precision and Recall has been reported for the different experiments.

**Audio**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 56.0 | **56.0** | 57.2 | **57.3**
XGB | 55.6 | **56.0** | 56.9 | 56.8
SVM | 33.7 | 15.2 | 17.4 | 21.5
MNB | 31.3 | 9.1 | 19.6 | 17.2
LR | 33.4 | 14.9 | 17.8 | 20.9
MLP | 41.0 | 36.5 | 42.2 | 35.9
LSTM | 43.6 | 43.4 | 53.2 | 40.6
ARE (4-class) | 56.3 | - | 54.6 | -
E1 (4-class) | 56.2 | 45.9 | **67.6** | 48.9
**E1** | **56.6** | 55.7 | 57.3 | **57.3**

E1: Ensemble (RF + XGB + MLP)

**Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 62.2 | 60.8 | 65.0 | 62.0
XGB | 56.9 | 55.0 | 70.3 | 51.8
SVM | 62.1 | 61.7 | 62.5 | **63.5**
MNB | 61.9 | 62.1 | **71.8** | 58.6
LR | 64.2 | 64.3 | 69.5 | 62.3
MLP | 60.6 | 61.5 | 62.4 | 63.0
LSTM | 63.1 | 62.5 | 65.3 | 62.8
TRE (4-class) | **65.5** | - | 63.5 | -
E1 (4-class) | 63.1 | 61.4 | **67.7** | 59.0
**E2** | 64.9 | **66.0** | 71.4 | 63.2

E2: Ensemble (RF + XGB + MLP + MNB + LR)
E1: Ensemble (RF + XGB + MLP)

**Audio + Text**

Models | Accuracy | F1 | Precision | Recall
---|---|---|---|---
RF | 65.3 | 65.8 | 69.3 | 65.5
XGB | 62.2 | 63.1 | 67.9 | 61.7
SVM | 63.4 | 63.8 | 63.1 | 65.6
MNB | 60.5 | 60.3 | 70.3 | 57.1
MLP | 66.1 | 68.1 | 68.0 | 69.6
LR | 63.2 | 63.7 | 66.9 | 62.3
LSTM | 64.2 | 64.7 | 66.1 | 65.0
MDRE (4-class) | **75.3** | - | 71.8 | -
E1 (4-class) | 70.3 | 67.5 | **73.2** | 65.5
**E2** | 70.1 | **71.8** | 72.9 | **71.5**

For more details, please refer to the [report](https://arxiv.org/abs/1904.06022)

## Citation
If you find this work useful, please cite:

```
@ARTICLE{2019arXiv190406022S,
       author = {{Sahu}, Gaurav},
        title = "{Multimodal Speech Emotion Recognition and Ambiguity Resolution}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Machine Learning, Computer Science - Computation and Language, Statistics - Machine Learning},
         year = "2019",
        month = "Apr",
          eid = {arXiv:1904.06022},
        pages = {arXiv:1904.06022},
archivePrefix = {arXiv},
       eprint = {1904.06022},
 primaryClass = {cs.LG},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190406022S},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
