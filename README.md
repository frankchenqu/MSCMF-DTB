## MSCMF-DTB: Drug–Target Binding Prediction

This repository contains the implementation of MSCMF-DTB, supporting both classification (DTI) and regression (DTA) tasks. It includes data preprocessing pipelines, training scripts, evaluation tools, and an independent attention-interpretability module.

## Environment Requirements
* Python ≥ 3.8
* PyTorch ≥ 1.10
* RDKit
* h5py
* NumPy
* scikit-learn
* tape-proteins

## DTI Classification Pipeline
## 1. Data Preparation
* Feature construction is implemented in featurizer.py.
* Generate HDF5 feature files by running:
* python featurizer.py
## 2. Model Training
* Run the classification model with:
* python main.py
## Training includes:
* Loading HDF5 datasets (Human, DrugBank, etc.)
* five-fold cross-validation
* AMP mixed-precision training
* ReduceLROnPlateau adaptive learning-rate scheduling
* Automatic saving of the best validation model
## 3. Evaluation Metrics
* ROC-AUC
* PR-AUC
* Accuracy
* Precision
* Recall
* F1-score
* Log-loss

## DTA Regression Pipeline
## 1. Data Preparation
* Feature construction is implemented in regFeaturizer.py.
* Generate HDF5 feature files with:
* python regFeaturizer.py
## 2. Model Training
* Run the regression model with:
* python regMain.py
## Training includes:
* Loading Davis / KIBA datasets
* Train/dev/test splitting based on predefined fold files
* AMP mixed precision
* R-Drop
* Rank loss
* Multi-sample dropout
* EMA (Exponential Moving Average)
* Saving the best-performing model on validation sets
## 3. Evaluation Metrics
* MSE
* Concordance Index (CI)
* RM²
* R²

## Attention Extraction Module
The project provides an independent interpretability module in attention.py for extracting cross-attention weights from the trained DTI classification model.
You may create a script such as extract_attention.py to load the best model and call functions from attention.py.

## Features
1. No retraining required — simply load the saved best.pt checkpoint.
2. Cross-attention extraction (Protein → Compound) from Transformer decoder layers.
3. Automatic saving of attention-weight matrices for downstream visualization.

## Authors
This code was originally created by Yuxue Pan. Pan is currently a master student at Zhejiang University of Science and Technology under joint supervision of Dr. Qu Chen and Prof. Juan Huang.

This code serves as the Supporting Information for the manuscript entitled "MSCMF-DTB: A Multi-Scale Cross-Modal Fusion Framework for Drug–Target Binding Prediction (submitted)" and can be downloaded for free.

edited on January 5th, 2026

