# Intent Classification on Speech audio

This repository contains notebooks for Creation of dataset, Training/Fine-tuning transformers models, Performance Analysis and Inferencing on the best-model.

Fine-tuning Wav2Vec2 Huggingface model for audio intent classification

## Table of Contents
* [Demo](#demo)
* [Overview](#overview)
* [Problem Statement](#problem-statement)
* [Source and Useful Links](#source-and-useful-links)
* [Real-world/business Objectives and Constraints](#real-world-business-objectives-and-constraints)
* [Mapping to Machine Learning Problem](#mapping-to-machine-learning-problem)
* [Model Training](#model-training)
* [Technical Aspect](#technical-aspect)
* [Installation](#installation)
* [Run](#run)
* [Directory Tree](#directory-tree)
* [Future Work](#future-work)
* [Technologies used](#technologies-used)
* [Team](#team)
* [Credits](#credits)

## Demo

Link for Inference on audio file on Huggingface: https://huggingface.co/MuhammadIqbalBazmi/wav2vec2-base-intent-classification-ori

## Overview
Intent Classification is used in Conversational AI where we are suppose to understand the intent of the user's speech. Here, we will be processing the Speech data directly instead of using Speech-2-text model.


## Problem Statement
Classify the intent of the Speech audios.

## Source and Useful links
Data Source: [Click Here!](https://huggingface.co/datasets/MuhammadIqbalBazmi/intent-dataset)
Wav2Vec2 : [Click here!](https://ai.facebook.com/blog/wav2vec-20-learning-the-structure-of-speech-from-raw-audio/)
Wav2Vec2 on the Edge: Performance Evaluation: [Download here!](https://arxiv.org/pdf/2202.05993.pdf)
Youtube: [Audio Classification](https://www.youtube.com/watch?v=iuvDLKql3yk&ab_channel=JulienSimon)


## Real-World Business Objectives and Constraints
* Low-latency required because we need to understand it and response in real-time with thousands of request in parallel.

## Mapping to Machine Learning Problem 
1. ### Data
  * Dataset: [Download from here!](https://huggingface.co/datasets/MuhammadIqbalBazmi/intent-dataset)
  * Total datapoints: 160
  * Data-split: TRAIN-TEST \[70:30\]
  * Number of classes: 9
  * Name of classes: 
     \[
     "battery", 
     "Running_operating_cost", 
     "Locate_Dealer", 
     "casual_talk_greeting",
     "Top_speed", 
     "casual_talk_goodbye",
     "About_iQube", 
     "bike_modes",
     "book_now"
     ]
     
2. ### Types of Machine Learning Problem
  * This problem is proposed as Multi-class Audio Classification.
  * As we are having very less amount of data. So, we are fine-tuning Speech Language Model Wav2Vec2.

3. ### Performance Metrics
  * Accuracy: We fine-tuned the model based on Accuracy
  * Precision
  * Recall
  * f1
  * NOTE: We Calculate Precision, Recall and F1 for the best performing model. We used `micro`, `macro`, `weighted` as an `average` option because Multi-class classification.

## Model Training
  * fine-tuned many different models on the given dataset with `accuracy` as a metric.
  * fine-tuned the model for 45 epochs.
  * Finally, uploaded the model to hub.
  * Best Performing model is: [Click to see on huggingface hub!](https://huggingface.co/MuhammadIqbalBazmi/wav2vec2-base-intent-classification-ori)
  * Best Accuracy we get: 0.9167


## Installation
The code is written in python==3.9.5. If you want to install python,  [Click here](https://www.python.org/downloads/). Don't forget to upgrade python if you are using a lower version. upgrade using `pip install --upgrade python`.
1. Create a new environment
    ```
    conda create -y -n <environment_name> python=3.9.5
    ```
2. Clone the repository
3. install requirements.txt:
    ```
    pip isntall -r requirements.txt
    ```
    
## Run
You will have to manually run each jupyter notebooks and see the result.
**For Inference, use huggingface hub model : [Click Here](https://huggingface.co/MuhammadIqbalBazmi/wav2vec2-base-intent-classification-ori)**

## Directory Tree
├───dataset
│   ├───audio
│   └───csv
├───jupyter_notebooks
│   ├───1.data_analysis_and_processing.ipynb
│   ├───2.fine-tuning_speech_models_for_intent_classification.ipynb
│   └───3.inferencing_and_Performance_analysis_of_fine-tuned_models.ipynb
├───inference.py
└───README.md

## Future Work
1. Get mode data and fine-tune on top of that to get more better result and also balanced the audio files per class.

## Technologies used
[![](https://huggingface.co/front/assets/huggingface_logo-noborder.svg)](https://huggingface.co/)

## Team
<a href="https://github.com/iqbal786786"><img src="https://avatars.githubusercontent.com/u/32350208?v=4" width=300></a>
|-|
[Muhammad Iqbal Bazmi](https://github.com/iqbal786786) |)

## Credits
1. [Huggingface](https://huggingface.co/)
