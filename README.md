# Natural Language Processing with Disaster Tweets (Kaggle Competition)

## Competition Description

In this competition, a machine learning model should be built to predict which Tweets are about real disasters and which ones are not. The trainining and testing data of 10,000 tweets that were hand classified is provided.  

Submittions are evaluated using F1 between the predicted and expected answers. 

https://www.kaggle.com/competitions/nlp-getting-started/

## Approach

TensorFlow Hub is a repository of pre-trained TensorFlow models. I used TF2.0 Saved Model (v1) as trasfer learning. TF2.0 Saved Model (v1) is token based text embedding trained on English Googld News and used as lower layers of the first netowrk.

https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1

## Result

With few lines of code, the F1 score of 0.80784 has been achieved so far. 

I include test2.py file to test the model to see if it makes correct predictions on a couple of example sentences. I set up the automated Pytest with GitHub actions, which all passed. 