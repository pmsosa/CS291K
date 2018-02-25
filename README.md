# CS291K
Sentiment Analysis of Twitter data using a combined CNN-LSTM Neural Network model

- Paper: https://www.academia.edu/35947062/Twitter_Sentiment_Analysis_using_combined_LSTM-CNN_Models
- Blog Post: http://konukoii.com/blog/2018/02/19/twitter-sentiment-analysis-using-combined-lstm-cnn-models/

### Motivation
This project seeks to extend the work we did previously on sentiment analysis using simple Feed-Foward Neural Networks (Found here: [paper](https://www.academia.edu/30498927/Twitter_Sentiment_Analysis_with_Neural_Networks) & [repo](https://github.com/pmsosa/Twitter-Sentiment-Analysis)).
Instead, we wish to experiment with building a combined CNN-LSTM Neural Net model using Tensorflow to perform sentiment analysis on Twitter data.

### Dependencies
```
sudo -H pip install -r requirements.txt
```

### Run the Code
- On train.py change the variable MODEL_TO_RUN = {0 or 1}
  - 0 = CNN-LSTM
  - 1 = LSTM-CNN
- Feel free to change other variables (batch_size, filter_size, etc...)
- Run ```python train.py``` (or, with proper permissions, ```./train.py```

### Code Structure ###
- [lstm_cnn.py](./lstm_cnn.py) : Contains the LSTM_CNN Model class to be instantiated.
- [cnn_lstm.py](./cnn_lstm.py) : Contains the CNN_LSTM Model class to be instantiated.
- [train.py](./train.py) : Main runner for the code. It instantiates a model, trains it and validates it.
- [batchgen.py](./batchgen.py) : Contains a couple of functions needed to pre-process and tokenize the dataset.



