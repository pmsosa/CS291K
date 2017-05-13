import csv
import re
import random
import numpy as np

from IPython import embed

#Separates a file with mixed positive and negative examples into two.
def separate_dataset(filename):
    good_out = open("good_"+filename,"w+");
    bad_out  = open("bad_"+filename,"w+");

    seen = 1;
    with open(filename,'r') as f:
        reader = csv.reader(f)
        reader.next()

        for line in reader:
            seen +=1
            sentiment = line[1]
            sentence = line[3]

            if (sentiment == "0"):
                bad_out.write(sentence+"\n")
            else:
                good_out.write(sentence+"\n")

            if (seen%10000==0):
                print seen;

    good_out.close();
    bad_out.close();



#Load Dataset
def get_dataset(goodfile,badfile,limit,randomize=True):
    good_x = list(open(goodfile,"r").readlines())
    good_x = [s.strip() for s in good_x]
    
    bad_x  = list(open(badfile,"r").readlines())
    bad_x  = [s.strip() for s in bad_x]

    if (randomize):
        random.shuffle(bad_x)
        random.shuffle(good_x)

    good_x = good_x[:limit]
    bad_x = bad_x[:limit]

    x = good_x + bad_x
    x = [clean_str(s) for s in x]


    positive_labels = [[0, 1] for _ in good_x]
    negative_labels = [[1, 0] for _ in bad_x]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x,y]




#Clean Dataset
def clean_str(string):


    #EMOJIS
    string = re.sub(r":\)","emojihappy1",string)
    string = re.sub(r":P","emojihappy2",string)
    string = re.sub(r":p","emojihappy3",string)
    string = re.sub(r":>","emojihappy4",string)
    string = re.sub(r":3","emojihappy5",string)
    string = re.sub(r":D","emojihappy6",string)
    string = re.sub(r" XD ","emojihappy7",string)
    string = re.sub(r" <3 ","emojihappy8",string)

    string = re.sub(r":\(","emojisad9",string)
    string = re.sub(r":<","emojisad10",string)
    string = re.sub(r":<","emojisad11",string)
    string = re.sub(r">:\(","emojisad12",string)

    #MENTIONS "(@)\w+"
    string = re.sub(r"(@)\w+","mentiontoken",string)
    
    #WEBSITES
    string = re.sub(r"http(s)*:(\S)*","linktoken",string)

    #STRANGE UNICODE \x...
    string = re.sub(r"\\x(\S)*","",string)

    #General Cleanup and Symbols
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()



#Generate random batches
#Source: https://github.com/dennybritz/cnn-text-classification-tf/blob/master/data_helpers.py
def gen_batch(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

if __name__ == "__main__":
    separate_dataset("small.txt");


#42
#642