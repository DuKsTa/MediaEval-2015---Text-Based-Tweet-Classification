import numpy as np
import pandas as pd
import re
import emoji
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow_addons as tfa

from keras.models import Sequential
from keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from keras.layers import Embedding, Dense, LSTM, Dropout, Bidirectional


from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from langdetect import detect


user_pattern = "@\w+"
id_pattern = "([0-9])\d{6,}"
hashtag_pattern = "#\w+"
link_pattern = "http(s)?://[^\s]+"
stoplist = stopwords.words('english')
stoplist.extend([',', ':',"?", "!", "[", "]", "(", ")", "..." , ";", "¿", "!", ".", "\\\\", "-", "_"])

tknzr = TweetTokenizer()
sntmnt = SentimentIntensityAnalyzer()

def tweetLimit(tweet):
    if len(tweet) > 140:
        pos = re.search(id_pattern, tweet)
        if pos:
            return tweet[:pos.start(1)]
        else:
            return tweet[:140]
    else:
        return tweet
    
def tweet_length(tweet):
    return len(tweet)
            
def hashtags_num(tweet):
    num = re.findall(hashtag_pattern, tweet)
    return len(num)

def links_num(tweet):
    urls = re.findall(link_pattern, tweet)
    return len(urls)

def sentiment(tweet):
    return sntmnt.polarity_scores(tweet)['compound']

def topic(imageid):
    return imageid.split("_")[0]


def emojis_num(tweet):
    return emoji.emoji_count(tweet)

# emoji.demojize(tweet, delimiters=("", ""))
def clean_emoji(tweet):
    return emoji.replace_emoji(tweet, "emoji")

def clean_user(tweet):
    return re.sub(user_pattern, "user", tweet)

def clean_hashtag(tweet):
    return re.sub(hashtag_pattern, "tag", tweet)

def clean_link(tweet):
    return re.sub(link_pattern, "link", tweet)

def clean_stop_word(tweet):
    tweet = tweet.split()
    tweet = [word.lower() for word in tweet if word not in stoplist]
    return ' '.join(tweet)


def clean_all(tweet):
    tweet = clean_emoji(tweet)
    tweet = clean_hashtag(tweet)
    tweet = clean_link(tweet)
    tweet = clean_user(tweet)
    tweet = clean_stop_word(tweet)
    tweet = tknzr.tokenize(tweet)
    return tweet


# Change Labels to Numeric Values
def labelToNumeric(data):    
    data.loc[data['label'] == 'humor', 'label'] = 1
    data.loc[data['label'] == 'fake', 'label'] = 1
    data.loc[data['label'] == 'real', 'label'] = 0

# Detect the Language of Each Tweet
def findLang(data):
    langs = []
    for tweet in data["tweetText"]:
        try:
            lang = detect(tweet)
        except:
            ## some tweets will have excessive number of emojis
            lang = detect(tweet[:20])
            
        langs.append(lang)

    data["lang"] = langs

# Confusion Matrix

def confusion_matrix(y_true, y_pred):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    
    for true, pred in zip(y_true, y_pred):
        if true == pred == 1:
            tp += 1
        elif true == pred == 0:
            tn += 1
        elif (true == 0) and pred == 1:
            fp += 1
        elif (true == 1) and pred == 0:
            fn += 1
    
    try:
        precision = tp / (tp + fp)

        recall = tp / (tp + fn)
        
    except ZeroDivisionError:
        recall = precision = 0
    
    return precision, recall

# F-1 Score
def f1_score(precision, recall):
    
    if precision == 0 or recall == 0:
        return 0
    
    return 2 * ((precision * recall) / (precision + recall))


# Call back to calculate the f-1 Score

class F1History(tf.keras.callbacks.Callback):

    def __init__(self, train, validation=None):
        super(F1History, self).__init__()
        self.validation = validation
        self.train = train

    def on_epoch_end(self, epoch, logs={}):

        logs['F1_score_train'] = float('-inf')
        X_train, y_train = self.train[0], self.train[1]
        y_pred = (self.model.predict(X_train).ravel()>0.4)+0
        precision, recall = confusion_matrix(y_train, y_pred)
        score = f1_score(precision, recall)       
        logs['F1_score_train'] = np.round(score, 5)
        logs['Precision'] = np.round(precision, 5)
        logs['Recall'] = np.round(recall, 5)
        
        if (self.validation):
            logs['F1_score_val'] = float('-inf')
            X_valid, y_valid = self.validation[0], self.validation[1]
            y_val_pred = (self.model.predict(X_valid).ravel()>0.4)+0
            val_precision, val_recall = confusion_matrix(y_valid, y_val_pred)
            val_score = f1_score(val_precision, val_recall)
            logs['F1_score_val'] = np.round(val_score, 5)
            logs['Precision_val'] = np.round(val_precision, 5)
            logs['Recall_val'] = np.round(val_recall, 5)


training = pd.read_csv('mediaeval-2015-trainingset.txt', sep="\t")
validation = pd.read_csv('mediaeval-2015-testset.txt', sep="\t")

findLang(training)
findLang(validation)


training = training[training.lang == 'en']


training['tweetText'] = training['tweetText'].apply(tweetLimit)
training['tweetLength'] = training['tweetText'].apply(tweet_length)
training['links'] = training['tweetText'].apply(links_num)
training['hashtags'] = training['tweetText'].apply(hashtags_num)
training['emoji'] = training['tweetText'].apply(emojis_num)
training['sentiment'] = training['tweetText'].apply(sentiment)
training['topic'] = training['imageId(s)'].apply(topic)
training['filteredTweet'] = training['tweetText'].apply(clean_all)

validation['tweetText'] = validation['tweetText'].apply(tweetLimit)
validation['tweetLength'] = validation['tweetText'].apply(tweet_length)
validation['links'] = validation['tweetText'].apply(links_num)
validation['hashtags'] = validation['tweetText'].apply(hashtags_num)
validation['emoji'] = validation['tweetText'].apply(emojis_num)
validation['sentiment'] = validation['tweetText'].apply(sentiment)
validation['filteredTweet'] = validation['tweetText'].apply(clean_all)

labelToNumeric(validation)
labelToNumeric(training)

text_dataset = [t for tweet in training['filteredTweet'] for t in tweet]
workout = [t for t in training['filteredTweet']]
test = [t for t in validation['filteredTweet']]



max_features = 6000  # Maximum vocab size.
max_len = 140  # Sequence length to pad the outputs to.

toknizer = Tokenizer()
toknizer.fit_on_texts(text_dataset)
train_sequences = toknizer.texts_to_sequences(workout)
test_sequences = toknizer.texts_to_sequences(test)

padded_train_sequences = pad_sequences(train_sequences, maxlen=max_len)
padded_test_sequences = pad_sequences(test_sequences, maxlen=max_len)

training_tensor = tf.convert_to_tensor(padded_train_sequences, dtype=tf.int32)
testing_tensor = tf.convert_to_tensor(padded_test_sequences, dtype=tf.int32)

y_train = np.asarray(training['label']).astype(np.int32)
y_test = np.asarray(validation['label']).astype(np.float32)

x_train = tf.convert_to_tensor(padded_train_sequences, dtype=tf.int32)
x_test = tf.convert_to_tensor(padded_test_sequences, dtype=tf.int32)


class_weights = compute_class_weight(class_weight= 'balanced', classes=[0,1], y=y_train)

class_weights = {0 : class_weights[0], 1:class_weights[1]}

print("Class Weights are {}".format(class_weights))


def build_model(hp):
    embed_size = 64
    sequence_length = max_len

    dense_activations = hp.Choice('activation', values=['relu', 'tanh', 'softmax'])
    hp_embedding_size = hp.Choice('embeds', values= [8, 16, 32, 64, 128, 256])
    hp_lr = hp.Choice('lr', values= [0.01, 0.001, 0.0001])
    hp_vocab_size = hp.Int('vocab', min_value=1000, max_value= 20000, step=50)
    hp_dropout_values = hp.Float('dropout', min_value = 0.05, max_value = 0.5, step = 0.05)
    hp_f1_threshold = hp.Float('F1_Threshold', min_value= 0.3, max_value= 0.6, step = 0.025)

    model = Sequential()

    model.add(Embedding(input_dim=hp_vocab_size, output_dim=hp_embedding_size, input_length=sequence_length))

    model.add(Bidirectional(LSTM(units=hp_embedding_size, return_sequences=True)))
    # model.add(Bidirectional(LSTM(units=hp_embedding_size, return_sequences=True)))

    model.add(Bidirectional(LSTM(units=1)))

    model.add(Dropout(hp_dropout_values))

    model.add(Dense(hp_embedding_size, activation=dense_activations))

    model.add(Dense(1, activation='sigmoid'))

    opt = tf.keras.optimizers.Adam(learning_rate = hp_lr)
    f1 = tfa.metrics.F1Score(num_classes= 1, threshold=hp_f1_threshold)

    model.compile(
        loss = 'binary_crossentropy',
        optimizer = opt,
        metrics= [f1]
    )

    return model


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=2)

tune = kt.Hyperband(build_model, 
                    objective=kt.Objective('f1_score', direction='max'),
                    max_epochs = 30,
                    factor = 10,
                    hyperband_iterations=5,
                    directory='dir2',
                    project_name='BIlstm')

tune.search(x_train, y_train, epochs=50, validation_split=0.2, callbacks=[reduce_lr,
                    F1History(train=(x_train, y_train),
                             validation=(x_test, y_test))])

best_hps = tune.get_best_hyperparameters(num_trials=1)[0]

print("The best parameters are {}".format(best_hps))

model = tune.hypermodel.build(best_hps)

history = model.fit(x=x_train, y=y_train,
          epochs = 20,
          callbacks=[reduce_lr,
                    F1History(train=(x_train, y_train),
                             validation=(x_test, y_test))])

print(history.history)
history_df = pd.DataFrame(history.history)
# Save the DataFrame to a CSV file
history_df.to_csv('2lstm-64.csv')