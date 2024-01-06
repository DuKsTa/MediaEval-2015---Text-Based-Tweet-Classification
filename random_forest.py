import numpy as np
import pandas as pd
import re
import emoji
import tensorflow as tf
import matplotlib.pyplot as plt

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from langdetect import detect

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score


id_pattern = "([0-9])\d{6,}"
hashtag_pattern = "#\w+"
link_pattern = "http(s)?://[^\s]+"
stoplist = stopwords.words('english')
stoplist.extend([',', ':',"?", "!", "[", "]", "..." , ";", "¿"])

tknzr = TweetTokenizer()
sntmnt = SentimentIntensityAnalyzer()


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

def clean_emoji(tweet):
    return emoji.replace_emoji(tweet, "emoji")

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
    tweet = clean_stop_word(tweet)
    tweet = tknzr.tokenize(tweet)
    return tweet


training = pd.read_csv('mediaeval-2015-trainingset.txt', sep="\t")
validation = pd.read_csv('mediaeval-2015-testset.txt', sep="\t")


training.drop(columns=["tweetId", "userId", "timestamp"], axis='columns', inplace=True)

findLang(training)
findLang(validation)

topics  = {}
test_topics = {}

for i in training['imageId(s)']:
    i = i.split("_")
    if i[0] not in topics:
        topics[i[0]] = 1
    else:
        topics[i[0]] += 1
        
for i in validation['imageId(s)']:
    i = i.split("_")
    if i[0] not in test_topics:
        test_topics[i[0]] = 1
    else:
        test_topics[i[0]] += 1

mapped_test = list(test_topics.keys())
mapped_topics = list(topics.keys())


def topics_mapping(topic):
    return mapped_topics.index(topic)

def test_mapping(topic):
    return mapped_test.index(topic)

training = training[training.lang == 'en']
# validation = validation[validation.lang == 'en']


def tweetLimit(tweet):
    if len(tweet) > 140:
        pos = re.search(id_pattern, tweet)
        if pos:
            return tweet[:pos.start(1)]
        else:
            return tweet[:140]
    else:
        return tweet
    

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
validation['topic'] = validation['imageId(s)'].apply(topic)
validation['filteredTweet'] = validation['tweetText'].apply(clean_all)

labelToNumeric(validation)
labelToNumeric(training)

forest_true = np.asarray(training['label']).astype(np.int32)
forest_train = training.drop(columns=["tweetText", "filteredTweet", "imageId(s)", "username", "lang", "label", "timestamp", "tweetId", "userId"])

forest_true_test = np.asarray(validation['label']).astype(np.int32)
forest_test = validation.drop(columns=["tweetText", "filteredTweet", "imageId(s)", "username", "lang", "label", "timestamp", "tweetId", "userId"])


random_forest = RandomForestClassifier()


parameters = {'n_estimators' : range(1, 200),
              'max_depth': range(1, 20),
             'criterion': ['gini', 'entropy'],
              'max_features': ['auto', 'sqrt', 'log2']
             }


GridSearch = GridSearchCV(random_forest, param_grid = parameters, scoring ='f1', n_jobs=-1)


GridSearch.fit(forest_train, forest_true)
best = GridSearch.best_params_
best_estimator = GridSearch.best_estimator_

y_pred = best_estimator.predict(forest_test)

precision, recall = confusion_matrix(forest_true_test, y_pred)

print("Best parameters are: {}".format(best))

print(f1_score(precision, recall))