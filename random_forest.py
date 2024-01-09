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

import language_tool_python as lang_tool

user_pattern = "@\w+"
id_pattern = "([0-9])\d{6,}"
hashtag_pattern = "#\w+"
link_pattern = "http(s)?://[^\s]+"
punctuations1_pattern = "\[.*?\]"
punctuations2_pattern = "\[.*?\]"
random_pattern = "[^a-z0-9]"

stoplist = stopwords.words('english')
stoplist.extend([',', ':',"?", "!", "[", "]", "(", ")", "..." , ";", "¿", "!", ".", "\\\\", "-", "_"])

tknzr = TweetTokenizer()
sntmnt = SentimentIntensityAnalyzer()
grammar = lang_tool.LanguageTool('en-US')


# Change Labels to Numeric Values
def labelToNumeric(data):    
    data.loc[data['label'] == 'humor', 'label'] = 1
    data.loc[data['label'] == 'fake', 'label'] = 1
    data.loc[data['label'] == 'real', 'label'] = 0


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

def mistakes_num(tweet):
    return len(grammar.check(tweet))
        
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

def clean_random(tweet):
    return re.sub(random_pattern, " ", tweet)
    
def clean_hashtag(tweet):
    return re.sub(hashtag_pattern, "tag", tweet)

def clean_link(tweet):
    return re.sub(link_pattern, "link", tweet)

def clean_stop_word(tweet):
    tweet = re.sub(punctuations2_pattern, " ", tweet)
    tweet = re.sub(punctuations1_pattern, " ", tweet)
    return tweet


def clean_all(tweet):
    tweet = clean_emoji(tweet)
    tweet = clean_hashtag(tweet)
    tweet = clean_link(tweet)
    tweet = clean_user(tweet)
    tweet = clean_stop_word(tweet)
    tweet = clean_random(tweet)
    tweet = tweet.lower()
    tweet = tknzr.tokenize(tweet)
    return tweet


training = pd.read_csv('mediaeval-2015-trainingset.txt', sep="\t")
test = pd.read_csv('mediaeval-2015-testset.txt', sep="\t")

findLang(training)
findLang(test)

topics  = {}
test_topics = {}

for i in training['imageId(s)']:
    i = i.split("_")
    if i[0] not in topics:
        topics[i[0]] = 1
    else:
        topics[i[0]] += 1
        
for i in test['imageId(s)']:
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
# test = test[test.lang == 'en']
    

training['tweetText'] = training['tweetText'].apply(tweetLimit)
training['tweetLength'] = training['tweetText'].apply(tweet_length)
training['links'] = training['tweetText'].apply(links_num)
training['hashtags'] = training['tweetText'].apply(hashtags_num)
training['emoji'] = training['tweetText'].apply(emojis_num)
training['sentiment'] = training['tweetText'].apply(sentiment)
training['topic'] = training['imageId(s)'].apply(topic)
training['filteredTweet'] = training['tweetText'].apply(clean_all)

test['tweetText'] = test['tweetText'].apply(tweetLimit)
test['tweetLength'] = test['tweetText'].apply(tweet_length)
test['links'] = test['tweetText'].apply(links_num)
test['hashtags'] = test['tweetText'].apply(hashtags_num)
test['emoji'] = test['tweetText'].apply(emojis_num)
test['sentiment'] = test['tweetText'].apply(sentiment)
test['topic'] = test['imageId(s)'].apply(topic)
test['filteredTweet'] = test['tweetText'].apply(clean_all)

labelToNumeric(test)
labelToNumeric(training)

forest_true = np.asarray(training['label']).astype(np.int32)
forest_train = training.drop(columns=["tweetLength", "tweetText", "filteredTweet", "imageId(s)", "username", "label", "lang", "timestamp", "tweetId"])

forest_true_test = np.asarray(test['label']).astype(np.int32)
forest_test = test.drop(columns=["tweetLength","tweetText", "filteredTweet", "imageId(s)", "username", "lang", "label", "timestamp", "tweetId"])


random_forest = RandomForestClassifier()


parameters = {'n_estimators' : range(1, 300),
              'max_depth': range(1, 30),
             'criterion': ['gini', 'entropy'],
             }


GridSearch = GridSearchCV(random_forest, param_grid = parameters, scoring ='f1', n_jobs=-1)


GridSearch.fit(forest_train, forest_true)
best = GridSearch.best_params_
best_estimator = GridSearch.best_estimator_

y_pred = best_estimator.predict(forest_test)

precision, recall = confusion_matrix(forest_true_test, y_pred)

print("Best parameters are: {}".format(best))

print(f1_score(precision, recall))

feature_importances = optimised_random_forest.feature_importances_
feature_names = forest_train.columns
feature_importance_dict = dict(zip(forest_train, feature_importances))

sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

for feature, importance in sorted_features:
    print(f"Feature: {feature}, Importance: {importance}")