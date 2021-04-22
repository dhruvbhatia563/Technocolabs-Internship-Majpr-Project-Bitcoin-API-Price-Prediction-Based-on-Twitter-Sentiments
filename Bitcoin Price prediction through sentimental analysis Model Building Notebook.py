#!/usr/bin/env python
# coding: utf-8

# # Bitcoin-Price-Prediction-Using-Twitter-Sentiment-Analysis

# ### Techncolobas Data Science Internship Major Project
# 
#  - Project shows that real-time Twitter data can be used to predict market movement of Bitcoin Price. 
#  - The goal of this project is to prove whether Twitter data relating to cryptocurrencies can be utilized to develop advantageous crypto coin trading strategies. 
#  - By way of supervised machine learning techniques, have outlined several machine learning pipelines with the objective of identifying cryptocurrency market movement. 
#  - The prominent alternative currency ex- amined in this paper is Bitcoin (BTC). Our approach to cleaning data and applying supervised learning algorithms such as logistic regression, Decision Tree Classifier, and LDA leads to a final prediction accuracy exceeding 70%. 
#  - In order to achieve this result, rigorous error analysis is employed in order to ensure that accurate inputs are utilized at each step of the model.

# ![image.png](attachment:image.png)

# # Importing Libraries and Packages

# In[ ]:


#pip install -q wordcloud
#pip install gensim
#!pip install vaderSentiment


# In[1]:


# Filtering out the warnings
import warnings

warnings.filterwarnings('ignore')


# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import wordcloud

import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger') 


# In[24]:


import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report, confusion_matrix


# # Getting the Dataset

# In[6]:


tweets = pd.read_csv('bitcointweetsscraping.csv')
tweets.head()


# In[4]:


price = pd.read_csv('livebitcoindata.csv')
price.head()


# # About the Data

# In[7]:


tweets.info()


# In[8]:


price.info()


# In[9]:


tweets.shape


# In[10]:


price.shape


# In[11]:


tweets.isnull().sum().any()


# In[12]:


price.isnull().sum().any()


# #### No Null values in both datasets

# In[13]:


tweets.describe()


# #### Average length of any tweet is approx. 126
# #### Maximum number of Re-Tweets are: 11002

# In[14]:


price.describe()


# # Cleaning of Tweets

# In[15]:


# Apostrophe Dictionary
apostrophe_dict = {
"ain't": "am not / are not",
"aren't": "are not / am not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he had / he would",
"he'd've": "he would have",
"he'll": "he shall / he will",
"he'll've": "he shall have / he will have",
"he's": "he has / he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how has / how is",
"i'd": "I had / I would",
"i'd've": "I would have",
"i'll": "I shall / I will",
"i'll've": "I shall have / I will have",
"i'm": "I am",
"i've": "I have",
"isn't": "is not",
"it'd": "it had / it would",
"it'd've": "it would have",
"it'll": "it shall / it will",
"it'll've": "it shall have / it will have",
"it's": "it has / it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she had / she would",
"she'd've": "she would have",
"she'll": "she shall / she will",
"she'll've": "she shall have / she will have",
"she's": "she has / she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as / so is",
"that'd": "that would / that had",
"that'd've": "that would have",
"that's": "that has / that is",
"there'd": "there had / there would",
"there'd've": "there would have",
"there's": "there has / there is",
"they'd": "they had / they would",
"they'd've": "they would have",
"they'll": "they shall / they will",
"they'll've": "they shall have / they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we had / we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what shall / what will",
"what'll've": "what shall have / what will have",
"what're": "what are",
"what's": "what has / what is",
"what've": "what have",
"when's": "when has / when is",
"when've": "when have",
"where'd": "where did",
"where's": "where has / where is",
"where've": "where have",
"who'll": "who shall / who will",
"who'll've": "who shall have / who will have",
"who's": "who has / who is",
"who've": "who have",
"why's": "why has / why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you had / you would",
"you'd've": "you would have",
"you'll": "you shall / you will",
"you'll've": "you shall have / you will have",
"you're": "you are",
"you've": "you have"
}

def contx_to_exp(text):
  for key in apostrophe_dict:
    value = apostrophe_dict[key]
    text = text.replace(key, value)
    return text


# In[16]:


# Emotion detection by different symbols
emotion_dict = {
":)": "happy",
":‑)": "happy",
":-]": "happy",
":-3": "happy",
":->": "happy",
"8-)": "happy",
":-}": "happy",
":o)": "happy",
":c)": "happy",
":^)": "happy",
"=]": "happy",
"=)": "happy",
"<3": "happy",
":-(": "sad",
":(": "sad",
":c": "sad",
":<": "sad",
":[": "sad",
">:[": "sad",
":{": "sad",
">:(": "sad",
":-c": "sad",
":-< ": "sad",
":-[": "sad",
":-||": "sad"
}

def emotion_check(text):
  for key in emotion_dict:
    value = emotion_dict[key]
    text = text.replace(key, value)
    return text


# In[17]:


def clean_text(text):
  text = re.sub(r'https?:\/\/\S*'," ", text) # Removing the url from the text
  text = re.sub(r'@\S+', " ", text) # Removing twitter handles from the text
  text = re.sub('#'," ", text) # removing # from the data
  text = re.sub(r'RT', "", text) # Removing the Re-tweet mark
  text = re.sub(r"\s+"," ", text)  # Removing Extra Spaces
  text = text.lower()
  return text

#removes pattern in the input text
import re
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt.lower()


# In[18]:


#removing the twitter handles @user
tweets['clean_tweet'] = np.vectorize(remove_pattern)(tweets['original_Tweets'], "@[\w]*")

#using above functions
tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x : clean_text(x))
tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x : contx_to_exp(x))
tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x : emotion_check(x))

#removing special characters, numbers and punctuations
tweets['clean_tweet'] = tweets['clean_tweet'].str.replace("[^a-zA-Z]", " ")


#remove short words
tweets['clean_tweet'] = tweets['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

# Removing every thing other than text
tweets['clean_tweet'] = tweets['clean_tweet'].apply( lambda x: re.sub(r'[^\w\s]',' ',x))  # Replacing Punctuations with space
tweets['clean_tweet'] = tweets['clean_tweet'].apply( lambda x: re.sub(r'[^a-zA-Z]', ' ', x)) # Raplacing all the things with space other than text
tweets['clean_tweet'] = tweets['clean_tweet'].apply( lambda x: re.sub(r"\s+"," ", x)) # Removing extra spaces


#individual words as tokens
tokenized_tweet = tweets['clean_tweet'].apply(lambda x: x.split())


#stem the words

from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

tokenized_tweet = tokenized_tweet.apply(lambda sentence: [lemmatizer.lemmatize(stemmer.stem(word)) for word in sentence])



#combine words into single sentence 
for i in range(len(tokenized_tweet)):
    tokenized_tweet[i] = " ".join(tokenized_tweet[i])
    

    
tweets['clean_tweet'] = tokenized_tweet
tweets.head()


# # Ading two dataframes

# In[19]:


merge = pd.DataFrame(data=price[['Open','High','Low','Close','Volume']], columns=['Open','High','Low','Close','Volume'])
merge.info()


# In[20]:


merge['text'] = tweets['clean_tweet']
merge.info()


# In[21]:


merge = merge.dropna(subset=['text'])
merge.info()


# In[22]:


merge.head()


# # Calculating Sentiment Polarity and Subjectivity
# 
#  - The subjectivity shows how subjective or objective a statement is.
# 
#  - The polarity shows how positive/negative the statement is, a value equal to 1 means the statement is positive, a value equal to 0 means the statement is neutral and a value of -1 means the statement is negative.

# In[23]:


from textblob import TextBlob     # for performing NLP Functions i.e detection of Polarity and Subjectivity

polarity=[]     #list that contains polarity of tweets
subjectivity=[]    ##list that contains subjectivity of tweets

for i in merge.text.values:
    try:
        analysis = TextBlob(i) # [i] records to the first data in dataset
        polarity.append(analysis.sentiment.polarity)
        subjectivity.append(analysis.sentiment.subjectivity)
        
    except:
        polarity.append(0)
        subjectivity.append(0)
        

        
# adding sentiment polarity and subjectivity column to dataframe

merge['polarity'] = polarity
merge['subjectivity'] = subjectivity
merge.head()


# #### To create a function t
#  - to get sentiment scores (neg, pos, neu, & compound). 
#  
# ##### The compound score is a metric that calculates the sum of all the lexicon ratings which have been normalized between 
#  - -1(most extreme negative) and +1 (most extreme positive).
# 
# ##### Pos is the positive percentage score, neg is the negative percentage score, and neu is the neutral percentage score.
# 
# #### The total for %pos + %neg + %neu = 100%

# In[25]:


#Create a function to get the sentiment scores (using Sentiment Intensity Analyzer)
def getSIA(text):
    sia = SentimentIntensityAnalyzer()
    sentiment = sia.polarity_scores(text)
    return sentiment


# In[26]:


#Get the sentiment scores 
compound = []
neg = []
neu = []
pos = []
SIA = 0
for i in range(0, len(merge['text'])):
    SIA = getSIA(merge['text'][i])
    compound.append(SIA['compound'])
    neg.append(SIA['neg'])
    neu.append(SIA['neu'])
    pos.append(SIA['pos'])


# In[27]:


#Store the sentiment scores in the data frame
merge['Compound'] =compound
merge['Negative'] =neg
merge['Neutral'] =neu
merge['Positive'] = pos


# In[28]:


merge.head()


# In[29]:


merge.info()


# # To find sentiments of tweets

# In[30]:


# create a function get the sentiment text
def getSentiment(score):
    if score < 0:
        return "negative"
    elif score == 0:
        return "neutral"
    else:
        return "positive"


# In[31]:


# create a column to store the text sentiment
merge['tweet_sentiment'] = merge['polarity'].apply(getSentiment)
merge.head()


# In[32]:


# create a function get the sentiment text
def getSentiment(score):
    if score < 0:
        return 0 #negative
    elif score == 0:
        return 1  #neutral
    else:
        return 2 #positive


# In[33]:


# create a column to store the text sentiment
merge['tweet_sentiment_flag'] = merge['polarity'].apply(getSentiment)
merge.head()


# In[47]:


# inspect sentiment

plt.figure(figsize=[10,5])
sns.countplot(merge.tweet_sentiment_flag)
plt.title('Sentiment V/s. Count', fontsize=15)

label = (merge.tweet_sentiment_flag.value_counts(normalize=True)*100).round(2)
for i in range(3):
    plt.text(x = i, y = label[i], s = label[i],horizontalalignment='center',rotation = 360, color = "black", 
             weight="bold", fontsize=15)
    
plt.legend

plt.show()


# ### Negative Sentimets Count = 9
# ### Neutral Sentimets Count = 65
# ### Positive Sentimets Count = 26

# In[49]:


# scatter plot to show the subjectivity and the polarity
plt.figure(figsize=(14,10))

for i in range(merge.shape[0]):
    plt.scatter(merge["polarity"].iloc[[i]].values[0], merge["subjectivity"].iloc[[i]].values[0], color="Purple")

plt.title("Sentiment Analysis Scatter Plot")
plt.xlabel('polarity')
plt.ylabel('subjectivity')
plt.show()


# ### As data is less but still can visulaise that more of the sentiments are tilted towrds positive with more opinion related tweets rather than factual

# # Creating Target Column
# 
#  - Price Indicator
#      - which will showcase the price in negative or positive in nature
#      - price indicator negative = 'zero' value means price will go down
#      - price indicator posiitve = 'one' value means price will go up

# In[50]:


price_indicator = [merge.Close[0] - merge['Open'][0]]
for i in range(99):
    price_indicator.append(merge.Close[i+1] - merge.Close[i])
price_indicator


# In[51]:


merge['price_indicator'] = 0
for i in range(len(price_indicator)):
    merge['price_indicator'][i] = price_indicator[i]
    
merge.head()


# In[60]:


merge['target'] = 0
for i in range(100):
    if merge.price_indicator[i] > 0:
        merge['target'][i] = 1 
        
# 0 - price down
# 1 - price up

merge.head()


# In[61]:


merge.info()


# In[62]:


#Create a list of columns to keep in the completed data set and show the data.

keep_columns = ['Open','High','Low','Close','Volume','polarity','subjectivity','Compound','Negative','Neutral','Positive','target']
df = merge[keep_columns]
df.head()


# # Model Building

# In[63]:


#Create the feature data set
X = df
X = np.array(X.drop(['target'],1))
#Create the target data set
y = np.array(df['target'])


# In[65]:


#Split the data into 80% training and 20% testing data sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)


# In[66]:


X_train


# # TPOT Classifier
# 
# ![image.png](attachment:image.png)

# In[72]:


from tpot import TPOTClassifier
from sklearn.metrics import confusion_matrix,accuracy_score,roc_auc_score


# In[73]:


# Instantiate TPOTClassifier
tpot = TPOTClassifier(
    generations=5, #number of iterations to run ; pipeline optimisation process ; by default value is 100
    population_size=20, #number of individuals to retrain in the genetic programing popluation in every generation, by default value is 100
    verbosity=2, #it will state how much info TPOT will communicate while it is running
    scoring='roc_auc', #use to evaluate the quality of given pipeline
    random_state=42,
    disable_update_check=True,
    config_dict='TPOT light'
)
tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Print best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Print idx and transform
    print(f'{idx}. {transform}')


# In[74]:


tpot.fitted_pipeline_


# # Model 1: Decision tree classifier

# In[76]:


from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(criterion='entropy', max_depth=8,
                                        min_samples_leaf=10,
                                        min_samples_split=6,
                                        random_state=42)
clf.fit(X_train,y_train)


# In[77]:


y_predicted = clf.predict(X_test)


# In[78]:


y_predicted


# In[79]:


print( classification_report(y_test, y_predicted) )


# In[81]:


accuracy_score(y_test,y_predicted)*100


# #### Model is 55% accurate

# ## Creating Pipeline to see which model has more accuracy

# In[86]:


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[88]:


pipeline_lr = Pipeline([('scaler1',StandardScaler()),
                       ('pca1',PCA(n_components=2)),
                       ('lr_classifier',LogisticRegression(random_state=0))])


# In[89]:


pipeline_dt = Pipeline([('scaler2',StandardScaler()),
                       ('pca2',PCA(n_components=2)),
                       ('dt_classifier',DecisionTreeClassifier())])


# In[90]:


pipeline_randomforest = Pipeline([('scaler3',StandardScaler()),
                       ('pca3',PCA(n_components=2)),
                       ('rf_classifier',RandomForestClassifier())])


# In[92]:


pipeline = [pipeline_lr,pipeline_dt,pipeline_randomforest]


# In[93]:


best_accuracy=0.0
best_classifier=0
best_pipeline=""


# In[94]:


pipe_dict = {0:'Logistic Regression', 1:'Decision Tree', 2:'RandomForest'}

for pipe in pipeline:
    pipe.fit(X_train,y_train)
    


# In[95]:


for i,model in enumerate(pipeline):
    print("{}Test Accuracy: {}".format(pipe_dict[i],model.score(X_test,y_test)))


# # Model 2: LDA

# In[96]:


model = LinearDiscriminantAnalysis().fit(X_train, y_train)


# In[97]:


#Get the models predictions/classification
predictions = model.predict(X_test)
predictions


# In[98]:


print( classification_report(y_test, predictions) )


# In[99]:


accuracy_score(y_test,predictions)*100


# #### Model is 70% accurate

# In[100]:


X_test


# In[102]:


import pickle

# Saving model to disk
pickle.dump(model, open('bitcoin.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('bitcoin.pkl', 'rb'))
print(model.predict([[ 6.00269258e+04,  6.00775508e+04,  6.00269258e+04,
         6.00572305e+04,  0.00000000e+00,  0.00000000e+00,
         0.00000000e+00,  7.72000000e-02,  0.00000000e+00,
         8.85000000e-01,  1.15000000e-01]]))


# In[ ]:





# In[ ]:




