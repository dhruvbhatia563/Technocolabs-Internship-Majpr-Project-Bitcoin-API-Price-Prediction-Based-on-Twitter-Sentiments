from flask_code import app
import numpy as np
import pandas as pd
import re
import random
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

import pickle


model = pickle.load(open('bitcoin.pkl', 'rb'))

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

def clean_text(text):
    text = re.sub(r'https?:\/\/\S*'," ", text) # Removing the url from the text
    text = re.sub(r'@\S+', " ", text) # Removing twitter handles from the text
    text = re.sub('#'," ", text) # removing # from the data
    text = re.sub(r'RT', "", text) # Removing the Re-tweet mark
    text = re.sub(r"\s+"," ", text)  # Removing Extra Spaces
    text = text.lower()
    return text

#removes pattern in the input text
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for word in r:
        input_txt = re.sub(word, "", input_txt)
    return input_txt.lower()

price = pd.read_csv('livebitcoindata.csv')



from flask import render_template, request


@app.route('/')
@app.route('/home')
def homepage():
    return render_template('homepage.html',title='Home')

@app.route('/about')
def about():
    return render_template('about.html',title='About Us')

@app.route('/tweet',methods=['GET','POST'])
def tweet():
    if request.method == 'POST':
        user = request.form["TweetonBitcoin"]
        data = {'o_t':  [user]}
        user_df = pd.DataFrame (data, columns = ['o_t'])

        #removing the twitter handles @user
        user_df['clean_tweet'] = np.vectorize(remove_pattern)(user_df['o_t'], "@[\w]*")

        #using above functions
        user_df['clean_tweet'] = user_df['clean_tweet'].apply(lambda x : clean_text(x))
        user_df['clean_tweet'] = user_df['clean_tweet'].apply(lambda x : contx_to_exp(x))
        user_df['clean_tweet'] = user_df['clean_tweet'].apply(lambda x : emotion_check(x))

        #removing special characters, numbers and punctuations
        user_df['clean_tweet'] = user_df['clean_tweet'].str.replace("[^a-zA-Z]", " ")


        #remove short words
        user_df['clean_tweet'] = user_df['clean_tweet'].apply(lambda x: " ".join([w for w in x.split() if len(w)>3]))

        # Removing every thing other than text
        user_df['clean_tweet'] = user_df['clean_tweet'].apply( lambda x: re.sub(r'[^\w\s]',' ',x))  # Replacing Punctuations with space
        user_df['clean_tweet'] = user_df['clean_tweet'].apply( lambda x: re.sub(r'[^a-zA-Z]', ' ', x)) # Raplacing all the things with space other than text
        user_df['clean_tweet'] = user_df['clean_tweet'].apply( lambda x: re.sub(r"\s+"," ", x)) # Removing extra spaces


        #individual words as tokens
        tokenized_tweet = user_df['clean_tweet'].apply(lambda x: x.split())


        #stem the words
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()

        tokenized_tweet = tokenized_tweet.apply(lambda sentence: [lemmatizer.lemmatize(stemmer.stem(word)) for word in sentence])

        #combine words into single sentence 
        for i in range(len(tokenized_tweet)):
            tokenized_tweet[i] = " ".join(tokenized_tweet[i])



        user_df['clean_tweet'] = tokenized_tweet


        # for performing NLP Functions i.e detection of Polarity and Subjectivity

        polarity=[]     #list that contains polarity of tweets
        subjectivity=[]    ##list that contains subjectivity of tweets

        for i in user_df.clean_tweet.values:
            try:
                analysis = TextBlob(i) # [i] records to the first data in dataset
                polarity.append(analysis.sentiment.polarity)
                subjectivity.append(analysis.sentiment.subjectivity)
            except:
                polarity.append(0)
                subjectivity.append(0)



        # adding sentiment polarity and subjectivity column to dataframe

        user_df['polarity'] = polarity
        user_df['subjectivity'] = subjectivity


        #Create a function to get the sentiment scores (using Sentiment Intensity Analyzer)
        def getSIA(text):
            sia = SentimentIntensityAnalyzer()
            sentiment = sia.polarity_scores(text)
            return sentiment

        #Get the sentiment scores 
        compound = []
        neg = []
        neu = []
        pos = []
        SIA = 0
        for i in range(0, len(user_df['clean_tweet'])):
            SIA = getSIA(user_df['clean_tweet'][i])
            compound.append(SIA['compound'])
            neg.append(SIA['neg'])
            neu.append(SIA['neu'])
            pos.append(SIA['pos'])

        #Store the sentiment scores in the data frame
        user_df['Compound'] =compound
        user_df['Negative'] =neg
        user_df['Neutral'] =neu
        user_df['Positive'] = pos
        
        list1 = []
        for i in range(price.shape[0]):
            list1.append(i)
        random_index = random.choice(list1)
        
        a=[]
        a.append(price.Open[random_index])
        a.append(price.High[random_index])
        a.append(price.Low[random_index])
        a.append(price.Close[random_index])
        a.append(price.Volume[random_index])

        a.append(user_df.polarity[0])
        a.append(user_df.subjectivity[0])
        a.append(user_df.Compound[0])
        a.append(user_df.Negative[0])
        a.append(user_df.Neutral[0])
        a.append(user_df.Positive[0])

        final_features = [np.array(a)]


        prediction = model.predict(final_features)

        output = int(prediction[0]) #1 stands for price up; 0 stands for price down
        
        if user_df['polarity'][0] > 0:   # Positive Sentiment
            if output == 1:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "POSITIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)
            else:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "POSITIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)
            
        elif user_df['polarity'][0] < 0:  # Negative Sentiment
            if output == 1:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "NEGATIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)
            else:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "NEGATIVE" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)

        else:    # Neutral Sentiment
            if output == 1:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "NEUTRAL" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Up as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)            
            else:
                prediction_text1='Entered Tweet is = {}'.format(user)
                prediction_text2='Tweet Sentiment is "NEUTRAL" as polarity = {}'.format(user_df['polarity'][0])
                prediction_text3='Price Down as value predicted is = {}'.format(output)
                combine = prediction_text1 + ' || ' + prediction_text2 + ' || ' + prediction_text3
                return render_template('tweet.html', prediction_text = combine)        
       
    
    return render_template('tweet.html',title='Write a Tweet')

@app.route('/link')
def link():
    return render_template('link.html',title='GitHub Link')

