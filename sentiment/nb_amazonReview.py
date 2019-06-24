#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
from sklearn.externals import joblib 
from sklearn.pipeline import Pipeline


# In[3]:


current_path = os.getcwd()


# In[14]:


class naive_bayes_sentiment:
    def data_preproc(self):
        
        #Loading the data
        data = open('/Users/seethu-8363/Downloads/amazon_reviews.csv').read()
        
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))

        labels[np.where(labels=='__label__2')] = "Positive"
        labels[np.where(labels=='__label__1')] = "Negative"
        # create a dataframe using texts and lables
        trainDF = pd.DataFrame()
        trainDF['text'] = texts
        trainDF['label'] = labels
        
        #lower case
        trainDF["text"] = trainDF["text"].apply(lambda x: " ".join(x.lower() for x in x.split()))
        trainDF["text"].head()
        
        #STOP WORDS
        #Removal of stop words
        stop = stopwords.words('english')
        ##Creating a list of custom stopwords
        new_words = ["using", "show", "result", "large", "also", "iv", "one", "two", "new", "previously", "shown","way","ever",
             "us","up","cd","that","all","second"]
        stop.append(new_words)
        trainDF["text"] = trainDF["text"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
        
        
        #Removal of punctautaion 
        trainDF["text"] = trainDF["text"].str.replace('[^\w\s]','  ')

        
        # split the dataset into training and validation datasets 
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
        
        # label encode the target variable 
        encoder = preprocessing.LabelEncoder()
        train_y = encoder.fit_transform(train_y)
        valid_y = encoder.fit_transform(valid_y)
        
    
        return train_x, train_y, valid_x, valid_y
        
    def train_model_pipeline(self, classifier, train_x, label , valid_x , valid_y):
            # fit the training dataset on the classifier
            
            tfidf_vectorizer = TfidfVectorizer()
            nb = classifier
            #creating pipeline for model mand vectoriser
            tfidf_nv_pipe = Pipeline([('tfidf', tfidf_vectorizer), ('nb', nb)])

            #Fitting the data
            tfidf_nv_pipe.fit(train_x,label)
    
            # predict the labels on validation dataset
            predictions = tfidf_nv_pipe.predict(valid_x)
    
            return metrics.f1_score(predictions, valid_y),tfidf_nv_pipe
    
    def load_model(self, train_x , train_y , valid_x , valid_y):
        
        # Naive Bayes on Word Level TF IDF Vectors
        accuracy,model = self.train_model_pipeline(naive_bayes.MultinomialNB(), train_x, train_y, valid_x , valid_y)
     
        #pickle model
        joblib.dump(model,current_path+"/model/sentiment_analysis_naive_bayes_model.pkl")
        print("NB, WordLevel TF-IDF ACCURACY: ", accuracy)
        
        return model
    
    def prediction(self,input_str):
        try:
            loaded_model = joblib.load("/Users/seethu-8363/Documents/Test/virenv1/model/sentiment_analysis_naive_bayes_model.pkl")
            res = loaded_model.predict([input_str])
        except:
            train_x , train_y , valid_x , valid_y = self.data_preproc()
            self.load_model(train_x , train_y , valid_x , valid_y)
            loaded_model = joblib.load("/Users/seethu-8363/Documents/Test/virenv1/model/sentiment_analysis_naive_bayes_model.pkl")
            res = loaded_model.predict([input_str])
            
        return res


# In[ ]:




