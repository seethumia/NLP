#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV


# In[7]:


current_path = os.getcwd()


# In[8]:


class chit_chat:
    
    def data_preproc(self):
        
        #Loading the data
        data = open('chit_chat.txt').read()
        
        labels, texts = [], []
        for i, line in enumerate(data.split("\n")):
            content = line.split()
            labels.append(content[0])
            texts.append(" ".join(content[1:]))
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
        
        
#         #Lemmatization
#         lemmatizer = WordNetLemmatizer()
#         trainDF['text'] = lemmatizer.lemmatize(trainDF["text"])
    
        
        # split the dataset into training and validation datasets 
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size = 0.6, random_state = 42)
        
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
    
            return metrics.f1_score(predictions,valid_y,average = 'micro') , tfidf_nv_pipe
    

    def load_model(self, train_x , train_y , valid_x , valid_y):
        
        # Naive Bayes Multinomial on Word Level TF IDF Vectors
        accuracy,model = self.train_model_pipeline(naive_bayes.MultinomialNB(), train_x, train_y, valid_x , valid_y)
     
        #pickle model
        joblib.dump(model,current_path+"/model/chat_bot_firstCut.pkl")
        print("NB, WordLevel TF-IDF ACCURACY: ", accuracy)
        
        return model,accuracy
    
    def predictions(self,inp_str):
            try:
                loaded_model = joblib.load("/Users/seethu-8363/Documents/Test/virenv1/model/chat_bot_firstCut.pkl")
                res = loaded_model.predict([inp_str])
            except:
                train_x , train_y , valid_x , valid_y = self.data_preproc()
                model = self.load_model(train_x , train_y , valid_x , valid_y)
                loaded_model = joblib.load("/Users/seethu-8363/Documents/Test/virenv1/model/chat_bot_firstCut.pkl")
                res = loaded_model.predict([inp_str])
            
            
            if res == 0 : 
                return "Howdy partner,Hope you are having a fun day...How may i assist u?"
            
            elif res == 1 :
                a = ['fun','funny','joke','joking']
                if any(x in inp_str for x in a):
                    return "jokes"
                elif "hi" in inp_str:
                    return "wassuppp"
                elif "friend" in inp_str:
                    return "thats so sweet of u"
                else:
                    return "sorry u lost me"    
               
            elif res == 2 :
                return "some song"
            
            elif res == 3 :
                return "fun facts"
           
            elif res == 4 :
                return "Now now that's personel isn't it"
            
            elif res == 5 :
                return "Perhaps a song could cheer u up...."
            
            elif res == 6 : 
                 return "I could sing,crack a joke or two.Not to brag but care for some fun facts.I'm ur guy bro"
            
            
                
            


# In[12]:


#Multinomial NB
obj = chit_chat()
obj.predictions("wht can u do")


# In[ ]:





# In[ ]:




