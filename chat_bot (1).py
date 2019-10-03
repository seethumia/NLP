#!/usr/bin/env python
# coding: utf-8

# In[68]:


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
import random


# In[69]:


current_path = os.getcwd()


# In[70]:


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
                    
                    jokes = ('A robber robs a bank, gets all the money and is about to leave, but before that he asks a customer who’s lying on the floor, “Have you seen me rob this bank?”-“Yes, sir,” says the customer and gets promptly shot.-“Have you seen me rob this bank?”the robber asks another customer.-“Absolutely not, sir, but my wife here saw everything!”'
                    ,'Dentist: “This will hurt a little.”\nPatient: “OK.”\nDentist: “I’ve been having an affair with your wife for a while now.”','Why can’t a bike stand on its own? It’s two tired.','A nurse told me "Sorry for the wait!" I replied "Its alright Im patient."Son: Dad, what is an idiot?Dad: An idiot is a person who tries to explain his ideas in such a strange and long way that another person who is listening to him cant understand him. Do you understand me?Son: No.')
                    
                    num = random.randrange(0,4)
                    
                    return jokes[num]
                    
                elif "hi" in inp_str:
                    return "wassuppp"
                
                elif "friend" in inp_str:
                    return "thats so sweet of u"
                
                else:
                    return "sorry u lost me"    
               
            elif res == 2 :
                songs = ('Thats why we seize the moment, and try to freeze it and own itSqueeze it and hold it cause we consider these minutes golden - Eminem - Sing for the moment \nhttps://www.youtube.com/watch?v=D4hAVemuQXY','"I have died everyday, waiting for you Darling dont be afraid, I have loved you for a thousand years Ill love you for a thousand more"- Christina Perri, A thousand years \nhttps://www.youtube.com/watch?v=rtOvBOTyX00' , 'Let it go, let it go, cant hold back anymore..... song link https://youtu.be/L0MK7qz13bU','Life is worth living\nLife is worth living, so live another day \nThe meaning of forgiveness\nPeople make mistake, doesnt mean\nyou have to give in\nLife is worth living again\nSong by : Justin Bieber\nLink : https://youtu.be/e934LuQlAeg ')
                num = random.randrange(0,3)
                    
                return songs[num]
            
            elif res == 3 :
                return "fun facts"
           
            elif res == 4 :
                return "Now now that's personel isn't it"
            
            elif res == 5 :
                return "Perhaps a joke could cheer u up....!!! \nSon: Dad, what is an idiot?\nDad: An idiot is a person who tries to explain his ideas in such a strange and long way that another person who is listening to him can't understand him. Do you understand me?\nSon: No."
            
            elif res == 6 : 
                 return "I could sing,crack a joke or two.Not to brag but care for some fun facts.I'm ur guy bro"
            
            
                
            


# In[79]:


#Multinomial NB
obj1 = chit_chat()
obj1.predictions("boring")




inp_str = "hii!!!"
any(trainDF[trainDF.text.apply(lambda x : inp_str in x)])  





