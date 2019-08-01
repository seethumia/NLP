import pandas as pd
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer , TfidfTransformer
from sklearn import decomposition, ensemble
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import os
from sklearn.externals import joblib 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV

current_path = os.getcwd()

class naive_bayes_sentiment:
    def data_preproc(self):
        
        #Loading the data
        data = open('/Users/seethu-8363/Downloads/amazon_reviews.csv').read()
        
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

        
        # split the dataset into training and validation datasets 
        train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size = 0.8)
        
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
    
            return metrics.f1_score(predictions, valid_y , average = 'micro'),tfidf_nv_pipe 
    
    def load_svm_model(self, train_x , train_y , valid_x , valid_y):
            
       # SVM RBC on Word Level TF IDF Vectors with Grid Search
        accuracy,model = self.train_model_pipeline(svm.SVC(kernel = 'rbf'), train_x, train_y, valid_x , valid_y) 
        return accuracy,model
        
    def prediction_svm(self,input_str):
            train_x , train_y , valid_x , valid_y = self.data_preproc()
            accuracy = self.load_svm_model(train_x , train_y , valid_x , valid_y)
            print("NB, WordLevel TF-IDF ACCURACY: ", accuracy)
            #loaded_model = joblib.load("/Users/seethu-8363/Documents/Test/virenv1/model/sentiment_analysis_naive_bayes_model.pkl")




#SVC
obj = naive_bayes_sentiment()
obj.prediction_svm('good')



Grid Search
#Loading the data
data = open('/Users/seethu-8363/Downloads/amazon_reviews.csv').read()
        
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    labels.append(content[0])
    texts.append(" ".join(content[1:]))
​
# create a dataframe using texts and lables
trainDF = pd.DataFrame()
trainDF['text'] = texts
trainDF['label'] = labels
​
​
​
# split the dataset into training and validation datasets 
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size = 0.8)
​
​
# count_vect = CountVectorizer()
# X_train_counts = count_vect.fit_transform(train_x)
# tfidf_transformer = TfidfTransformer()
# X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
# clf = MultinomialNB().fit(X_train_tfidf, y_train)
# label encode the target variable 
​
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
        
#vectorizer
​
#Fit the vectorizer on train data and tranform the vectorizer on train and test
​
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(train_x)
vec_train_data = tfidf_vectorizer.transform(train_x)
vec_test_data = tfidf_vectorizer.transform(valid_x)
​
​
model = svm.SVC(kernel = 'rbf')
​
#Making a grid based on the values provided by Random Search
param_grid = {
    'C' : [0.001, 0.01, 0.1, 1, 10],
    'gamma' : [0.001, 0.01, 0.1, 1]
            }
        
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = model, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2,scoring = 'f1')
​
grid_search.fit(vec_train_data,train_y)
grid_search.best_params_
​
Fitting 3 folds for each of 20 candidates, totalling 60 fits
[Parallel(n_jobs=-1)]: Using backend LokyBackend with 12 concurrent workers.
[Parallel(n_jobs=-1)]: Done  17 tasks      | elapsed:    4.4s
[Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed:    9.8s finished
{'C': 10, 'gamma': 1}
model = svm.SVC(C = 10 , kernel = 'rbf',gamma= 1)
model.fit(vec_train_data,train_y)
prediction = model.predict(vec_test_data)
print(metrics.f1_score(prediction, valid_y))
​
#SVM Model
#accuarcy before grid search = 0.48575
#accuracy after grid search = 0.8399846015655076
0.8399846015655076
​
vec_test_data.shape
(8000, 13675)
vec_train_data.shape
(2000, 13675)
GRid search
svm_model = svm.SVC(kernel = 'rbf')
svm_model.get_params().keys()
dict_keys(['C', 'cache_size', 'class_weight', 'coef0', 'decision_function_shape', 'degree', 'gamma', 'kernel', 'max_iter', 'probability', 'random_state', 'shrinking', 'tol', 'verbose'])
import sklearn
sklearn.metrics.SCORERS.keys()
dict_keys(['explained_variance', 'r2', 'max_error', 'neg_median_absolute_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'accuracy', 'roc_auc', 'balanced_accuracy', 'average_precision', 'neg_log_loss', 'brier_score_loss', 'adjusted_rand_score', 'homogeneity_score', 'completeness_score', 'v_measure_score', 'mutual_info_score', 'adjusted_mutual_info_score', 'normalized_mutual_info_score', 'fowlkes_mallows_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted'])
#Making a grid based on the values provided by Random Search
​
param_grid = {
    Cs = [0.001, 0.01, 0.1, 1, 10]
    gammas = [0.001, 0.01, 0.1, 1]
}
#Create a based model
rf = RandomForestRegressor()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 2)
​
# Fit the grid search to the data
grid_search.fit(X_train, y_train)
grid_search.best_params_
​
​
Cs = [0.001, 0.01, 0.1, 1, 10]
gammas = [0.001, 0.01, 0.1, 1]
