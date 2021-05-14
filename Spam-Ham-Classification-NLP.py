# # SMS CLASSIFICATION USING NLP

# importing necessary libraries
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# Loading the input dataset
dframe = pd.read_csv(r'./sms-dataset.csv')

# Data Cleaning
# Dropping unnecessary columns
dframe.drop(columns= ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)

def convert_class_to_num(text):
    if text == 'ham':
        return 1
    else:
        return 0

# Converting the class names to numerics
dframe['class']=dframe['class'].apply(convert_class_to_num)

# Segregating the output from input features
X = dframe['message']
y = dframe['class']

# Bag of Words Calculation
cv = CountVectorizer()
X = cv.fit_transform(X)

# Splitting the data into training and test dataset
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.3,random_state = 42)

# Model building and training
nlp_model = MultinomialNB()
nlp_model.fit(X_train,y_train)

# Serializing the model
pickle.dump(nlp_model,open('nlp_model.pkl','wb'))