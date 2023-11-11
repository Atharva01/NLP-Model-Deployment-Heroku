import pandas as pd
#import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
#from sklearn.externals import joblib
import pickle



df = pd.read_csv('data/SMSSpamCollection',sep='\t', names=['label','message'])
X = df['message']
y = df['label']
print(f"Data Loaded Successfully!")

cv = CountVectorizer()
X = cv.fit_transform(X)

pickle.dump(cv,open('transform.pkl','wb'))
print(f"Transform pickle dump saved!")

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size = 0.2)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
print(f'Model Trained')
score = clf.score(X_test,y_test)
print(f"Testing the model-Score:{score}")
filename = 'nlp_model.pkl'
pickle.dump(clf,open(filename,'wb'))
print(f"Model pickle dump saved!")

message = 'Hello Friend!'
data = [message]
vect = cv.transform(data).toarray()
my_prediction = clf.predict(vect)
print(my_prediction)

'''
        {% if prediction == 1 %}
      <h2 style="color: red">Spam</h2>
      {% elif prediction == 0 %}
      <h2 style="color: blue">Not a Spam (It is a Ham)</h2>
      {% endif %}
'''