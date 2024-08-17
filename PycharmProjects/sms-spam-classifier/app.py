import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
nltk.download('punkt')



ps = PorterStemmer()
def transform_text(text):
#Tokenize and lower case
  text = text.lower()
  text = nltk.word_tokenize(text)
#Check special Characters
  y = []
  for i in text:
    if i.isalnum():
      y.append(i)

  text = y[:]
  y.clear()
#Removing stopwords and Punctuation
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      y.append(i)
  text = y[:]
  y.clear()
#Stemming converting into base word
  for i in text:
    y.append(ps.stem(i))
  return ' '.join(y)




tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model (2).pkl','rb'))

st.title("SMS Spam|Ham Detector ")
input_msg = st.text_area("Please Enter the msg or Email ")
if st.button("Predict"):
    #text preprocessing
    transformed_msg = transform_text(input_msg)
    #vectorization
    vectorized_msg = tfidf.transform([transformed_msg])
    #model prediction
    result_prediction = model.predict(vectorized_msg)[0]
    #Display the result
    if result_prediction == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


