import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
  text=text.lower()
  text=nltk.word_tokenize(text)
  txt=[]
  for i in text:
    if  i.isalnum():
      txt.append(i)

  text=txt[:]
  txt.clear()
  for i in text:
    if i not in stopwords.words('english') and i not in string.punctuation:
      txt.append(i)  
      
  text=txt[:]
  txt.clear()    
  for i in text:
    txt.append(ps.stem(i))

  return " ".join(txt)

tfdf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfdf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")