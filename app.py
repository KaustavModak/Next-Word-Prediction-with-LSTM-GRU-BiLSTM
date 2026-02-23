import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# model=load_model('next_word_lstm.h5')

with open('tokenizer.pickle','rb') as handle:
    tokenizer=pickle.load(handle)

def load_selected_model(choice):
    if choice == "LSTM":
        return load_model("next_word_lstm.keras", compile=False)
    elif choice == "GRU":
        return load_model("next_word_gru.keras", compile=False)
    elif choice == "BiLSTM":
        return load_model("next_word_bilstm.keras", compile=False)

def predict_next_word(model,tokenizer,text,max_len,n):
    for i in range(n):
        token_text=tokenizer.texts_to_sequences([text])[0]
        padded_text=pad_sequences([token_text],padding='pre',maxlen=max_len-1)
        pos=np.argmax(model.predict(padded_text))
        for word,index in tokenizer.word_index.items():
            if index==pos:
                text=text+" "+word
    return text

st.title("Next Word Prediction with LSTM/GRU/BiLSTM")
model_choice = st.selectbox(
    "Choose the model:",
    ["LSTM", "GRU", "BiLSTM"]
)
model = load_selected_model(model_choice)
input_text=st.text_input("Enter the sequence","To be or not to be")
input_no_of_words=st.number_input("Enter the number of next words to be predicted",min_value=1)
if st.button("Predict the next words"):
    max_len=model.input_shape[1]+1    # max_len is always model.input_shape[1] + 1 (13+1=14 which was previously the max_len)
    # model.input_shape only contains the size of i/ps but max_len contains both i/ps and 1 o/p, thus max_len is always model.input_shape
    output_text=predict_next_word(model,tokenizer,input_text,max_len,input_no_of_words)
    st.write(f"{output_text}")







