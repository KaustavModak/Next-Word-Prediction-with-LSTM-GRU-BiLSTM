## Next Word Prediction using LSTM, GRU & BiLSTM

This project demonstrates Next-Word Prediction using three powerful deep learning architectures:

- LSTM (Long Short-Term Memory)

- GRU (Gated Recurrent Unit)

- BiLSTM (Bidirectional LSTM)

Given an input sequence of words, the model predicts the next word.
A Streamlit application is provided where you can choose the model (LSTM / GRU / BiLSTM) and generate new text.

## ⭐ Project Overview

Next-word prediction is a classic language modeling problem where a model is trained on large text data to learn grammatical structure, vocabulary relationships, and contextual meaning.
The goal is to generate the most probable next word based on prior words.

This project includes:

- Data preprocessing

- Tokenization

- Sequence generation

- Model training (LSTM, GRU, BiLSTM)

- Saving trained models

- Streamlit inference app

## 🧠 Model Architectures Explained

Below are detailed explanations of all three models, written in simple and clear terms.

### 1️⃣ LSTM (Long Short-Term Memory)

LSTM is a special variant of RNN designed to solve the vanishing gradient problem in traditional RNNs.
It is widely used for sequential tasks such as next-word prediction, speech recognition, machine translation, etc.

Each LSTM cell contains:

- Input gate – decides what new information to store

- Forget gate – decides what information to discard

- Output gate – controls the final output

LSTM can retain long-term dependencies, meaning it remembers words that appeared far earlier in the sentence.

In this project, the LSTM:

- Learns patterns in text

- Outputs a probability distribution over the vocabulary

- Picks the most likely next word

Because of its memory mechanism, LSTM produces smooth, context-aware predictions.

### 2️⃣ GRU (Gated Recurrent Unit)

GRU is another RNN architecture similar to LSTM but simpler and faster.
It uses two gates:

- Update gate – how much of past information to keep

- Reset gate – how much of previous information to forget

Unlike LSTM, GRU does not have a separate memory cell.
This makes GRUs:

- Computationally more efficient

- Faster to train

- Lighter (fewer parameters)

Despite being simpler, GRUs perform extremely well in NLP tasks.

In next-word prediction:

- GRUs learn text structure

- Predict the most probable next word

- Handle long sequences efficiently

GRUs are ideal when training speed and performance both matter.

### 3️⃣ BiLSTM (Bidirectional LSTM)

BiLSTM is an extension of LSTM where the input sequence is processed in both forward and backward directions.

It consists of:

- Forward LSTM

- Backward LSTM

- Combined output

This allows the model to learn:

- Past context

- Future context

In language modeling, this is powerful because meaning often depends on surrounding words.

For next-word prediction:

- BiLSTM captures richer semantics

- Learns deeper word dependencies

- Produces more accurate predictions

BiLSTM is the most advanced architecture used in this project.

## 🚀 How the App Works

The Streamlit app allows you to:

Select model: LSTM / GRU / BiLSTM

Enter input text

Select how many words you want to generate

Get predicted text in real time

The app uses:

- Saved model (.h5 files)

- Tokenizer (tokenizer.pickle)

- Padding + softmax predictions

Greedy decoding using argmax

## 📦 Project Structure
📁 Next-Word-Prediction/
│
├── next_word_lstm.h5
├── next_word_gru.h5
├── next_word_bilstm.h5
├── tokenizer.pickle
├── app.py                 # Streamlit App
├── training_notebook.ipynb
└── README.md

## 🛠️ Installation & Setup
1. Clone the repository
git clone https://github.com/yourusername/Next-Word-Prediction-using-LSTM-GRU-BiLSTM.git
cd Next-Word-Prediction-using-LSTM-GRU-BiLSTM

2. Create virtual environment
python -m venv venv
venv\Scripts\activate    # Windows

3. Install dependencies
pip install -r requirements.txt

4. Run Streamlit
streamlit run app.py

## 💡 Usage

- Open the Streamlit UI

 Choose your model (LSTM, GRU, BiLSTM)

- Enter initial text

- Choose how many words to predict

- Get generated output instantly

## 📚 Technologies Used

- TensorFlow / Keras

- Streamlit

- NumPy

- NLTK / Text Processing

- Python 3.x

## 🧪 Future Improvements

- Add beam search for better predictions

- Deploy on HuggingFace Spaces

- Add transformer-based models (GPT-2 / BERT)

- Add temperature sampling for creative text generation
