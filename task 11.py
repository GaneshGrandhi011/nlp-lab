import re
import requests
import torch
import torch.nn as nn
from bs4 import BeautifulSoup
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

class ChunkerModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return torch.sigmoid(self.fc(last_hidden))

def fetch_text(url):
    html = requests.get(url).text
    soup = BeautifulSoup(html, 'html.parser')
    paragraphs = [p.get_text() for p in soup.find_all('p')]
    text = ' '.join(paragraphs)
    return re.sub(r'\s+', ' ', text)

def preprocess(text):
    tok = Tokenizer(num_words=5000)
    tok.fit_on_texts([text])
    sequences = tok.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    return padded, tok

def segment_text(url):
    text = fetch_text(url)
    seq, tok = preprocess(text)
    return text.split('. ')[:5]

print("""Natural language processing (NLP) is a subfield of computer science and especially artificial intelligence.
It is primarily concerned with providing computers with the ability to process data encoded in natural
language and is thus closely related to information retrieval, knowledge representation and
computational linguistics, a subfield of linguistics.
Major tasks in natural language processing are speech recognition, text classification, natural-language
understanding, and natural-language generation.""")

print(*segment_text("https://en.wikipedia.org/wiki/Natural_language_processing"), sep="\n")
