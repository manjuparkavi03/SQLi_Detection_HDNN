import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Bidirectional
import pickle
import os

# 1. Dataset Path
file_path = 'data/SQLiV3.csv' 

if os.path.exists(file_path):
    print(f"✅ Dataset found! Loading...")
    df = pd.read_csv(file_path)
else:
    print(f"❌ Error: '{file_path}' file-ah kanom!")
    exit()

# 2. Smart Data Cleaning (Zero Samples Error Fix)
print("Original Rows:", len(df))

# Sentence columns-ah string-ah mathi, empty rows-ah remove pannuvom
df['Sentence'] = df['Sentence'].astype(str)
df = df.dropna(subset=['Sentence', 'Label'])

# Label column-la '0' or '1' irukura rows-ah mattum filter pannuvom
# Dataset-la text labels irunthaal athai numbers-ah mathuvom
df['Label'] = df['Label'].astype(str).str.strip() # Remove spaces
df = df[df['Label'].isin(['0', '1'])] # 0 matrum 1 mattum edukum
df['Label'] = df['Label'].astype(int)

print(f"Cleaned Rows: {len(df)} samples found.")

if len(df) == 0:
    print("❌ Error: Dataset empty-ah iruku! Check if 'Label' column has 0 or 1.")
    exit()

# 3. Preprocessing
max_words = 10000 
max_len = 150    

tokenizer = Tokenizer(num_words=max_words, lower=True, oov_token="<OOV>")
tokenizer.fit_on_texts(df['Sentence'])
sequences = tokenizer.texts_to_sequences(df['Sentence'])
X = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
y = df['Label'].values

# 4. Hybrid DNN Architecture
model = Sequential([
    Embedding(max_words, 128), # Removed deprecated input_length
    Conv1D(64, 5, activation='relu'),
    MaxPooling1D(pool_size=4),
    Bidirectional(LSTM(64)), 
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 5. Training
print(f"Training starts...")
model.fit(X, y, epochs=5, batch_size=128, validation_split=0.2)

# 6. Save Files
model.save('sqli_model.h5')
with open('tokenizer.pkl', 'wb') as handle:
    pickle.dump(tokenizer, handle)

print("\n🔥 Success! AI Model trained and saved.")