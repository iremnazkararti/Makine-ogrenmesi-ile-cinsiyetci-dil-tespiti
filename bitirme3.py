import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from transformers import BertTokenizer, TFBertModel, TFDistilBertModel, BertTokenizerFast
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding
import nltk
import re
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.utils.class_weight import compute_class_weight

# NLTK stopwords yükleniyor
nltk.download('stopwords')
stop_words = stopwords.words('turkish')

# Temizleme fonksiyonu
def temizle(metin):
    metin = metin.lower()  # Küçük harfe çevirme
    metin = re.sub(r'\d+', '', metin)  # Sayıları kaldırma
    metin = re.sub(r'[^\w\s]', '', metin)  # Noktalama işaretlerini kaldırma
    metin = ' '.join([kelime for kelime in metin.split() if kelime not in stop_words])  # Stopwords'ü kaldırma
    return metin

# Veri setini yükleme
def veri_yukle(dosya_yolu):
    return pd.read_excel(dosya_yolu)

# Veri setini yükleme
veri_seti = veri_yukle("C:\\Users\\Naz\\Desktop\\bitirme 2\\temizlenmis_veri_seti_etiketli.xlsx")

# Temizleme işlemi
veri_seti['Temiz_Tweet'] = veri_seti['Tweet'].apply(temizle)

# Özellik (X) ve hedef (y) değişkenlerini belirleme
X = veri_seti['Temiz_Tweet']
y = veri_seti['Cinsiyetçi']

# Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# GloVe embedding yükleme fonksiyonu
def load_glove_embeddings(file_path):
    embeddings = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings

# GloVe embedding dosyasını yükleme (dosya yolu argüman olarak verilmelidir)
glove_embeddings = load_glove_embeddings("C:\\Users\\Naz\\Desktop\\bitirme 2\\glove.6B.100d.txt")

# Metinleri sayısal verilere dönüştürme
def text_to_sequence(texts, glove_embeddings, max_length=100):
    sequences = []
    for text in texts:
        words = text.split()
        word_vectors = []
        for word in words:
            if word in glove_embeddings:
                word_vectors.append(glove_embeddings[word])
        if len(word_vectors) == 0:
            word_vectors.append(np.zeros(100))  # Eğer kelime yoksa sıfır vektör ekleyin
        sequences.append(np.array(word_vectors))
    return pad_sequences(sequences, maxlen=max_length, padding='post')

# Eğitim ve test verisini sayısal verilere dönüştürme
X_train_glove = text_to_sequence(X_train, glove_embeddings)
X_test_glove = text_to_sequence(X_test, glove_embeddings)

# **Sınıf Ağırlıkları Hesaplama**
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# LSTM Modeli
model = Sequential()
model.add(LSTM(100, input_shape=(X_train_glove.shape[1], X_train_glove.shape[2]), return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Modeli eğitme (class_weight ile)
history = model.fit(X_train_glove, y_train, epochs=5, batch_size=64, validation_data=(X_test_glove, y_test), class_weight=class_weight_dict)

# Test verisi ile tahmin yapma
y_pred = (model.predict(X_test_glove) > 0.5)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Sonuçları yazdırma
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')

# Confusion Matrix görselleştirmesi
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Sexist', 'Sexist'],
            yticklabels=['Non-Sexist', 'Sexist'])
plt.title('Confusion Matrix')
plt.xlabel('Tahmin Edilen')
plt.ylabel('Gerçek')
plt.show()
