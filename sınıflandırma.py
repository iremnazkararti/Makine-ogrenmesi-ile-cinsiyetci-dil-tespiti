import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
import nltk
import re
from wordcloud import WordCloud
from imblearn.over_sampling import SMOTE

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords


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

# Veri setini yükleme (relative path kullanılabilir)
def veri_yukle(dosya_yolu):
    return pd.read_excel(dosya_yolu)

# Veri setini yükleme
veri_seti = veri_yukle("C:\\Users\\Naz\\Desktop\\bitirme 2\\temizlenmis_veri_seti_etiketli.xlsx")

# Temizleme işlemi
veri_seti['Temiz_Tweet'] = veri_seti['Tweet'].apply(temizle)

# Özellik (X) ve hedef (y) değişkenlerini belirleme
X = veri_seti['Temiz_Tweet']
y = veri_seti['Cinsiyetçi']

# bazı veri seti görselleştirmeleri

# Cinsiyetçi ve cinsiyetçi olmayan tweetlerin sayısal dağılımı
plt.figure(figsize=(8, 6))
sns.countplot(x='Cinsiyetçi', data=veri_seti, hue='Cinsiyetçi', palette='viridis', legend=False)
plt.title('Cinsiyetçi ve Cinsiyetçi Olmayan Tweet Dağılımı')
plt.xlabel('Cinsiyetçi (1: Evet, 0: Hayır)')
plt.ylabel('Tweet Sayısı')
plt.show()

# Pie chart ile sınıf dağılımı
plt.figure(figsize=(8, 8))
veri_seti['Cinsiyetçi'].value_counts().plot.pie(autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title('Cinsiyetçi ve Cinsiyetçi Olmayan Tweetlerin Dağılımı')
plt.ylabel('')
plt.show()


# NLTK stopwords'ü yükle
nltk.download('stopwords')
stop_words = stopwords.words('turkish')  # Türkçe stopwords listesi

# Cinsiyetçi tweetleri seçme (Cinsiyetçi etiket == 1)
cinsiyetci_tweetler = veri_seti[veri_seti['Cinsiyetçi'] == 1]['Temiz_Tweet']

# TF-IDF uygulama, stop_words parametresine Türkçe stopwords listesini veriyoruz
tfidf = TfidfVectorizer(max_features=10000, stop_words=stop_words)
X_tfidf = tfidf.fit_transform(cinsiyetci_tweetler)


# Cinsiyetçi tweetleri seçme (Cinsiyetçi etiket == 1)
cinsiyetci_tweetler = veri_seti[veri_seti['Cinsiyetçi'] == 1]['Temiz_Tweet']

# TF-IDF uygulama, stop_words parametresine Türkçe stopwords listesini veriyoruz
tfidf = TfidfVectorizer(max_features=10000, stop_words=stop_words)
X_tfidf = tfidf.fit_transform(cinsiyetci_tweetler)


# Cinsiyetçi tweetler için TF-IDF uygulama
tfidf = TfidfVectorizer(max_features=10000, stop_words=stop_words)
X_tfidf = tfidf.fit_transform(cinsiyetci_tweetler)


# "bir" ve "kadar" gibi kelimeleri stopwords listesine ekleme
extra_stopwords = ['bir', 'kadar', 'var','eşcinsel','suriyeli']
stop_words.extend(extra_stopwords)

# Cinsiyetçi tweetler için TF-IDF uygulama
tfidf = TfidfVectorizer(max_features=10000, stop_words=stop_words)
X_tfidf = tfidf.fit_transform(cinsiyetci_tweetler)

# Kelime sıklıklarını hesaplama
kelime_sikliklari = np.asarray(X_tfidf.sum(axis=0)).flatten()
kelime_adi = tfidf.get_feature_names_out()

# Kelime sıklıklarını bir DataFrame'e dönüştürme
kelime_df = pd.DataFrame(list(zip(kelime_adi, kelime_sikliklari)), columns=['Kelime', 'Sıklık'])
kelime_df = kelime_df.sort_values(by='Sıklık', ascending=False)

# Stopwords'leri çıkarma (yeni eklenen kelimelerle birlikte)
kelime_df = kelime_df[~kelime_df['Kelime'].isin(stop_words)]

# İlk 5 kelimeyi al
top_5_kelime = kelime_df.head(5)

# Görsel olarak en sık kullanılan 5 kelimenin bar plot'unu oluşturma
plt.figure(figsize=(10, 6))
sns.barplot(x='Sıklık', y='Kelime', data=top_5_kelime)
plt.title('Cinsiyetçi Tweetlerde En Sık Kullanılan 5 Kelime')
plt.xlabel('Kelime Sıklığı')
plt.ylabel('Kelime')
plt.show()











# TF-IDF uygulama
tfidf = TfidfVectorizer(max_features=10000)  # Max özellik sayısı ile sınırlandırma
X_tfidf = tfidf.fit_transform(X)

# SVD ile boyut indirgeme (3 boyuta indirgeme)
svd = TruncatedSVD(n_components=3, random_state=42)  # Daha fazla boyut
X_svd = svd.fit_transform(X_tfidf)

# SVD sonrası negatif değerleri kontrol etme ve sıfırlama
X_svd = np.abs(X_svd)

# Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_svd, y, test_size=0.3, random_state=42)

# SMOTE ile veri dengeleme
smote = SMOTE(random_state=42, k_neighbors=5)  # K komşu sayısını değiştirebilirsiniz
X_train, y_train = smote.fit_resample(X_train, y_train)

# Modelleri tanımlama
models = {
    'SVM': SVC(C=10, kernel='rbf', random_state=42, max_iter=2000),  # SVM için kernel değiştirildi ve C değeri arttırıldı
    'Naive Bayes': MultinomialNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Neural Network (SVD)': MLPClassifier(hidden_layer_sizes=(100,), max_iter=2000, random_state=42),  # Neural Network için max_iter arttırıldı
    'Logistic Regression': LogisticRegression(random_state=42)  # Lojistik Regresyon modeli eklendi
}



# Sonuçları tutmak için boş bir DataFrame oluşturuyorum
results = []

# Modelleri eğitme ve doğruluklarını hesaplama
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append({
        'Model': model_name,
        'Doğruluk': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    })

# Sonuçları DataFrame olarak gösterme
results_df = pd.DataFrame(results)
print(results_df)

# Grafiksel gösterim (Doğruluk karşılaştırması)
plt.figure(figsize=(13, 6))
sns.barplot(x='Model', y='Doğruluk', data=results_df)
plt.title('Modellerin Doğruluk Karşılaştırması')
plt.show()

# Confusion Matrix görselleştirmesi için tüm matrisleri tek bir grafikte gösterme
plt.figure(figsize=(15, 10))  # Figür boyutunu ayarla

for i, (model_name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    plt.subplot(2, 3, i + 1)  # Alt grafik düzeni (örneğin, 2x3 grid)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Sexist', 'Sexist'],
                yticklabels=['Non-Sexist', 'Sexist'])
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Tahmin Edilen')
    plt.ylabel('Gerçek')

plt.tight_layout()  # Alt grafiklerin çakışmaması için düzenleme
plt.show()

# Cinsiyetçi ve cinsiyetçi olmayan tweetlerin kelime bulutlarını oluşturma
# Cinsiyetçi tweetler
cinsiyetci_tweetler = veri_seti[veri_seti['Cinsiyetçi'] == 1]['Temiz_Tweet']
cinsiyetci_kelime_bulutu = ' '.join(cinsiyetci_tweetler)
cinsiyetci_wc = WordCloud(width=800, height=400, background_color='white').generate(cinsiyetci_kelime_bulutu)

# Cinsiyetçi olmayan tweetler
non_cinsiyetci_tweetler = veri_seti[veri_seti['Cinsiyetçi'] == 0]['Temiz_Tweet']
non_cinsiyetci_kelime_bulutu = ' '.join(non_cinsiyetci_tweetler)
non_cinsiyetci_wc = WordCloud(width=800, height=400, background_color='white').generate(non_cinsiyetci_kelime_bulutu)

# Cinsiyetçi tweetlerin kelime bulutunu görselleştirme
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cinsiyetci_wc, interpolation='bilinear')
plt.title('Cinsiyetçi Tweetler')
plt.axis('off')

# Cinsiyetçi olmayan tweetlerin kelime bulutunu görselleştirme
plt.subplot(1, 2, 2)
plt.imshow(non_cinsiyetci_wc, interpolation='bilinear')
plt.title('Cinsiyetçi Olmayan Tweetler')
plt.axis('off')



plt.show()
