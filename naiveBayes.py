import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Peta untuk konversi output numerik kembali ke label teks
SENTIMENT_MAP = {1: 'GERD', 0: 'MAAG'}

data = {
    'Keluhan': [
        "Saya merasa nyeri di ulu hati", "Ulu hati saya terasa nyeri dan mual",
        "Saya cepat kenyang dan setiap sehabis makan terasa mual dan kembung",
        "Perut saya sakit saat saya telat makan sedikit", "Belum makan dari pagi dan sekarang terasa nyeri di ulu hati dan mual",
        "Dada saya sesak dan terasa panas", "Dada saya terasa terbakar dan mulut saya pahit",
        "Saat membungkuk setelah makan, makanan terasa naik kembali ke mulut",
        "Dada saya merasa panas dan napas saya bau menganggu",
        "Perut saya sakit sampai ke dada, dan saat menelan makanan terasa ada yang mengganjal di tenggorokan",
        "Saya muntah-muntah dari pagi walaupun belum makan apapun",
        "Saat berbaring setelah makan, makanan yang dimakan terasa naik kembali dan mengakibatkan mual hebat"
    ],
    'Diagnosis': [0,0,0,0,0,1,1,1,1,1,0,1] # 0: MAAG, 1: GERD
}

df = pd.DataFrame(data)
X = df['Keluhan']
y = df['Diagnosis']

# Menggunakan test_size 0.3 untuk pelatihan yang lebih baik
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Feature Extraction (Vectorization)
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train_text)

# 4. Pelatihan Model Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# =================================================================
# LANGKAH EKSPOR DATA UNTUK JAVASCRIPT
# =================================================================
import json

vocab = vectorizer.get_feature_names_out().tolist()
class_log_prior = model.class_log_prior_.tolist()
feature_log_prob = model.feature_log_prob_.tolist()

# Simpan hasil ekspor dalam format yang mudah disalin
export_data = {
    "VOCABULARY_MAP": vocab,
    "CLASS_LOG_PRIOR": class_log_prior,
    "FEATURE_LOG_PROB": feature_log_prob
}

print("\n--- DATA EKSPOR UNTUK JAVASCRIPT ---")
# Output JSON ini akan mempermudah penyalinan data model ke JS
print(json.dumps(export_data, indent=4))