import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

nltk.download('stopwords')

# 1. Dataset load
df = pd.read_csv("training.csv", encoding='latin-1', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']

# 2. Required columns
df = df[['sentiment', 'text']]

# 3. Balanced dataset
df_neg = df[df['sentiment'] == 0].head(5000)
df_pos = df[df['sentiment'] == 4].head(5000)
df = pd.concat([df_neg, df_pos])

print("Balanced Data:\n", df['sentiment'].value_counts())

# 4. Stopwords
stop_words = set(stopwords.words('english'))

# 5. Cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)

# 6. Apply cleaning
df['clean_text'] = df['text'].apply(clean_text)

# 7. Convert labels
df['sentiment'] = df['sentiment'].replace(4, 1)

# 8. Features & Labels
X = df['clean_text']
y = df['sentiment']

# 9. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 10. TF-IDF
vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ==============================
# 🔥 Model 1: Logistic Regression
# ==============================
lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)

lr_pred = lr_model.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_accuracy)

# ==============================
# 🔥 Model 2: Naive Bayes
# ==============================
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

nb_pred = nb_model.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

print("Naive Bayes Accuracy:", nb_accuracy)

# ==============================
# 🔥 Model 3: SVM
# ==============================
svm_model = LinearSVC()
svm_model.fit(X_train, y_train)

svm_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)

print("SVM Accuracy:", svm_accuracy)