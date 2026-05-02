import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# ==============================
# 🔥 Page Config
# ==============================
st.set_page_config(page_title="Sentiment Analyzer", layout="wide")

# ==============================
# 🔥 Advanced CSS + Animations
# ==============================
st.markdown("""
<style>
body { background-color: #0e1117; }

.main-title {
    font-size: 48px;
    font-weight: 800;
    text-align: center;
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: fadeIn 2s ease-in-out;
}

.sub-title {
    text-align: center;
    color: #A0A0A0;
    margin-bottom: 30px;
    animation: fadeIn 3s ease-in-out;
}

.card {
    background: #1c1f26;
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.3);
    transition: transform 0.3s ease, box-shadow 0.3s ease;
}

.card:hover {
    transform: translateY(-10px) scale(1.05);
    box-shadow: 0px 8px 25px rgba(0,0,0,0.6);
}

.stButton>button {
    background: linear-gradient(90deg, #00C9FF, #92FE9D);
    color: black;
    font-weight: bold;
    border-radius: 8px;
    padding: 10px 20px;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.1);
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ==============================
# 🔥 Title
# ==============================
st.markdown('<div class="main-title">🐦 Tweet Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Analyze tweets using AI & Machine Learning 🚀</div>', unsafe_allow_html=True)

# ==============================
# 🔥 Sidebar
# ==============================
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Logistic Regression", "Naive Bayes", "SVM"]
)

# ==============================
# 🔥 Load Data
# ==============================
@st.cache_data
def load_data():
    df = pd.read_csv("training.csv", encoding='latin-1', header=None)
    df.columns = ['sentiment', 'id', 'date', 'query', 'user', 'text']
    df = df[['sentiment', 'text']]

    df_neg = df[df['sentiment'] == 0].head(5000)
    df_pos = df[df['sentiment'] == 4].head(5000)

    return pd.concat([df_neg, df_pos])

df = load_data()

# ==============================
# 🔥 Preprocessing
# ==============================
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

df['clean_text'] = df['text'].apply(clean_text)
df['sentiment'] = df['sentiment'].replace(4, 1)

# ==============================
# 🔥 Vectorization
# ==============================
X = df['clean_text']
y = df['sentiment']

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# ==============================
# 🔥 Train Models
# ==============================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LogisticRegression()
nb = MultinomialNB()
svm = LinearSVC()

lr.fit(X_train, y_train)
nb.fit(X_train, y_train)
svm.fit(X_train, y_train)

lr_acc = accuracy_score(y_test, lr.predict(X_test))
nb_acc = accuracy_score(y_test, nb.predict(X_test))
svm_acc = accuracy_score(y_test, svm.predict(X_test))

# ==============================
# 🔥 KPI Cards
# ==============================
st.markdown("## 📊 Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f'<div class="card">Logistic Regression<br><h2>{round(lr_acc,3)}</h2></div>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<div class="card">Naive Bayes<br><h2>{round(nb_acc,3)}</h2></div>', unsafe_allow_html=True)

with col3:
    st.markdown(f'<div class="card">SVM<br><h2>{round(svm_acc,3)}</h2></div>', unsafe_allow_html=True)

# ==============================
# 🔥 Model Selection
# ==============================
if model_choice == "Logistic Regression":
    model = lr
elif model_choice == "Naive Bayes":
    model = nb
else:
    model = svm

# ==============================
# 🔥 Sample Tweets (Auto Fill)
# ==============================
st.markdown("### 💡 Try Sample Tweets")

col1, col2, col3 = st.columns(3)

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

with col1:
    if st.button("😊 Positive Example"):
        st.session_state.input_text = "I absolutely love this product! It works perfectly."

with col2:
    if st.button("😞 Negative Example"):
        st.session_state.input_text = "This is the worst experience ever. Totally disappointed."

with col3:
    if st.button("🎲 Random Example"):
        st.session_state.input_text = "The movie was okay, not too bad but not great either."

# ==============================
# 🔥 Input Section
# ==============================
st.markdown("### ✨ Enter text and watch AI predict instantly!")

user_input = st.text_area(
    "Type your tweet here...",
    value=st.session_state.input_text,
    height=120
)

# ==============================
# 🔥 Prediction
# ==============================
if st.button("🚀 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Enter some text!")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]

        if prediction == 1:
            st.success("😊 Positive Sentiment")
        else:
            st.error("😞 Negative Sentiment")

# ==============================
# 🔥 Chart
# ==============================
st.markdown("## 📈 Model Comparison")

st.bar_chart({
    "Logistic Regression": lr_acc,
    "Naive Bayes": nb_acc,
    "SVM": svm_acc
})