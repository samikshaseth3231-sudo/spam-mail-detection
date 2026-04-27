import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Page config
st.set_page_config(page_title="Spam Classifier", page_icon="📩")

# Dark UI + styling
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #00FFAA;
        }
        .subtitle {
            text-align: center;
            color: #AAAAAA;
            margin-bottom: 20px;
        }
        .box {
            padding: 20px;
            border-radius: 12px;
            background-color: #1c1f26;
        }
        .stTextArea textarea {
            background-color: #2b2f38;
            color: white;
        }
        .stButton>button {
            width: 100%;
            border-radius: 10px;
            height: 45px;
            background-color: #00FFAA;
            color: black;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = str(text).lower()
    words = text.split()
    
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in stop_words]
    words = [ps.stem(w) for w in words]
    
    return " ".join(words)  

# Load model (UNCHANGED)
model = pickle.load(open('model.pk1', 'rb'))
tf = pickle.load(open('vectorizer.pk1', 'rb'))

# UI
st.markdown('<div class="title">📩 Spam Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Detect spam messages instantly</div>', unsafe_allow_html=True)

st.markdown('<div class="box">', unsafe_allow_html=True)

input_sms = st.text_area("✉️ Enter your message")

if st.button("🔍 Predict"):
    transformed_sms = transform_text(input_sms)
    vector_input = tf.transform([transformed_sms])
    result = model.predict(vector_input)[0]

    st.markdown("---")

    if result == 1:
        st.error("🚨 Spam Message Detected!")
        st.image("https://cdn-icons-png.flaticon.com/512/564/564619.png", width=120)
    else:
        st.success("✅ Not Spam Message!")
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=120)

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Made with ❤️ using Streamlit")