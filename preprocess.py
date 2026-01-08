import nltk
import re
from nltk.corpus import stopwords

# download stopwords (first time only)
nltk.download('stopwords')

def clean_text(text):
    """
    This function cleans input text by:
    - Lowercasing
    - Removing special characters
    - Removing stopwords
    """
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)
