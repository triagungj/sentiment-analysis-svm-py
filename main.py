import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib  

nltk.download('punkt')

class SentimentAnalysis:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.vectorizer = CountVectorizer()
        self.model = SVC(kernel='linear')
        
        # Initialize Indonesian stopword remover and stemmer
        factory = StopWordRemoverFactory()
        self.stopword = factory.create_stop_word_remover()
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
    def load_data(self):
        if self.data_path:
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Data path is not provided.")
    
    def preprocess_text(self, text):
        text = text.lower()
        text = self.stopword.remove(text)
        text = self.stemmer.stem(text)
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        return ' '.join(tokens)
    
    def prepare_data(self):
        self.df['cleaned_text'] = self.df['text'].apply(self.preprocess_text)
        self.X = self.df['cleaned_text']
        self.y = self.df['sentiment']
        print(self.X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
    
    def train_model(self):
        self.model.fit(self.X_train_vec, self.y_train)
    
    def evaluate_model(self):
        y_pred = self.model.predict(self.X_test_vec)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("Classification Report:\n", classification_report(self.y_test, y_pred, zero_division=0))
    
    def save_model(self, model_path):
        joblib.dump((self.model, self.vectorizer), model_path)
        print(f"Model and vectorizer saved to {model_path}")
    
    def load_model(self, model_path):
        self.model, self.vectorizer = joblib.load(model_path)
        print(f"Model and vectorizer loaded from {model_path}")
    
    def analyze_sentiment(self, text):
        cleaned_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)
        print(f"Text: {text}")
        print(f"Cleaned Text: {cleaned_text}")
        print(f"Vectorized Text: {vectorized_text}")
        print(f"Prediction: {prediction}")
        return prediction[0]
    
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_model()
        self.evaluate_model()

# Menggunakan kelas untuk pelatihan dan penyimpanan model
sentiment_analysis = SentimentAnalysis(data_path='sentiment_dataset_id.csv')
sentiment_analysis.run()
sentiment_analysis.save_model('svm_model.pkl')

# Memuat model dan menganalisis sentimen
sentiment_analysis.load_model('svm_model.pkl')
sample_text = "Industri Pinjol Rugi di Q1, Alarm Bank Bunyi!"
predicted_sentiment = sentiment_analysis.analyze_sentiment(sample_text)
print(f"Predicted Sentiment: {predicted_sentiment}")
