import os
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib  

# Download required NLTK data
nltk.download('punkt')

class SentimentAnalysis:
    def __init__(self, data_path=None, model_path='svm_model.pkl', n_splits=5):
        self.data_path = data_path
        self.model_path = model_path
        self.vectorizer = CountVectorizer()
        self.model = SVC(kernel='linear')
        self.n_splits = n_splits  # Number of splits for K-Fold Cross-Validation
        
        # Initialize Indonesian stopword remover and stemmer
        factory = StopWordRemoverFactory()
        self.stopword = factory.create_stop_word_remover()
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
    def load_data(self):
        if self.data_path:
            try:
                self.df = pd.read_csv(self.data_path)
            except FileNotFoundError:
                raise FileNotFoundError("The data file was not found. Please check the path.")
            except Exception as e:
                raise Exception(f"An error occurred while loading the data: {e}")
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
        # Preprocess text data
        self.df['cleaned_text'] = self.df['text'].apply(self.preprocess_text)
        self.X = self.df['cleaned_text']
        self.y = self.df['sentiment']
        print("Cleaned Text Data:\n", self.X.head())
        self.X_vec = self.vectorizer.fit_transform(self.X)
    
    def train_and_evaluate_kfold(self):
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        accuracies = []
        
        for train_index, test_index in kf.split(self.X_vec):
            X_train, X_test = self.X_vec[train_index], self.X_vec[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            accuracies.append(accuracy)
            
            print("Fold Accuracy:", accuracy)
            print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        
        print("Average Accuracy:", sum(accuracies) / len(accuracies))
    
    def save_model(self):
        joblib.dump((self.model, self.vectorizer), self.model_path)
        print(f"Model and vectorizer saved to {self.model_path}")
    
    def load_model(self):
        self.model, self.vectorizer = joblib.load(self.model_path)
        print(f"Model and vectorizer loaded from {self.model_path}")
    
    def analyze_sentiment(self, text):
        cleaned_text = self.preprocess_text(text)
        vectorized_text = self.vectorizer.transform([cleaned_text])
        prediction = self.model.predict(vectorized_text)
        print(f"Text: {text}")
        print(f"Cleaned Text: {cleaned_text}")
        print(f"Vectorized Text Shape: {vectorized_text.shape}")
        print(f"Prediction: {prediction}")
        return prediction[0]
    
    def run(self):
        self.load_data()
        self.prepare_data()
        self.train_and_evaluate_kfold()
        self.save_model()

# Using the class for training, saving the model, or loading the model and analyzing sentiment
sentiment_analysis = SentimentAnalysis(data_path='sentiment_dataset_id.csv')
sentiment_analysis.run()
