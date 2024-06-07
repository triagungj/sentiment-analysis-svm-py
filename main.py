import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import joblib

# Download required NLTK data
nltk.download('punkt')

class SentimentAnalysis:
    def __init__(self, data_path=None):
        self.data_path = data_path
        self.vectorizer = CountVectorizer()
        
        # Initialize Indonesian stopword remover and stemmer
        factory = StopWordRemoverFactory()
        self.stopword = factory.create_stop_word_remover()
        stemmer_factory = StemmerFactory()
        self.stemmer = stemmer_factory.create_stemmer()
        
        # DataFrame to store preprocessing steps
        self.preprocessing_steps = []
    
    def load_data(self):
        if self.data_path:
            self.df = pd.read_csv(self.data_path)
        else:
            raise ValueError("Data path is not provided.")
    
    def preprocess_text(self, text, index):
        original_text = text
        text = text.lower()
        lower_text = text
        text = self.stopword.remove(text)
        stopword_removed_text = text
        text = self.stemmer.stem(text)
        stemmed_text = text
        tokens = word_tokenize(text)
        tokens = [word for word in tokens if word.isalnum()]
        final_text = ' '.join(tokens)
        
        # Append each step to the list
        self.preprocessing_steps.append({
            'index': index,
            'original_text': original_text,
            'lower_text': lower_text,
            'stopword_removed_text': stopword_removed_text,
            'stemmed_text': stemmed_text,
            'final_text': final_text
        })
        
        return final_text
    
    def prepare_data(self):
        # Preprocess text data
        self.df['cleaned_text'] = self.df['text'].apply(lambda x: self.preprocess_text(x, self.df.index[self.df['text'] == x].tolist()[0]))
        
        # Convert the list of dicts to a DataFrame and save to CSV
        preprocessing_steps_df = pd.DataFrame(self.preprocessing_steps)
        preprocessing_steps_df.to_csv('preprocessing_steps.csv', index=False)
        
        self.X = self.df['cleaned_text']
        self.y = self.df['sentiment']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.1, random_state=42)
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
    
    def train_evaluate_model(self, kernel='linear'):
        model = SVC(kernel=kernel)
        model.fit(self.X_train_vec, self.y_train)
        
        # Calculate training accuracy
        y_train_pred = model.predict(self.X_train_vec)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # Calculate testing accuracy
        y_pred = model.predict(self.X_test_vec)
        test_accuracy = accuracy_score(self.y_test, y_pred)
        
        print(f"Kernel: {kernel}")
        print("Training Accuracy:", train_accuracy)
        print("Testing Accuracy:", test_accuracy)
        print("Classification Report:\n", classification_report(self.y_test, y_pred, zero_division=0))
        
        # Generate and print confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - Kernel: {kernel}')
        plt.show()
        
        return train_accuracy, test_accuracy
    
    def run(self):
        self.load_data()
        self.prepare_data()
        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        results = {}
        for kernel in kernels:
            train_acc, test_acc = self.train_evaluate_model(kernel=kernel)
            results[kernel] = {'train_accuracy': train_acc, 'test_accuracy': test_acc}
        print("Comparison of Kernel Performance:")
        for kernel, metrics in results.items():
            print(f"Kernel: {kernel}, Train Accuracy: {metrics['train_accuracy']}, Test Accuracy: {metrics['test_accuracy']}")
        
# Using the class for training and comparing different kernels
sentiment_analysis = SentimentAnalysis(data_path='sentiment_dataset_id.csv')
sentiment_analysis.run()
