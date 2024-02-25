import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_and_vectorize(data_path):
    # Load the dataset
    data = pd.read_csv(data_path)

   
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X = tfidf_vectorizer.fit_transform(data['comment'])
    y = data['class']
    
    return X, y, tfidf_vectorizer  # Return the trained TF-IDF vectorizer along with X and y , so no need to serialize it separately

# Example usage
if __name__ == "__main__":
    X, y, tfidf_vectorizer = preprocess_and_vectorize('comments_dataset.csv')
    # Save X, y, and tfidf_vectorizer to files or pass them to the model training function
    with open('tfidf_vectorizer.pkl', 'wb') as file:
        pickle.dump(tfidf_vectorizer, file)




