# model_training.py

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from feature_engineering import preprocess_and_vectorize  # Import the preprocessing function from feature_engineering.py
import pickle

def train_model(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Choose a model and train it
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    save_model(model, 'trained_model.pkl')
def save_model(model, model_path):              #Model is serialized to test the model in unseen data without ground truth values
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)

# Example usage
if __name__ == "__main__":
    X, y,tfidf_vectorizer = preprocess_and_vectorize('comments_dataset.csv')
    train_model(X, y)
