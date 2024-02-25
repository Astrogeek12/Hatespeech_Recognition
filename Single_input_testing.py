import pandas as pd
import pickle
import re

def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def load_vectorizer(vectorizer_path):
    with open(vectorizer_path, 'rb') as file:
        vectorizer = pickle.load(file)
    return vectorizer

def preprocess_comments(data):
    # Remove special characters, numbers, and convert to lowercase
    data = re.sub(r'[^\w\s]', '', data)  # Remove special characters
    data = re.sub(r'\d+', '', data)      # Remove numbers
    data = data.lower()                  # Convert to lowercase
    return data

def transform_comments(data, vectorizer):
    # Transform the preprocessed comments into TF-IDF features using the TF-IDF vectorizer
    return vectorizer.transform([data])

def test_model_on_input(model, vectorizer):
    # Take input from the console
    input_comment = input("Enter the comment: ")
    
    # Preprocess the input comment
    preprocessed_comment = preprocess_comments(input_comment)
    
    # Transform the preprocessed comment into TF-IDF features using the TF-IDF vectorizer
    transformed_comment = transform_comments(preprocessed_comment, vectorizer)
    
    # Make prediction on the transformed comment using the loaded model
    prediction = model.predict(transformed_comment)
    
    # Print the prediction
    print("Predicted class:", prediction[0])

if __name__ == "__main__":
    # Load the trained model
    trained_model = load_model('trained_model.pkl')
    
    # Load the TF-IDF vectorizer used during training
    tfidf_vectorizer = load_vectorizer('tfidf_vectorizer.pkl')
    
    # Test the model on input from the console
    test_model_on_input(trained_model, tfidf_vectorizer)



################ Refinement of dataset is needed very badlyyy ###############################