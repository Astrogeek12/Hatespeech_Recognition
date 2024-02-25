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
    data = data.apply(lambda x: re.sub(r'[^\w\s]', '', x))  # Remove special characters
    data = data.apply(lambda x: re.sub(r'\d+', '', x))      # Remove numbers
    data = data.apply(lambda x: x.lower())                  # Convert to lowercase
    return data

def transform_comments(data, vectorizer):
    # Transform the preprocessed comments into TF-IDF features using the TF-IDF vectorizer
    return vectorizer.transform(data)

def test_model_on_unseen_data(model, vectorizer, unseen_data_path,output_path):
    # Load the unseen data
    unseen_data = pd.read_csv(unseen_data_path)
    
    # Preprocess the comments in the unseen data
    unseen_comments = preprocess_comments(unseen_data['comment'])
    
    # Transform the preprocessed comments into TF-IDF features using the TF-IDF vectorizer
    unseen_comments_tfidf = transform_comments(unseen_comments, vectorizer)
    
    # Make predictions on the preprocessed and transformed unseen data using the loaded model
    predictions = model.predict(unseen_comments_tfidf)

    results_df = pd.DataFrame({'Preprocessed Comment': unseen_comments, 'Prediction': predictions})
    
    # Save the results to a new CSV file
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
    
    
    


if __name__ == "__main__":
    # Load the trained model
    trained_model = load_model('trained_model.pkl')
    
    # Load the TF-IDF vectorizer used during training
    tfidf_vectorizer = load_vectorizer('tfidf_vectorizer.pkl')
    
    # Test the model on unseen data
    test_model_on_unseen_data(trained_model, tfidf_vectorizer, 'testing_data.csv', 'predictions.csv')




