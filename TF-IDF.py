import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import scipy.sparse as sp


def load_text_files(directory):
    """
    Load text files from a given directory, preserving the order and file names.

    Args:
        directory (str): Path to directory containing text files

    Returns:
        tuple: (documents, file_names)
    """
    documents = []
    file_names = []

    for filename in sorted(os.listdir(directory)):
        if filename.endswith('.xlsx'):
            filepath = os.path.join(directory, filename)
            df = pd.read_excel(filepath, header=None)

            # Check if the DataFrame has at least two columns
            if df.shape[1] >= 2:
                # Assuming the second column contains the text
                documents.extend(df.iloc[:, 1].fillna('').astype(str).tolist())
                file_names.extend([filename] * len(df))

    return documents, file_names


def create_tfidf_matrices(input_dirs, output_dir, output_dir_top_20 , output_dir_matrix_shape , output_dir_number_of_ft, output_dir_processed):
    """
    Create TF-IDF matrices for different text types and sources.

    Args:
        input_dirs (dict): Dictionary of input directories
        output_dir (str): Directory to save output files
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for source_type, source_dir in input_dirs.items():
        # Load documents and file names
        documents, file_names = load_text_files(source_dir)

        if not documents:
            print(f"No documents found in {source_dir}")
            continue

        # Create TF-IDF Vectorizer
        vectorizer = TfidfVectorizer(
            stop_words=None,  # We've already removed stop words
            token_pattern=r'\b\w+\b'  # Ensure we're using whole words
        )

        # Fit and transform documents
        tfidf_matrix = vectorizer.fit_transform(documents)

        # Get feature names (words)
        feature_names = vectorizer.get_feature_names_out()

        # Create a DataFrame of feature importances
        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'TF-IDF Importance': tfidf_matrix.sum(axis=0).A1
        }).sort_values('TF-IDF Importance', ascending=False)

        # Save matrices and feature importance
        # TF-IDF Matrix
        sp.save_npz(os.path.join(output_dir_matrix_shape, f'tfidf_matrix_{source_type}.npz'), tfidf_matrix)

        # Feature Importance
        feature_importances.to_csv(
            os.path.join(output_dir_number_of_ft, f'feature_importances_{source_type}.csv'),
            index=False
        )

        # Top 20 features
        top_20_features = feature_importances.head(20)
        top_20_features.to_csv(
            os.path.join(output_dir_top_20 , f'top_20_features_{source_type}.csv'),
            index=False
        )

        # Metadata file to track document sources
        metadata_df = pd.DataFrame({
            'Source File': file_names,
            'TF-IDF Vector Index': range(len(file_names))
        })
        metadata_df.to_csv(
            os.path.join(output_dir_processed, f'document_metadata_{source_type}.csv'),
            index=False
        )

        print(f"Processed {source_type}:")
        print(f"- Matrix shape: {tfidf_matrix.shape}")
        print(f"- Number of features: {len(feature_names)}")


def main():
    # Base input directory
    base_input_dir = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\output_sheets'

    # Output directory
    output_dir_top_20 = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\TF_IDF_output\top_20'
    output_dir_processed = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\TF_IDF_output\processed'
    output_dir_matrix_shape = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\TF_IDF_output\matrix_shape'
    output_dir_number_of_ft = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\TF_IDF_output\number_of_ft'
    output_dir = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\TF_IDF_output'

    os.makedirs(output_dir_processed, exist_ok=True)
    os.makedirs(output_dir_matrix_shape, exist_ok=True)
    os.makedirs(output_dir_top_20, exist_ok=True)
    os.makedirs(output_dir_number_of_ft, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    # Input directories
    input_dirs = {
        'clean_A-J': os.path.join(base_input_dir, 'clean'),
        'clean_BBC': os.path.join(base_input_dir, 'clean'),
        'clean_J-P': os.path.join(base_input_dir, 'clean'),
        'clean_NY-T': os.path.join(base_input_dir, 'clean'),
        'lemmatized_A-J': os.path.join(base_input_dir, 'lemmatized'),
        'lemmatized_BBC': os.path.join(base_input_dir, 'lemmatized'),
        'lemmatized_J-P': os.path.join(base_input_dir, 'lemmatized'),
        'lemmatized_NY-T': os.path.join(base_input_dir, 'lemmatized'),
        'clean_no_stopwords_A-J.xlsx': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_BBC.xlsx': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_J-P.xlsx': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_NY-T.xlsx': os.path.join(base_input_dir, 'stop_word_clean'),
        'lemmatized_no_stopwords_A-J.xlsx': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_BBC.xlsx': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_J-P.xlsx': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_NY-T.xlsx': os.path.join(base_input_dir, 'stop_word_lemmatized'),
    }

    # Create TF-IDF matrices
    create_tfidf_matrices(input_dirs, output_dir, output_dir_top_20 , output_dir_matrix_shape , output_dir_number_of_ft, output_dir_processed)


if __name__ == "__main__":
    main()