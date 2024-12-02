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

def create_tfidf_matrices(input_dirs, output_dir, output_dir_top_20, output_dir_matrix_shape, output_dir_number_of_ft, output_dir_processed):
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

        # Save the TF-IDF matrix as a dense CSV (vectors)
        tfidf_dense = tfidf_matrix.toarray()  # Convert sparse matrix to dense
        tfidf_vectors_df = pd.DataFrame(tfidf_dense, columns=feature_names)
        tfidf_vectors_df['Source File'] = file_names  # Add metadata (file names)

        # Save vectors to a CSV file
        vectors_output_file = os.path.join(output_dir_processed, f'vectors_{source_type}.csv')
        tfidf_vectors_df.to_csv(vectors_output_file, index=False)
        print(f"Vectors for {source_type} saved to {vectors_output_file}")

        # Save the sparse matrix for further processing
        sp.save_npz(os.path.join(output_dir_matrix_shape, f'tfidf_matrix_{source_type}.npz'), tfidf_matrix)

        # Save Feature Importance
        feature_importances = pd.DataFrame({
            'Feature': feature_names,
            'TF-IDF Importance': tfidf_matrix.sum(axis=0).A1
        }).sort_values('TF-IDF Importance', ascending=False)

        feature_importances.to_csv(
            os.path.join(output_dir_number_of_ft, f'feature_importances_{source_type}.csv'),
            index=False
        )

        # Save Top 20 features
        top_20_features = feature_importances.head(20)
        top_20_features.to_csv(
            os.path.join(output_dir_top_20, f'top_20_features_{source_type}.csv'),
            index=False
        )

        # Save metadata file
        metadata_df = pd.DataFrame({
            'Source File': file_names,
            'TF-IDF Vector Index': range(len(file_names))
        })
        metadata_output_file = os.path.join(output_dir_processed, f'document_metadata_{source_type}.csv')
        metadata_df.to_csv(metadata_output_file, index=False)

        print(f"Processed {source_type}:")
        print(f"- Matrix shape: {tfidf_matrix.shape}")
        print(f"- Number of features: {len(feature_names)}")


def main():
    # Base input directory
    base_input_dir = r'output_sheets'

    # Output directory
    output_dir_top_20 = r'top_20'
    output_dir_processed = r'TF_IDF_output\processed'
    output_dir_matrix_shape = r'TF_IDF_output\matrix_shape'
    output_dir_number_of_ft = r'TF_IDF_output\number_of_ft'
    output_dir = r'TF_IDF_output'

    os.makedirs(output_dir_processed, exist_ok=True)
    os.makedirs(output_dir_matrix_shape, exist_ok=True)
    os.makedirs(output_dir_top_20, exist_ok=True)
    os.makedirs(output_dir_number_of_ft, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)


    # Input directories
    input_dirs = {
        'clean_no_stopwords_A-J': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_BBC': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_J-P': os.path.join(base_input_dir, 'stop_word_clean'),
        'clean_no_stopwords_NY-T': os.path.join(base_input_dir, 'stop_word_clean'),
        'lemmatized_no_stopwords_A-J': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_BBC': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_J-P': os.path.join(base_input_dir, 'stop_word_lemmatized'),
        'lemmatized_no_stopwords_NY-T': os.path.join(base_input_dir, 'stop_word_lemmatized'),
    }

    # Create TF-IDF matrices
    create_tfidf_matrices(input_dirs, output_dir, output_dir_top_20 , output_dir_matrix_shape , output_dir_number_of_ft, output_dir_processed)


if __name__ == "__main__":
    main()