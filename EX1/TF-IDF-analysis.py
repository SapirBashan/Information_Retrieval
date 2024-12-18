import os
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, chi2
from sklearn.preprocessing import LabelEncoder
import scipy.sparse as sp


def load_tfidf_matrices(input_dirs):
    """
    Load TF-IDF matrices and metadata.

    Args:
        input_dirs (dict): Dictionary containing matrix, metadata, and feature file paths.

    Returns:
        dict: Dictionary containing matrices, labels, and feature names.
    """
    matrices = {}
    for source, paths in input_dirs.items():
        try:
            matrix_path = paths['matrix']
            metadata_path = paths['metadata']
            feature_path = paths['feature']

            # Debugging: Print paths being checked
            print(f"Checking paths for source {source}:")
            print(f"  Matrix path: {matrix_path}")
            print(f"  Metadata path: {metadata_path}")
            print(f"  Feature path: {feature_path}")

            # Check if the matrix file exists
            if not os.path.exists(matrix_path):
                print(f"Missing matrix file for {source}: {matrix_path}")
                continue

            # Check if metadata and feature files exist
            if not os.path.exists(metadata_path):
                print(f"Missing metadata file for {source}: {metadata_path}")
                continue
            if not os.path.exists(feature_path):
                print(f"Missing feature file for {source}: {feature_path}")
                continue

            # Load sparse matrix and metadata
            tfidf_matrix = sp.load_npz(matrix_path)
            metadata = pd.read_csv(metadata_path)
            feature_names = pd.read_csv(feature_path)['Feature'].tolist()

            # Extract and encode group labels
            label_encoder = LabelEncoder()
            group_labels = label_encoder.fit_transform(metadata['Source File'].str.extract(r'(A-J|BBC|J-P|NY-T)')[0].fillna(''))

            # Store matrix data
            matrices[source] = {
                'matrix': tfidf_matrix,
                'labels': group_labels,
                'feature_names': feature_names
            }
        except Exception as e:
            print(f"Error loading data for {source}: {e}")

    return matrices



def calculate_information_gain(matrices):
    """
    Calculate Information Gain for each feature.

    Args:
        matrices (dict): Dictionary of TF-IDF matrices.

    Returns:
        dict: Feature importance scores using Information Gain.
    """
    information_gain_results = {}
    for source, data in matrices.items():
        try:
            # Convert sparse matrix to dense
            dense_matrix = data['matrix'].toarray()
            ig_scores = mutual_info_classif(dense_matrix, data['labels'])

            # Create DataFrame with results
            information_gain_results[source] = pd.DataFrame({
                'Feature': data['feature_names'],
                'Information Gain': ig_scores
            }).sort_values('Information Gain', ascending=False)
        except Exception as e:
            print(f"Error calculating Information Gain for {source}: {e}")

    return information_gain_results


def calculate_chi_squared(matrices):
    """
    Calculate Chi-squared scores for each feature.

    Args:
        matrices (dict): Dictionary of TF-IDF matrices.

    Returns:
        dict: Feature importance scores using Chi-squared.
    """
    chi_squared_results = {}
    for source, data in matrices.items():
        try:
            # Convert sparse matrix to dense
            dense_matrix = data['matrix'].toarray()
            chi2_scores, _ = chi2(dense_matrix, data['labels'])

            # Create DataFrame with results
            chi_squared_results[source] = pd.DataFrame({
                'Feature': data['feature_names'],
                'Chi-squared Score': chi2_scores
            }).sort_values('Chi-squared Score', ascending=False)
        except Exception as e:
            print(f"Error calculating Chi-squared for {source}: {e}")

    return chi_squared_results


def save_feature_importance_results(information_gain, chi_squared, output_dir):
    """
    Save feature importance results to Excel files.

    Args:
        information_gain (dict): Information Gain results.
        chi_squared (dict): Chi-squared results.
        output_dir (str): Directory for saving results.
    """
    os.makedirs(output_dir, exist_ok=True)
    for source in information_gain.keys():
        try:
            output_path = os.path.join(output_dir, f'feature_importance_{source}.xlsx')
            with pd.ExcelWriter(output_path) as writer:
                information_gain[source].to_excel(writer, sheet_name='Information Gain', index=False)
                chi_squared[source].to_excel(writer, sheet_name='Chi-squared', index=False)
            print(f"Saved feature importance results for {source} to {output_path}")
        except Exception as e:
            print(f"Error saving results for {source}: {e}")


def main():
    # Input directories
    input_dirs = {
        'clean_no_stopwords_A-J': {
            'matrix': r'TF_IDF_output/matrix_shape/tfidf_matrix_lemmatized_no_stopwords_A-J.npz',
            'metadata': r'TF_IDF_output/processed/document_metadata_lemmatized_no_stopwords_A-J.csv',
            'feature': r'TF_IDF_output/number_of_ft/feature_importances_lemmatized_no_stopwords_A-J.csv',
        },
        'clean_no_stopwords_BBC': {
            'matrix': r'TF_IDF_output/matrix_shape/tfidf_matrix_lemmatized_no_stopwords_BBC.npz',
            'metadata': r'TF_IDF_output/processed/document_metadata_lemmatized_no_stopwords_BBC.csv',
            'feature': r'TF_IDF_output/number_of_ft/feature_importances_lemmatized_no_stopwords_BBC.csv',
        },
        'clean_no_stopwords_J-P': {
            'matrix': r'TF_IDF_output/matrix_shape/tfidf_matrix_lemmatized_no_stopwords_J-P.npz',
            'metadata': r'TF_IDF_output/processed/document_metadata_lemmatized_no_stopwords_J-P.csv',
            'feature': r'TF_IDF_output/number_of_ft/feature_importances_lemmatized_no_stopwords_J-P.csv',
        },
        'clean_no_stopwords_NY-T': {
            'matrix': r'TF_IDF_output/matrix_shape/tfidf_matrix_lemmatized_no_stopwords_NY-T.npz',
            'metadata': r'TF_IDF_output/processed/document_metadata_lemmatized_no_stopwords_NY-T.csv',
            'feature': r'TF_IDF_output/number_of_ft/feature_importances_lemmatized_no_stopwords_NY-T.csv',
        },
    }

    # Output directory
    output_dir = r'feature_importance_results_lemma'

    # Load TF-IDF matrices
    matrices = load_tfidf_matrices(input_dirs)

    # Calculate feature importance
    information_gain = calculate_information_gain(matrices)
    chi_squared = calculate_chi_squared(matrices)

    # Save results
    save_feature_importance_results(information_gain, chi_squared, output_dir)


if __name__ == "__main__":
    main()
