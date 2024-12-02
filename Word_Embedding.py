import os
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.phrases import Phrases, Phraser
from nltk.corpus import stopwords
from sklearn.preprocessing import StandardScaler


def preprocess_text(text, remove_stopwords=True):
    """
    Preprocess text: lowercase, optional stopwords removal
    """
    tokens = text.lower().split()

    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]

    return tokens


def train_word2vec(documents, vector_size=100, window=5, min_count=1):
    """
    Train Word2Vec model on preprocessed documents
    """
    if not documents or not any(documents):
        raise ValueError("No valid documents provided for training Word2Vec.")

    # Detect phrases
    phrases = Phrases(documents, min_count=5)
    bigram = Phraser(phrases)

    # Apply bigram
    documents = [bigram[doc] for doc in documents]

    # Train Word2Vec
    model = Word2Vec(sentences=documents,
                     vector_size=vector_size,
                     window=window,
                     min_count=min_count,
                     workers=4)

    return model, documents


def create_document_vectors(model, documents):
    """
    Create document vectors by averaging word vectors
    """
    document_vectors = []
    for doc in documents:
        doc_vectors = [model.wv[word] for word in doc if word in model.wv]
        if doc_vectors:
            document_vectors.append(np.mean(doc_vectors, axis=0))
        else:
            document_vectors.append(np.zeros(model.vector_size))

    return np.array(document_vectors)


def main():
    input_dirs = {
        "clean_no_stopwords_A-J": r'TF_IDF_output/processed/vectors_clean_no_stopwords_A-J.csv',
        "clean_no_stopwords_BBC": r'TF_IDF_output/processed/vectors_clean_no_stopwords_BBC.csv',
        "clean_no_stopwords_J-P": r'TF_IDF_output/processed/vectors_clean_no_stopwords_J-P.csv',
        "clean_no_stopwords_NY-T": r'TF_IDF_output/processed/vectors_clean_no_stopwords_NY-T.csv',
        "lemmatized_no_stopwords_A-J": r'TF_IDF_output/processed/vectors_lemmatized_no_stopwords_A-J.csv',
        "lemmatized_no_stopwords_BBC": r'TF_IDF_output/processed/vectors_lemmatized_no_stopwords_BBC.csv',
        "lemmatized_no_stopwords_J-P": r'TF_IDF_output/processed/vectors_lemmatized_no_stopwords_J-P.csv',
        "lemmatized_no_stopwords_NY-T": r'TF_IDF_output/processed/vectors_lemmatized_no_stopwords_NY-T.csv',
    }

    output_dir = r'word_embeddings'
    os.makedirs(output_dir, exist_ok=True)

    for source, file_path in input_dirs.items():
        documents = []

        try:
            # Load documents from the CSV file
            df = pd.read_csv(file_path, header=None, low_memory=False)
            print(f"Loaded CSV file: {file_path}")

            docs = df.iloc[:, 1].fillna('').astype(str).tolist()

            # Preprocess documents
            preprocessed_docs = [preprocess_text(doc, remove_stopwords=True) for doc in docs]
            preprocessed_docs = [doc for doc in preprocessed_docs if doc]  # Remove empty docs

            if not preprocessed_docs:  # Skip if no valid tokens
                print(f"No valid tokens found for {file_path}, skipping.")
                continue

            documents.extend(preprocessed_docs)

            print(f"Sample tokens from preprocessed documents: {documents[:5]}")

            # Train Word2Vec
            w2v_model, processed_docs = train_word2vec(documents)

            print(f"Word2Vec vocabulary size: {len(w2v_model.wv.index_to_key)}")

            # Create document vectors
            doc_vectors = create_document_vectors(w2v_model, processed_docs)

            # Scale document vectors
            scaler = StandardScaler()
            doc_vectors_scaled = scaler.fit_transform(doc_vectors)

            # Save document vectors to CSV
            output_file_csv = os.path.join(output_dir, f'{source}_word2vec_vectors.csv')
            pd.DataFrame(doc_vectors_scaled).to_csv(output_file_csv, index=False)
            print(f"Document vectors saved to: {output_file_csv}")

            # Save model
            model_file = os.path.join(output_dir, f'{source}_word2vec_model')
            w2v_model.save(model_file)
            print(f"Word2Vec model saved to: {model_file}")

        except Exception as e:
            print(f"Unexpected error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
