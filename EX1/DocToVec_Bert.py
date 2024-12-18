from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from transformers import BertTokenizer, BertModel
import torch
import nltk
import pandas as pd
import os

# Predefined stopwords
PREDEFINED_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "while", "of", "to", "in", "on", "for", "with", "without", "by",
    "is", "it", "this", "that", "as", "at", "from", "was", "were", "be", "been", "are", "am", "can", "could", "should",
    "would", "may", "might", "do", "does", "did", "will", "shall", "you", "your", "yours", "we", "our", "ours", "he",
    "him", "his", "she", "her", "hers", "they", "them", "their", "theirs", "what", "which", "who", "whom", "whose",
    "why", "how", "where", "when", "not", "no", "yes", "all", "any", "some", "many", "few", "more", "most", "other",
    "another", "much", "such", "one", "two", "three", "about", "up", "down", "over", "under", "again", "further",
    "then", "once"
}


# Function to preprocess text
def preprocess_text(text):
    """
    Preprocess text: lowercase, tokenize, remove predefined stopwords.
    """
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in PREDEFINED_STOPWORDS]
    return tokens


# Load BERT tokenizer and model
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')


# Function to generate BERT embeddings
def get_bert_embedding(text):
    """
    Generate BERT embedding for a given text.
    """
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    outputs = bert_model(**inputs)
    # Use the [CLS] token embedding as the sentence representation
    return outputs.last_hidden_state[:, 0, :].detach().numpy().flatten()


# Load Excel file
file_path = r'posts_first_targil.xlsx'
sheets = pd.ExcelFile(file_path).sheet_names

# Output directory
output_dir = "DocToVec_Bert"
os.makedirs(output_dir, exist_ok=True)

# Prepare results
all_results = []

for sheet_name in sheets:
    print(f"Processing sheet: {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if 'Body Text' not in df.columns:
        print(f"'Body Text' column not found in {sheet_name}, skipping...")
        continue

    doc2vec_documents = []
    bert_vectors = []

    for idx, row in df.iterrows():
        text = str(row['Body Text']) if pd.notna(row['Body Text']) else ""

        # Preprocess for Doc2Vec
        tokens = preprocess_text(text)
        doc2vec_documents.append(TaggedDocument(words=tokens, tags=[f"{sheet_name}_{idx}"]))

        # BERT embeddings
        bert_vectors.append(get_bert_embedding(text))

    # Train Doc2Vec model
    doc2vec_model = Doc2Vec(vector_size=100, window=5, min_count=2, workers=4, epochs=20)
    doc2vec_model.build_vocab(doc2vec_documents)
    doc2vec_model.train(doc2vec_documents, total_examples=doc2vec_model.corpus_count, epochs=doc2vec_model.epochs)

    # Generate Doc2Vec vectors
    doc2vec_vectors = [doc2vec_model.dv[tag] for tag in doc2vec_model.dv.index_to_key]

    # Combine results into a DataFrame
    combined_df = pd.DataFrame({
        "Doc2Vec": [list(vec) for vec in doc2vec_vectors],
        "BERT": [list(vec) for vec in bert_vectors],
    })

    # Save results
    output_csv_path = os.path.join(output_dir, f"{sheet_name}_combined_embeddings.csv")
    combined_df.to_csv(output_csv_path, index=False)
    print(f"Saved embeddings for sheet '{sheet_name}' to: {output_csv_path}")

    # Append to global results
    all_results.append(combined_df)

# Save all results to a single CSV
all_results_df = pd.concat(all_results, ignore_index=True)
final_output_path = os.path.join(output_dir, "all_combined_embeddings.csv")
all_results_df.to_csv(final_output_path, index=False)
print(f"All embeddings saved to: {final_output_path}")
