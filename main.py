import pandas as pd
import re
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os

# Ensure required NLTK packages are downloaded
def ensure_nltk_packages():
    packages = [
        'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger', 'stopwords'
    ]
    for package in packages:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package)


ensure_nltk_packages()

# Text cleaning functions
def clean_text(text):
    if isinstance(text, str):
        # Remove apostrophes without splitting words, add spaces around other punctuation
        text = re.sub(r"'", "", text)  # Remove apostrophes
        text = re.sub(r"([^\w\s])", r" \1 ", text)  # Add space around punctuation
        return re.sub(r'\s+', ' ', text).strip()  # Normalize spaces
    return text


def get_wordnet_pos(word):
    from nltk.corpus.reader.wordnet import ADJ, VERB, NOUN, ADV
    tag_map = {'J': ADJ, 'V': VERB, 'N': NOUN, 'R': ADV}
    tag = nltk.pos_tag([word])[0][1][0].upper()
    return tag_map.get(tag, NOUN)


def lemmatize_text(text):
    if not isinstance(text, str):
        return ""
    lemmatizer = WordNetLemmatizer()
    # Remove all punctuation including apostrophes
    text = re.sub(r"[^\w\s]", "", text)
    tokens = word_tokenize(text)
    lemmatized = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]
    return " ".join(lemmatized)


# Remove stopwords
def remove_stopwords(text):
    if not isinstance(text, str):
        return ""
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    return " ".join(filtered_tokens)


# File paths
input_path = r'posts_first_targil.xlsx'
output_dir = r'output_sheets'

# Create output subdirectories
clean_dir = os.path.join(output_dir, 'clean')
lemmatized_dir = os.path.join(output_dir, 'lemmatized')
stop_word_clean_dir = os.path.join(output_dir, 'stop_word_clean')
stop_word_lemmatized_dir = os.path.join(output_dir, 'stop_word_lemmatized')

for directory in [clean_dir, lemmatized_dir, stop_word_clean_dir, stop_word_lemmatized_dir]:
    os.makedirs(directory, exist_ok=True)

# Processing data
try:
    df = pd.read_excel(input_path, sheet_name=None)
    for sheet_name, sheet_data in df.items():
        if 'Body Text' in sheet_data.columns:  # Update according to your column name
            # Clean and lemmatize the text
            sheet_data['Cleaned Text'] = sheet_data['Body Text'].fillna("").astype(str).apply(clean_text)
            sheet_data['Lemmatized Text'] = sheet_data['Body Text'].fillna("").astype(str).apply(lemmatize_text)

            # Add an ID column
            sheet_data['ID'] = sheet_data.index + 1  # Add 1 to start IDs from 1

            # Remove stop words from Cleaned Text and Lemmatized Text
            sheet_data['Cleaned Text without Stopwords'] = sheet_data['Cleaned Text'].apply(remove_stopwords)
            sheet_data['Lemmatized Text without Stopwords'] = sheet_data['Lemmatized Text'].apply(remove_stopwords)

            # Save files into corresponding directories
            sheet_data[['ID', 'Cleaned Text']].to_excel(
                os.path.join(clean_dir, f"{sheet_name}_clean.xlsx"), index=False, header=False
            )
            sheet_data[['ID', 'Lemmatized Text']].to_excel(
                os.path.join(lemmatized_dir, f"{sheet_name}_lemmatized.xlsx"), index=False, header=False
            )
            sheet_data[['ID', 'Cleaned Text without Stopwords']].to_excel(
                os.path.join(stop_word_clean_dir, f"{sheet_name}_stopword_clean.xlsx"), index=False, header=False
            )
            sheet_data[['ID', 'Lemmatized Text without Stopwords']].to_excel(
                os.path.join(stop_word_lemmatized_dir, f"{sheet_name}_stopword_lemmatized.xlsx"), index=False, header=False
            )

    print("Processing completed successfully. Files saved in structured directories.")

except Exception as e:
    print(f"An error occurred: {e}")