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
        tokens = re.findall(r'\w+|[^\w\s]', text, re.UNICODE)
        return ' '.join(tokens)
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
input_path = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\posts_first_targil.xlsx'
output_dir = r'C:\Users\sapir\OneDrive\שולחן העבודה\סמסטר א שנה ד\איחזור מידע\IR-TF_IDF_Ex1\output_sheets'

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

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

        # Save the clean text data (without stopwords and including ID)
        clean_no_stopwords_output_file = os.path.join(output_dir, f"clean_no_stopwords_{sheet_name}.xlsx")
        sheet_data[['ID', 'Cleaned Text without Stopwords']].to_excel(clean_no_stopwords_output_file, index=False,
                                                                      header=False)
        print(f"Cleaned text without stopwords for {sheet_name} saved as {clean_no_stopwords_output_file}")

        # Save the lemmatized text data (without stopwords and including ID)
        lemmatized_no_stopwords_output_file = os.path.join(output_dir, f"lemmatized_no_stopwords_{sheet_name}.xlsx")
        sheet_data[['ID', 'Lemmatized Text without Stopwords']].to_excel(lemmatized_no_stopwords_output_file,
                                                                         index=False, header=False)
        print(f"Lemmatized text without stopwords for {sheet_name} saved as {lemmatized_no_stopwords_output_file}")

    print("Cleaning, lemmatization, and stopwords removal completed successfully. All sheets saved as separate files.")

except Exception as e:
    print(f"An error occurred: {e}")
