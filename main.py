import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
import os


# הורדת חבילות NLTK רק אם צריך
def ensure_nltk_packages():
    packages = [
        'punkt', 'wordnet', 'omw-1.4', 'averaged_perceptron_tagger'
    ]
    for package in packages:
        try:
            nltk.data.find(f"corpora/{package}")
        except LookupError:
            nltk.download(package)


ensure_nltk_packages()


# פונקציות עיבוד
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


# קבצים
input_path = r'C:\Users\User\Desktop\Software Engineering\Year D\A\information retrieval\Ex1\posts_first_targil.xlsx'
output_dir = r'C:\Users\User\Desktop\Software Engineering\Year D\A\information retrieval\Ex1\output_sheets'

# יצירת תיקיית פלט אם לא קיימת
os.makedirs(output_dir, exist_ok=True)

# עיבוד
try:
    df = pd.read_excel(input_path, sheet_name=None)
    for sheet_name, sheet_data in df.items():
        if 'Body Text' in sheet_data.columns:  # עדכן בהתאם לעמודה שלך
            sheet_data['Body Text'] = sheet_data['Body Text'].fillna("").astype(str).apply(lemmatize_text)

        # שמירת כל גליון כקובץ נפרד
        output_file = os.path.join(output_dir, f"{sheet_name}.xlsx")
        sheet_data.to_excel(output_file, index=False)
        print(f"{sheet_name} נשמר כקובץ {output_file}")

    print("ניקוי וסיום לממציה בוצעו בהצלחה. כל הגליונות נשמרו כקבצים נפרדים.")

except Exception as e:
    print(f"תקלה התרחשה: {e}")
