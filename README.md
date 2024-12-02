# Text Embedding and Feature Analysis Pipeline

## Project Overview

This project implements a comprehensive text processing and embedding pipeline designed for advanced information retrieval and text analysis. By leveraging multiple state-of-the-art natural language processing techniques, the project extracts meaningful representations from textual data across various preprocessing strategies.

## 🚀 Key Components

### 1. Text Preprocessing (`main.py`)
- **Text Cleaning**: 
  - Remove apostrophes and normalize punctuation
  - Standardize text formatting
- **Text Lemmatization**: 
  - Convert words to their base or dictionary form
  - Preserve semantic meaning while reducing vocabulary complexity
- **Stopword Removal**: 
  - Eliminate common, low-information words
  - Improve signal-to-noise ratio in text representations

### 2. Embedding Techniques

#### A. TF-IDF Embedding (`TF-IDF.py`)
- Compute Term Frequency-Inverse Document Frequency vectors
- Highlight important words based on their uniqueness and frequency
- Generate feature importance rankings
- Support multiple document collections

#### B. Word2Vec Embedding (`Word_Embedding.py`)
- Train word vectors using context-based learning
- Detect and incorporate multi-word phrases
- Create document representations by averaging word vectors
- Apply standard scaling for normalized representations

#### C. Doc2Vec & BERT Embedding (`DocToVec_Bert.py`)
- Utilize Doc2Vec for document-level vector representations
- Leverage BERT's contextual embeddings
- Capture deep semantic nuances in text

#### D. Sentence-BERT Embedding (`SBERT_Embedding.py`)
- Generate sentence-level embeddings
- Utilize pre-trained transformer models
- Produce context-aware semantic representations

### 3. Feature Analysis (`TF-IDF-analysis.py`)
- Calculate Information Gain
- Compute Chi-squared feature importance
- Provide detailed insights into feature significance

## 🛠 Technologies & Libraries

- **Python**: Primary programming language
- **Preprocessing**: NLTK
- **Machine Learning**: scikit-learn
- **Embedding**: 
  - Gensim (Word2Vec, Doc2Vec)
  - Transformers (BERT)
  - Sentence-Transformers
- **Data Handling**: Pandas, NumPy
- **Feature Analysis**: Mutual Information, Chi-squared Test

## 📂 Project Structure

```
project_root/
│
├── main.py                   # Text preprocessing
├── TF-IDF.py                 # TF-IDF embedding
├── Word_Embedding.py         # Word2Vec embedding
├── DocToVec_Bert.py          # Doc2Vec & BERT embedding
├── TF-IDF-analysis.py        # Feature importance analysis
└── SBERT_Embedding.py        # Sentence-BERT embedding
```

## 🔍 Workflow

1. **Text Preprocessing**
   - Clean raw text data
   - Remove stopwords
   - Lemmatize tokens

2. **Embedding Generation**
   - Apply multiple embedding techniques
   - Create vector representations
   - Save embeddings for further analysis

3. **Feature Analysis**
   - Compute feature importance
   - Identify most significant terms
   - Generate detailed reports

## 🌟 Key Features

- Multiple preprocessing strategies
- Diverse embedding techniques
- Comprehensive feature analysis
- Scalable and modular design
- Supports various text document sources

## 📈 Performance Considerations

- Handles multiple document collections
- Supports lemmatized and non-lemmatized approaches
- Configurable embedding parameters
- Produces standardized and scaled vector representations

## 🔗 Dependencies

- Python 3.8+
- See `requirements.txt` for detailed library versions

## 🚧 Future Enhancements

- Integrate advanced transformer models
- Implement cross-lingual embeddings
- Develop interactive visualization tools
- Expand feature selection techniques

## 👥 Contributors

Noam Benisho 213200496

Sapir Bashan 214103368

## 🙏 Acknowledgments

- Course Lecturer
- NLP Research Community
```

