# 📘 Natural Language Processing (NLP) – Complete Beginner to Intermediate Guide

> **A professional, GitHub‑ready README.md** covering NLP fundamentals in the exact learning sequence you should follow — with **clear explanations, diagrams-in-words, and full Python code examples**.

---

## 📌 Table of Contents

1. What is NLP?
2. NLP Learning Roadmap (Sequential Order)
3. NLP Libraries Overview

   * NLTK
   * spaCy
   * scikit‑learn (sklearn)
4. Text Representation Techniques

   * Bag of Words (BoW)
   * CountVectorizer
   * N‑grams
   * TF‑IDF
5. Word Embeddings

   * What are Word Embeddings?
   * Word2Vec
   * GloVe
6. Classical NLP vs Modern NLP
7. End‑to‑End NLP Pipeline Example

---

## 1️⃣ What is NLP?

**Natural Language Processing (NLP)** is a field of Artificial Intelligence that enables machines to understand, interpret, and generate **human language**.

### Common NLP Applications

* Spam detection
* Sentiment analysis
* Chatbots
* Resume screening
* Search engines
* Machine translation

---

## 2️⃣ NLP Learning Roadmap (VERY IMPORTANT)

Follow this exact order 👇

| Step | Concept                       |
| ---- | ----------------------------- |
| 1    | Text Cleaning & Preprocessing |
| 2    | NLP Libraries (NLTK, spaCy)   |
| 3    | Bag of Words                  |
| 4    | CountVectorizer               |
| 5    | N‑grams                       |
| 6    | TF‑IDF                        |
| 7    | Word Embeddings               |
| 8    | Word2Vec                      |
| 9    | GloVe                         |

---

## 3️⃣ NLP Libraries Overview

---

## 🔹 NLTK (Natural Language Toolkit)

### What is NLTK?

NLTK is a **classical NLP library** mainly used for:

* Teaching
* Research
* Rule‑based NLP

### Features

* Tokenization
* Stopword removal
* Stemming
* Lemmatization
* POS tagging

### Installation

```bash
pip install nltk
```

### Example – Text Preprocessing

```python
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

text = "NLP is very interesting and powerful"

# Tokenization
tokens = word_tokenize(text)

# Stopword Removal
stop_words = set(stopwords.words('english'))
filtered = [w for w in tokens if w.lower() not in stop_words]

# Stemming
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]

print(stemmed)
```

---

## 🔹 spaCy

### What is spaCy?

spaCy is an **industrial‑grade NLP library** used in production systems.

### Why spaCy over NLTK?

| NLTK        | spaCy            |
| ----------- | ---------------- |
| Slow        | Very Fast        |
| Educational | Production Ready |
| Rule‑based  | ML‑based         |

### Installation

```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Example – Tokenization & Lemmatization

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple is looking at buying a startup in India")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

---

## 🔹 scikit‑learn (sklearn)

### Role of sklearn in NLP

sklearn is used for:

* Feature extraction
* Vectorization
* ML models

Key NLP tools:

* CountVectorizer
* TfidfVectorizer

---

## 4️⃣ Text Representation Techniques

Machines **cannot understand text**.
They only understand **numbers**.

So we convert text ➜ numbers.

---

## 🔹 Bag of Words (BoW)

### Concept

BoW represents text as **word frequency vectors**, ignoring grammar & order.

### Example

Sentence:

```
I love NLP
I love Machine Learning
```

Vocabulary:

```
[I, love, NLP, Machine, Learning]
```

Vectors:

```
[1,1,1,0,0]
[1,1,0,1,1]
```

### Limitations

* No word order
* No meaning
* Sparse vectors

---

## 🔹 CountVectorizer

### What is CountVectorizer?

A sklearn implementation of **Bag of Words**.

### Example

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    "I love NLP",
    "I love Machine Learning"
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

## 🔹 N‑grams

### What are N‑grams?

N‑grams capture **word sequences**.

| Type    | Example                     |
| ------- | --------------------------- |
| Unigram | NLP                         |
| Bigram  | Machine Learning            |
| Trigram | Natural Language Processing |

### Example with Bigram

```python
vectorizer = CountVectorizer(ngram_range=(2,2))
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names_out())
```

---

## 🔹 TF‑IDF (Term Frequency – Inverse Document Frequency)

### Why TF‑IDF?

BoW gives **equal importance** to all words.
TF‑IDF reduces weight of common words.

### Formula

```
TF‑IDF = TF × IDF
```

### Example

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

print(vectorizer.get_feature_names_out())
print(X.toarray())
```

---

## 5️⃣ Word Embeddings

---

## 🔹 What are Word Embeddings?

Word embeddings represent words in **dense vector space** where:

* Similar words are close
* Semantic meaning is preserved

Example:

```
king − man + woman ≈ queen
```

---

## 🔹 Word2Vec

### What is Word2Vec?

Word2Vec is a **neural network‑based** embedding model.

### Architectures

* CBOW (predict word from context)
* Skip‑Gram (predict context from word)

### Example using gensim

```python
from gensim.models import Word2Vec

sentences = [
    ['i', 'love', 'nlp'],
    ['i', 'love', 'machine', 'learning']
]

model = Word2Vec(sentences, vector_size=50, window=5, min_count=1)

print(model.wv['nlp'])
```

---

## 🔹 GloVe (Global Vectors)

### What is GloVe?

GloVe is a **count‑based + context‑based** embedding model.

### Key Difference

| Word2Vec      | GloVe             |
| ------------- | ----------------- |
| Local context | Global statistics |
| Predictive    | Count‑based       |

### Using Pretrained GloVe

```python
import numpy as np

def load_glove(path):
    embeddings = {}
    with open(path, 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    return embeddings
```

---

## 6️⃣ Classical NLP vs Modern NLP

| Classical NLP  | Modern NLP      |
| -------------- | --------------- |
| BoW            | Word Embeddings |
| TF‑IDF         | Transformers    |
| Sparse vectors | Dense vectors   |
| No context     | Context aware   |

---

## 7️⃣ End‑to‑End NLP Pipeline Example

```python
# 1. Text Cleaning
# 2. Tokenization
# 3. Vectorization (TF‑IDF)
# 4. Model Training

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

texts = ["I love NLP", "I hate spam"]
labels = [1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

model = LogisticRegression()
model.fit(X, labels)
```

---

## 🎯 Final Notes

✔ This README is **download‑ready**
✔ Covers **all asked concepts without skipping**
✔ Follows **industry‑correct learning sequence**
✔ Suitable for **GitHub, interviews, and projects**

---

### ⭐ If you want next:

* NLP interview questions
* NLP projects
* Transformer & BERT deep dive
* LangChain NLP pipelines

Just tell me 👍
