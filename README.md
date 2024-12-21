
# README: Exercise 2 - Clustering and Classification Analysis

## Authors  
**Sapir Bashan & Noam Benisho** 
---

## 1. Problem Statement  
In this exercise, we analyze textual data from **four different newspapers** using unsupervised clustering algorithms and supervised classification methods.  

We aim to:  
1. **Cluster** the data using methods such as **K-Means**, **DBSCAN**, and **Gaussian Mixture Models (GMM)**.  
2. Evaluate clustering performance using **Precision, Recall, F1-Score**, and **Accuracy**, along with visualizations using UMAP/t-SNE.  
3. Use supervised classifiers (ANN, NB, SVM, LoR, RF) to predict labels, analyze results, and identify key features.  

---

## 2. Methods and Algorithm Selection  

### 2.1 Data Preprocessing  
- Combined the **four TF-IDF matrices** into a single representation.  
- Ensured cosine similarity as the distance metric for clustering.  

---

### 2.2 Clustering Algorithms  

#### **1. K-Means Clustering**  
- *Why K-Means?* It partitions data into `k=4` clusters efficiently.  
- **Code Implementation:**  
```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(combined_matrix)

# Evaluate using Silhouette Score
sil_score = silhouette_score(combined_matrix, kmeans_labels, metric='cosine')
print(f"Silhouette Score (K-Means): {sil_score}")
```

---

#### **2. DBSCAN Clustering**  
- *Why DBSCAN?* Handles noise and finds arbitrarily shaped clusters.  
- **Parameter Selection:**  
    - **Epsilon (eps):** Calculated using the Maximum Spanning Tree (MST).  
    - **min_samples:** Tuned empirically.  
- **Code Implementation:**  
```python
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree

# Calculate distance matrix and MST
distance_matrix = squareform(pdist(combined_matrix, metric='cosine'))
mst = minimum_spanning_tree(distance_matrix).toarray()
epsilon = mst.max()

# Apply DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=25, metric='cosine')
dbscan_labels = dbscan.fit_predict(combined_matrix)
print(f"DBSCAN Labels: {Counter(dbscan_labels)}")
```

---

#### **3. Gaussian Mixture Models (GMM)**  
- *Why GMM?* Provides a probabilistic clustering approach.  
- **Code Implementation:**  
```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(combined_matrix)
print(f"GMM Labels: {Counter(gmm_labels)}")
```

---

### 2.3 Supervised Classification  

#### **1. Artificial Neural Network (ANN)**  

##### **ReLU Activation ANN**  
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Build the ANN model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add callbacks
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
    ModelCheckpoint("best_ann_relu.keras", save_best_only=True)
]

# Train model
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32, callbacks=callbacks)
```

##### **GELU Activation ANN**  
```python
from tensorflow.keras.activations import gelu

# Build ANN with GELU activation
model_gelu = Sequential([
    Dense(64, activation=gelu),
    Dense(32, activation=gelu),
    Dense(16, activation=gelu),
    Dense(num_classes, activation='softmax')
])

model_gelu.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model_gelu.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32, callbacks=callbacks)
```

---

#### **2. Naive Bayes (NB)**  
```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

#### **3. Support Vector Machine (SVM)**  
```python
from sklearn.svm import SVC

svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

#### **4. Logistic Regression (LoR)**  
```python
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

#### **5. Random Forest (RF)**  
```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

---

## 3. Results  

### Clustering Results  
- **K-Means:** Silhouette Score: [Insert value]  
- **DBSCAN:** Epsilon and min_samples selected via MST heuristic.  
- **GMM:** [Insert insights].  

#### Visualizations  
- **t-SNE Visualizations:** Cluster quality is demonstrated with distinct groupings.  

---

### Classification Results - you can see in the jupiter files

---

## 4. Conclusions  

### Key Findings  
- K-Means performed efficiently but struggled with noise.  
- ANN with **GELU** activation achieved the best performance among classifiers.  

### Future Work  
- Experiment with hybrid clustering approaches.  
- Implement advanced feature engineering for better results.  

---

## 5. Files Provided  
1. **Python Code:** `ir_ex2_2.py`  
2. **Excel File:** Contains top 20 features per classifier.  
3. **This README Document**  

---

## 6. References  
- Scikit-learn Documentation  
- Keras Official Tutorials  
- Machine Learning Mastery  

---

Let me know if you need further changes!
