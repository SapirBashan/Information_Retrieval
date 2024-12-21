# -*- coding: utf-8 -*-
"""IR_Ex2.2.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bKmoDSIPpGLyQivt2wITxpEg-raXxrPL
"""

!pip install umap-learn

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from collections import Counter
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import os

# שלב 1: קריאת הקובץ
file_path = "/content/drive/MyDrive/Ex2/IR-files/bert-sbert/sbert_vectors.csv"
data = pd.read_csv(file_path)

# בדיקה והצגה של הנתונים
print(data.head())
print()
print(data.info())
print()

# המרת עמודות הממדים (Dim0-Dim299) למטריצה
try:
    combined_matrix = data.loc[:, "Dim0":"Dim299"].to_numpy()  # כל העמודות מדימנסיה 0 עד 299
    print(f"Combined matrix shape: {combined_matrix.shape}")

    # שמירת עמודת Sheet כעמודת מטרה
    target_column = data['Sheet'].values
    print(f"Target column shape: {target_column.shape}")
except Exception as e:
    print(f"Error during conversion: {e}")

print(f"Example vector: {target_column}")  # הדפסת דוגמה לווקטור

print(f"Example vector: {combined_matrix[0]}")  # הדפסת דוגמה לווקטור

# K-Means
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans_labels = kmeans.fit_predict(combined_matrix)

# Silhouette Score
sil_score = silhouette_score(combined_matrix, kmeans_labels, metric='cosine')
print(f"Silhouette Score (K-Means): {sil_score}")

print(f"K-Means labels: {kmeans_labels}")

print(Counter(kmeans_labels))

# חישוב מטריצת מרחק (cosine)
distance_matrix = squareform(pdist(combined_matrix, metric='cosine'))

# חישוב MST (Minimum Spanning Tree) לצורך חישוב epsilon
mst = minimum_spanning_tree(distance_matrix).toarray()
epsilon = mst.max() if mst.max() > 0 else 0.5

print(f"Epsilon for DBSCAN: {epsilon}")

# DBSCAN
dbscan = DBSCAN(eps=epsilon, min_samples=25, metric='cosine')
dbscan_labels = dbscan.fit_predict(combined_matrix)

# הצגת תוצאות DBSCAN
print(f"DBSCAN labels: {dbscan_labels}")

print(Counter(dbscan_labels))

k = 5
nn = NearestNeighbors(n_neighbors=k, metric='cosine')
nn.fit(combined_matrix)
distances, indices = nn.kneighbors(combined_matrix)

# מיון המרחקים והצגה בגרף
distances = np.sort(distances[:, -1])
plt.plot(distances)
plt.title("K-Distance Graph")
plt.xlabel("Points")
plt.ylabel("Distance")
plt.show()

# DBSCAN
dbscan = DBSCAN(eps=0.2, min_samples=15, metric='cosine') # קביעת האפסילון ע"פ הגרף שקיבלנו
dbscan_labels = dbscan.fit_predict(combined_matrix)

# הצגת תוצאות DBSCAN
print(f"DBSCAN labels: {dbscan_labels}")
print(Counter(dbscan_labels))

# Mixture of Gaussians ע"י התפלגות גאוסיאנית
gmm = GaussianMixture(n_components=4, random_state=42)
gmm_labels = gmm.fit_predict(combined_matrix)

print(f"GMM labels: {gmm_labels}")
print(Counter(gmm_labels))

# tsne הורדה ל2 ממדים כדי לאפשר הצגה בגרף, תוך שמירה על יחסי הדמיון
tsne = TSNE(n_components=2, random_state=42)
reduced_data = tsne.fit_transform(combined_matrix)

# הצגת תוצאות ויזואליות עם K-Means
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=kmeans_labels, cmap='viridis')
plt.title("t-SNE Visualization with K-Means Clusters")
plt.colorbar(label='Cluster Label')
plt.show()

# תוצאות DBSCAN
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=dbscan_labels, cmap='plasma')
plt.title("t-SNE Visualization with DBSCAN Clusters")
plt.colorbar(label='Cluster Label')
plt.show()

# תוצאות GMM
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=gmm_labels, cmap='coolwarm')
plt.title("t-SNE Visualization with GMM Clusters")
plt.colorbar(label='Cluster Label')
plt.show()

# המרה של target_column לערכים מספריים באמצעות LabelEncoder
label_encoder = LabelEncoder()
numeric_target = label_encoder.fit_transform(target_column)  # ממיר מחרוזות למספרים

# חישוב המדדים עבור K-Means
print("Evaluation Metrics for K-Means:")
print(f"Precision: {precision_score(numeric_target, kmeans_labels, average='weighted')}")
print(f"Recall: {recall_score(numeric_target, kmeans_labels, average='weighted')}")
print(f"F1 Score: {f1_score(numeric_target, kmeans_labels, average='weighted')}")
print(f"Accuracy: {accuracy_score(numeric_target, kmeans_labels)}")
print()

# חישוב המדדים עבור DBSCAN
print("Evaluation Metrics for DBSCAN:")
print(f"Precision: {precision_score(numeric_target, dbscan_labels, average='weighted', zero_division=1)}") # הוספת הזירו דיבישין = 1 כדי להתמודד עם קבוצת הרעשים שנוצרת בדיביסקאן
print(f"Recall: {recall_score(numeric_target, dbscan_labels, average='weighted', zero_division=1)}")
print(f"F1 Score: {f1_score(numeric_target, dbscan_labels, average='weighted', zero_division=1)}")
print(f"Accuracy: {accuracy_score(numeric_target, dbscan_labels)}")
print()

# חישוב המדדים עבור GMM
print("Evaluation Metrics for GMM:")
print(f"Precision: {precision_score(numeric_target, gmm_labels, average='weighted')}")
print(f"Recall: {recall_score(numeric_target, gmm_labels, average='weighted')}")
print(f"F1 Score: {f1_score(numeric_target, gmm_labels, average='weighted')}")
print(f"Accuracy: {accuracy_score(numeric_target, gmm_labels)}")

"""# **עד כאן חלק א של תרגיל 2**"""

# טעינת הנתונים ויצירת המטריצה ומערך הסיווגים
def load_and_preprocess_data(file_path):
    try:
        # טען את הנתונים
        data = pd.read_csv(file_path)

        # זיהוי עמודות הממדים באופן דינאמי
        dimension_columns = [col for col in data.columns if col.startswith("Dim")]
        print(f"Found {len(dimension_columns)} dimension columns.")

        # המרת עמודות הממדים למטריצה
        combined_matrix = data[dimension_columns].to_numpy()
        print(f"Combined matrix shape: {combined_matrix.shape}")

        # שמירת עמודת Sheet כעמודת מטרה (בודק אם קיימת)
        if 'Sheet' in data.columns:
            target_column = data['Sheet'].values
            print(f"Target column shape: {target_column.shape}")
        else:
            raise KeyError("'Sheet' column not found in the data!")

        return combined_matrix, target_column

    except Exception as e:
        print(f"Error during processing file {file_path}: {e}")
        return None, None

# אימון והערכת המודל
def train_and_evaluate_classical_models(X_train, X_test, y_train, y_test):
    """Trains and evaluates NB, SVM, LoR, RF models."""
    results = {}
    models = {
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(kernel='linear', random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
        print(f"Results for {model_name}:\n", classification_report(y_test, y_pred))
    return results

from sklearn.model_selection import KFold

def train_and_evaluate_ann(X, y, num_classes):
    """Trains and evaluates an Artificial Neural Network with 10-Fold Cross-Validation."""
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    plt.figure(figsize=(10, 6))  # Initialize a single figure for aggregated plots

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        print(f"Training on Fold {fold}...")

        # Split data into training and validation for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Define the ANN model
        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(f"best_ann_model_fold_{fold}.keras", save_best_only=True)
        ]

        # Train the model
        history = model.fit(X_train, y_train,
                            epochs=15,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=0)

        # Evaluate the model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} Accuracy: {val_accuracy:.4f}")
        fold_accuracies.append(val_accuracy)

        # Plot training and validation accuracy for this fold
        plt.plot(history.history['accuracy'], label=f'Fold {fold} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Fold {fold} Val Accuracy')

    # Show aggregated accuracy plot
    plt.title("ANN (ReLU) Training and Validation Accuracy Across Folds")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()

    # Compute average accuracy across folds
    avg_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f"Average Accuracy Across Folds: {avg_accuracy:.4f}")

    return {'accuracy': avg_accuracy, 'fold_accuracies': fold_accuracies}

from sklearn.model_selection import KFold

# Function to train and evaluate ANN with GELU activation and an embedding layer
def train_and_evaluate_ann_gelu(X, y, num_classes):
    """Trains and evaluates an Artificial Neural Network with GELU activation and 10-Fold Cross-Validation."""
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y), start=1):
        print(f"Training on Fold {fold}...")

        # Splitting data into training and validation for this fold
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Build ANN Model with GELU activation
        model = Sequential([
            Dense(64, activation='gelu'),
            Dense(32, activation='gelu'),
            Dense(16, activation='gelu'),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True),
            ModelCheckpoint(f"best_ann_fold_{fold}.keras", save_best_only=True)
        ]

        # Train model
        history = model.fit(X_train, y_train,
                            epochs=15,
                            batch_size=32,
                            validation_data=(X_val, y_val),
                            callbacks=callbacks,
                            verbose=0)

        # Evaluate model
        val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
        print(f"Fold {fold} Accuracy: {val_accuracy:.4f}")
        fold_accuracies.append(val_accuracy)

        # Plotting training history
        plt.plot(history.history['accuracy'], label=f'Fold {fold} Train Accuracy')
        plt.plot(history.history['val_accuracy'], label=f'Fold {fold} Validation Accuracy')

    # Show aggregated accuracy plot
    plt.title("ANN Training History Across Folds")
    plt.legend()
    plt.show()

    # Average Accuracy
    avg_accuracy = np.mean(fold_accuracies)
    print(f"Average Accuracy Across Folds: {avg_accuracy:.4f}")

    return {'accuracy': avg_accuracy, 'fold_accuracies': fold_accuracies}

# הפונקציה הראשית, מזמנת את כל הפעולות לביצוע עבור כל מטריצה ומחזירה דאטה פריים של התוצאות
def main(file_path):
    print("Loading data...")
    combined_matrix, target_column = load_and_preprocess_data(file_path)

    # המרה של target_column לערכים מספריים באמצעות LabelEncoder
    label_encoder = LabelEncoder()
    numeric_target = label_encoder.fit_transform(target_column)  # ממיר מחרוזות למספרים

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(combined_matrix, numeric_target, test_size=0.2, random_state=42)

    print("Training classical models...")
    classical_results = train_and_evaluate_classical_models(X_train, X_test, y_train, y_test)

    print("Training ANN...")
    num_classes = len(np.unique(target_column))
    ann_results = train_and_evaluate_ann(combined_matrix, numeric_target, num_classes)

    print("Training ANN Gelu...")
    num_classes = len(np.unique(target_column))
    ann_gelu_results = train_and_evaluate_ann_gelu(combined_matrix, numeric_target, num_classes)

    # יצירת רשימה של שורות לתוצאה
    rows = []

    # הוספת תוצאות המודלים הקלאסיים
    for model_name, metrics in classical_results.items():
        for metric_name, metric_value in metrics.items():
            rows.append({"Model": model_name, "Metric": metric_name, "Value": metric_value})

    # הוספת תוצאות ה-ANN
    for metric_name, metric_value in ann_results.items():
        rows.append({"Model": "ANN", "Metric": metric_name, "Value": metric_value})

      # הוספת תוצאות ה-ANN
    for metric_name, metric_value in ann_gelu_results.items():
        rows.append({"Model": "ANN", "Metric": metric_name, "Value": metric_value})

    # יצירת DataFrame מכל השורות
    results = pd.DataFrame(rows)
    print("Results DataFrame created.")

    return results

# הרצת המיין עבור כל אחד מקבצי הנתונים
def process_files_in_folder(folder_path):
    # בדיקה אם התיקייה קיימת
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} not found!")
        return

    # עבור על כל התיקיות והקבצים בתיקייה ובתתי תיקיות
    for root, dirs, files in os.walk(folder_path):
        # מסנן רק קבצי CSV
        csv_files = [f for f in files if f.endswith('.csv')]

        print(f"Found {len(csv_files)} files to process in {root}.")

        for file_name in csv_files:
            try:
                # נתיב מלא לקובץ
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_name}")

                # הרצת main על הקובץ
                results = main(file_path)

                # שמירת התוצאה כקובץ CSV חדש
                result_file_name = f"result_{file_name}"
                result_file_path = os.path.join(root, result_file_name)

                # שמירת תוצאות, בהנחה ש-main מחזירה pandas DataFrame
                results.to_csv(result_file_path, index=False)
                print(f"Results saved to: {result_file_name}")

            except Exception as e:
                print(f"Error processing file {file_name}: {e}")

import os
# הרצת ההרצות
folder_path = "/content/drive/MyDrive/Ex2/IR-files"
process_files_in_folder(folder_path)

from google.colab import drive

# הרכבת ה-Drive
drive.mount('/content/drive')

# הצגת קבצים ב-Drive
!ls /content/drive/MyDrive

# Path to the directory
folder_path = "/content/drive/MyDrive/Ex2/IR-files"

# Function to get all CSV files in the directory and its subdirectories
def get_csv_files(path):
    csv_files = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.csv'):
                csv_files.append(os.path.join(root, file))
    return csv_files

# Get all CSV files
csv_files = get_csv_files(folder_path)

# Print the paths of the CSV files
for csv_file in csv_files:
    print(csv_file)

# דוגמה להרצה על הדוקטוואק בלבד
main("/content/drive/MyDrive/Ex2/IR-files/doc2vec/doc2vec_vectors.csv")