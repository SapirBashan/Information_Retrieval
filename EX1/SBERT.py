from sentence_transformers import SentenceTransformer
import pandas as pd
import os

# Load SBERT model
sbert_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can change the model to another SBERT variation

# Load Excel file
file_path = r'posts_first_targil.xlsx'
sheets = pd.ExcelFile(file_path).sheet_names

# Output directory
output_dir = "SBERT_Embeddings"
os.makedirs(output_dir, exist_ok=True)

# Prepare results
all_results = []

for sheet_name in sheets:
    print(f"Processing sheet: {sheet_name}")
    df = pd.read_excel(file_path, sheet_name=sheet_name)

    if 'Body Text' not in df.columns:
        print(f"'Body Text' column not found in {sheet_name}, skipping...")
        continue

    sbert_vectors = []

    for idx, row in df.iterrows():
        text = str(row['Body Text']) if pd.notna(row['Body Text']) else ""

        # SBERT embeddings
        sbert_vectors.append(sbert_model.encode(text))

    # Combine results into a DataFrame
    sbert_df = pd.DataFrame({
        "SBERT": [list(vec) for vec in sbert_vectors],
    })

    # Save results
    output_csv_path = os.path.join(output_dir, f"{sheet_name}_sbert_embeddings.csv")
    sbert_df.to_csv(output_csv_path, index=False)
    print(f"Saved embeddings for sheet '{sheet_name}' to: {output_csv_path}")

    # Append to global results
    all_results.append(sbert_df)

# Save all results to a single CSV
all_results_df = pd.concat(all_results, ignore_index=True)
final_output_path = os.path.join(output_dir, "all_sbert_embeddings.csv")
all_results_df.to_csv(final_output_path, index=False)
print(f"All embeddings saved to: {final_output_path}")