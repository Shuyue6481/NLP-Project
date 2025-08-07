# prepare_imdb_data.py
from datasets import load_dataset
import pandas as pd
import os

def prepare_imdb_data():
    """
    Loads the IMDb dataset and saves two files:
    1. A raw corpus (.txt) for tokenizer training.
    2. A labeled dataset (.csv) for model training.
    """
    print("Loading IMDb dataset...")
    # Load both train and test splits to get a larger corpus
    train_split = load_dataset("imdb", split="train")
    test_split = load_dataset("imdb", split="test")
    
    # Combine them into a single pandas DataFrame
    df_train = train_split.to_pandas()
    df_test = test_split.to_pandas()
    df = pd.concat([df_train, df_test], ignore_index=True)
    
    # --- 1. Create and save the raw corpus for tokenizer training ---
    os.makedirs("corpus", exist_ok=True)
    corpus_path = "corpus/imdb_corpus.txt"
    print(f"Writing raw text corpus to {corpus_path}...")
    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in df['text']:
            # Replace newlines within reviews with spaces for cleaner training
            f.write(text.replace("\n", " ") + "\n")
            
    # --- 2. Create and save the labeled data for model training ---
    os.makedirs("data", exist_ok=True)
    labeled_data_path = "data/imdb_labeled.csv"
    print(f"Saving labeled data for modeling to {labeled_data_path}...")
    # The combined dataframe is already in the right format ('text', 'label')
    df.to_csv(labeled_data_path, index=False)
    
    print("IMDb data preparation complete.")

if __name__ == "__main__":
    prepare_imdb_data()