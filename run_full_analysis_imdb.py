# run_full_analysis.py (Final Unified Version)
import pandas as pd
import argparse
import random
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from tokenizers import Tokenizer
import sentencepiece as spm

def add_typo_noise(text, error_rate=0.05):
    """Adds typo noise to a string."""
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < error_rate:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

def analyze_zipf(tokenized_texts, tokenizer_name):
    """Generates and saves a Zipf plot from already tokenized text."""
    print(f"\n--- Analyzing Zipf Curve for Tokenizer: {tokenizer_name} ---")
    
    token_counts = Counter()
    for text in tokenized_texts:
        token_counts.update(text.split())

    frequencies = [count for token, count in token_counts.most_common()]
    ranks = range(1, len(frequencies) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, frequencies, marker=".")
    
    plt.title(f"Zipf Curve for {tokenizer_name} on IMDb Corpus")
    plt.xlabel("Token Rank (log scale)")
    plt.ylabel("Token Frequency (log scale)")
    plt.grid(True)
    
    output_filename = f"results/zipf_plot_{tokenizer_name}.png"
    plt.savefig(output_filename)
    print(f"Zipf plot saved to {output_filename}")

def run_full_analysis(config):
    """
    Runs a complete analysis for a single tokenizer, now supporting all types.
    """
    print(f"--- Starting Full Analysis for Tokenizer: {config.tokenizer_path} ---")
    df = pd.read_csv(config.data_path).dropna()

    tokenizer_name = os.path.basename(config.tokenizer_path)

    # --- Pre-tokenization Step (Handles both .json and .model files) ---
    print("Pre-tokenizing text with custom tokenizer...")
    if config.tokenizer_path.endswith('.json'):
        # Handle BPE, WordPiece, Unigram from 'tokenizers' library
        custom_tokenizer = Tokenizer.from_file(config.tokenizer_path)
        df['tokenized_text'] = df['text'].apply(lambda x: " ".join(custom_tokenizer.encode(x).tokens))
    elif config.tokenizer_path.endswith('.model'):
        # Handle SentencePiece models
        sp = spm.SentencePieceProcessor()
        sp.load(config.tokenizer_path)
        df['tokenized_text'] = df['text'].apply(lambda x: " ".join(sp.encode(x, out_type=str)))
    else:
        print(f"Unsupported tokenizer file type: {config.tokenizer_path}")
        return

    # --- Part 1: Zipf Analysis ---
    analyze_zipf(df['tokenized_text'], tokenizer_name)

    # --- Part 2: Robustness Test ---
    print("\n--- Running Robustness Test ---")
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    print("Training model on clean data...")
    # Build a FIXED feature space tied to the tokenizer's own vocabulary
    if config.tokenizer_path.endswith('.json'):
        # Hugging Face tokenizers: use token list directly
        token_list = list(custom_tokenizer.get_vocab().keys())
    else:
        # SentencePiece: read tokens from companion .vocab file
        sp_vocab_path = os.path.splitext(config.tokenizer_path)[0] + ".vocab"
        token_list = []
        with open(sp_vocab_path, "r", encoding="utf-8") as vf:
            for line in vf:
                tok = line.strip().split("\t")[0]
                if tok:
                    token_list.append(tok)

    # The vectorizer uses the fixed vocabulary and respects pre-tokenized input
    vectorizer = TfidfVectorizer(
        vocabulary=token_list,
        tokenizer=str.split,
        token_pattern=None,
        lowercase=False,
    )
    X_train_tfidf = vectorizer.fit_transform(train_df['tokenized_text'])
    lr_model = LogisticRegression(max_iter=1000).fit(X_train_tfidf, train_df['label'])

    # Evaluate on clean data
    X_test_clean_tfidf = vectorizer.transform(test_df['tokenized_text'])
    clean_preds = lr_model.predict(X_test_clean_tfidf)
    clean_accuracy = accuracy_score(test_df['label'], clean_preds)
    
    # Evaluate on noisy data
    test_df_noisy = test_df.copy()
    test_df_noisy['text'] = test_df_noisy['text'].apply(add_typo_noise)
    # Re-tokenize the noisy text
    if config.tokenizer_path.endswith('.json'):
        test_df_noisy['tokenized_text'] = test_df_noisy['text'].apply(lambda x: " ".join(custom_tokenizer.encode(x).tokens))
    else:
        test_df_noisy['tokenized_text'] = test_df_noisy['text'].apply(lambda x: " ".join(sp.encode(x, out_type=str)))
    
    X_test_noisy_tfidf = vectorizer.transform(test_df_noisy['tokenized_text'])
    noisy_accuracy = accuracy_score(test_df_noisy['label'], lr_model.predict(X_test_noisy_tfidf))

    # Print reports
    print("\n--- Detailed Report on CLEAN Data ---")
    print(classification_report(test_df['label'], clean_preds, digits=4))
    print("\n--- Detailed Report on NOISY Data ---")
    print(classification_report(test_df_noisy['label'], lr_model.predict(X_test_noisy_tfidf), digits=4))

    # Final summary
    drop_percent = ((clean_accuracy - noisy_accuracy) / clean_accuracy) * 100
    print("\n--- Robustness Summary ---")
    print(f"Accuracy on clean data: {clean_accuracy:.4f}")
    print(f"Accuracy on noisy data (5% error): {noisy_accuracy:.4f}")
    print(f"Accuracy Drop: {drop_percent:.4f}%")
    print("---------------------------------------------------\n")
    
    # Save summary to CSV
    results_path = "results/full_results.csv"
    tokenizer_name_clean = tokenizer_name.replace("-imdb-10k.json", "").replace("-imdb-10k.model", "")
    clean_accuracy_rounded = round(clean_accuracy, 4)
    noisy_accuracy_rounded = round(noisy_accuracy, 4)
    drop_percent_rounded = round(drop_percent, 4)
    new_result = pd.DataFrame([
        {
            "Tokenizer": tokenizer_name_clean.upper(),
            "Clean Acc": clean_accuracy_rounded,
            "Noisy Acc": noisy_accuracy_rounded,
            "Drop (%)": drop_percent_rounded,
        }
    ])
    if os.path.exists(results_path):
        all_results = pd.read_csv(results_path)
        all_results = all_results[all_results.Tokenizer != tokenizer_name_clean.upper()]
        all_results = pd.concat([all_results, new_result], ignore_index=True)
    else:
        all_results = new_result
    all_results.to_csv(results_path, index=False)
    print(f"Results updated in {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a full analysis for a tokenizer.")
    parser.add_argument("--tokenizer_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/imdb_labeled.csv")
    parser.add_argument("--corpus_path", type=str, default="corpus/imdb_corpus.txt")
    args = parser.parse_args()
    run_full_analysis(args)