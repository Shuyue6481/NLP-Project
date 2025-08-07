# run_imdb_experiments.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import os

def run_imdb_experiments(data_path="data/imdb_labeled.csv", tokenizer_path="tokenizers/bpe-imdb-30k.json"):
    """
    Trains and evaluates models on the IMDb dataset.
    """
    df = pd.read_csv(data_path).dropna()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # --- Baseline: TF-IDF + Logistic Regression ---
    print("\n--- Running TF-IDF + Logistic Regression Baseline ---")
    vectorizer = TfidfVectorizer(max_features=10000)
    X_train_tfidf = vectorizer.fit_transform(train_df['text'])
    X_test_tfidf = vectorizer.transform(test_df['text'])
    
    lr_model = LogisticRegression(max_iter=1000)
    lr_model.fit(X_train_tfidf, train_df['label'])
    
    preds = lr_model.predict(X_test_tfidf)
    accuracy = accuracy_score(test_df['label'], preds)
    f1 = f1_score(test_df['label'], preds, average='weighted')
    
    print(f"TF-IDF + LR -> Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # --- Fine-tuning DistilBERT with a custom tokenizer ---
    print(f"\n--- Running DistilBERT Fine-tuning with {tokenizer_path} ---")
    
    if not os.path.exists(tokenizer_path):
        print(f"Tokenizer not found at {tokenizer_path}. Please train it first.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=256)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)
    
    # Key Change: num_labels=2 for positive/negative sentiment
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir="results/distilbert_imdb_finetuned",
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=2e-5,
        evaluation_strategy="epoch",
        logging_steps=100,
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_test_dataset,
    )
    
    trainer.train()
    eval_results = trainer.evaluate()
    print(f"DistilBERT with {tokenizer_path} -> Eval Accuracy: {eval_results['eval_accuracy']:.4f}, Eval F1: {eval_results['eval_f1']:.4f}")
    
if __name__ == "__main__":
    run_imdb_experiments()