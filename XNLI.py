############################### Load Data from XNLI Dataset############################

# load English data
from datasets import load_dataset
import pandas as pd

def load_xnli_en():
    xnli_en = load_dataset("xnli", "en")
    def combine_text(example):
        return {
            "text": example["premise"] + " [SEP] " + example["hypothesis"],
            "label": example["label"]
        }
    en_data = xnli_en["train"].map(combine_text)
    return pd.DataFrame(en_data)[["text", "label"]]

df_en = load_xnli_en()

# load Vietnamese data
def load_xnli_vi():
    xnli_vi = load_dataset("xnli", "vi")
    def combine_text(example):
        return {
            "text": example["premise"] + " [SEP] " + example["hypothesis"],
            "label": example["label"]
        }
    vi_data = xnli_vi["train"].map(combine_text)
    return pd.DataFrame(vi_data)[["text", "label"]]

df_vi = load_xnli_vi()


############################### Implement BPE Tokenizer ###############################
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormalizerSequence

def train_bpe_tokenizer(texts, vocab_size=10000, save_path="bpe_tokenizer_en.json"):
    with open("bpe_train.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")

    tokenizer = Tokenizer(BPE())
    tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"])
    tokenizer.train(["bpe_train.txt"], trainer)
    tokenizer.save(save_path)
    return tokenizer

# Word tokenization Text → Add the tokens_str column
def tokenize_with_method(df, tokenizer_obj, method_name):
    df_copy = df.copy()
    df_copy[f"tokens_str_{method_name}"] = df_copy["text"].apply(
        lambda x: " ".join(tokenizer_obj.encode(x).tokens)
    )
    return df_copy

# Draw the Zipf curve
from collections import Counter
import matplotlib.pyplot as plt

def plot_zipf(df_tokens, method_name):
    token_counts = Counter()
    for text in df_tokens:
        token_counts.update(text.split())

    sorted_tokens = token_counts.most_common()
    ranks = list(range(1, len(sorted_tokens) + 1))
    freqs = [freq for _, freq in sorted_tokens]

    plt.figure(figsize=(8, 5))
    plt.loglog(ranks, freqs)
    title = f"Zipf Curve - {method_name}"
    plt.title(title)
    plt.xlabel("Rank")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(f"{title}.png", dpi=300)
    plt.close()

# Construct the TF-IDF Logistic Regression classifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_tfidf_lr(df, tokens_str_column):
    X_texts = df[tokens_str_column]
    y = df["label"]

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_texts)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"[{tokens_str_column}] Accuracy:", round(acc, 4))
    print(classification_report(y_test, y_pred, digits=4))
    return acc


# english BPE
# Step 1: load data 
df_en = load_xnli_en()
# Step 2: Train the BPE tokenizer
bpe_tokenizer = train_bpe_tokenizer(df_en["text"], vocab_size=10000)
# Step 3: word tokenization
df_bpe = tokenize_with_method(df_en, bpe_tokenizer, "bpe")
# Step 4: Draw the Zipf curve
plot_zipf(df_bpe["tokens_str_bpe"], "BPE - English")
# step 5: TF-IDF + Logistic Regression
acc_clean_en_bpe = run_tfidf_lr(df_bpe, "tokens_str_bpe")

import random

# Add Noise Function: Randomly replace a certain proportion of characters in the text to simulate OCR or spelling mistakes for English.
def add_typo_noise_English(text, error_rate=0.05):
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < error_rate:
            chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

# Add typo noise to English
df_en_bpe_noisy = df_en.copy()
df_en_bpe_noisy["text"] = df_en_bpe_noisy["text"].apply(lambda x: add_typo_noise_English(x, error_rate=0.05))
# Tokenize noisy text
df_bpe_noisy_en = tokenize_with_method(df_en_bpe_noisy, bpe_tokenizer, "bpe_noisy")
# Get noisy accuracy
acc_noisy_en_bpe = run_tfidf_lr(df_bpe_noisy_en, "tokens_str_bpe_noisy")
# Calculate drop using saved clean accuracy
drop_en_bpe = acc_clean_en_bpe - acc_noisy_en_bpe
print(f"Accuracy of English BPE model reduces: {drop_en_bpe * 100:.2f}%")


# Vietnamese BPE
bpe_tokenizer_vi = train_bpe_tokenizer(df_vi["text"], vocab_size=10000, save_path="bpe_tokenizer_vi.json")

df_vi_bpe = tokenize_with_method(df_vi, bpe_tokenizer_vi, "bpe")

plot_zipf(df_vi_bpe["tokens_str_bpe"], "BPE - Vietnamese")

acc_clean_vi_bpe = run_tfidf_lr(df_vi_bpe, "tokens_str_bpe")


# Add Noise to Vietnamese
import random

VIETNAMESE_CHARS = "aăâbcdđeêghiklmnoôơpqrstuưvxy" \
                   "AĂÂBCDĐEÊGHIKLMNOÔƠPQRSTUƯVXY" \
                   "áàảãạắằẳẵặấầẩẫậéèẻẽẹếềểễệ" \
                   "íìỉĩịóòỏõọốồổỗộớờởỡợúùủũụ" \
                   "ứừửữựýỳỷỹỵ" \
                   "ÁÀẢÃẠẮẰẲẴẶẤẦẨẪẬÉÈẺẼẸẾỀỂỄỆ" \
                   "ÍÌỈĨỊÓÒỎÕỌỐỒỔỖỘỚỜỞỠỢÚÙỦŨỤ" \
                   "ỨỪỬỮỰÝỲỶỸỴ"

def add_typo_noise_vietnamese(text, error_rate=0.05):
    """
    Randomly replace a proportion of characters (including Vietnamese diacritics)
    to simulate OCR or spelling errors.
    """
    chars = list(text)
    for i in range(len(chars)):
        if chars[i].isalpha() and random.random() < error_rate:
            chars[i] = random.choice(VIETNAMESE_CHARS)
    return "".join(chars)

# Add typo noise to Vietnamese
df_vi_noisy = df_vi.copy()
df_vi_noisy["text"] = df_vi_noisy["text"].apply(lambda x: add_typo_noise_vietnamese(x, error_rate=0.05))

df_noisy_vi_bpe = tokenize_with_method(df_vi_noisy, bpe_tokenizer_vi, "bpe_noisy")

print("After adding noise to Vietnamese BPE model:")
acc_noisy_vi_bpe = run_tfidf_lr(df_noisy_vi_bpe, "tokens_str_bpe_noisy")

drop_vi_bpe = acc_clean_vi_bpe - acc_noisy_vi_bpe
print(f"\n Accuracy of Vietnamese BPE model reduces: {round(drop_vi_bpe * 100, 2)}%")


############################### Implement WordPiece Tokenizer ###############################

from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormalizerSequence

def train_wordpiece_tokenizer(texts, vocab_size=10000, save_path="wordpiece_tokenizer.json"):
    with open("wp_train.txt", "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line + "\n")

    tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
    tokenizer.pre_tokenizer = Whitespace()

    trainer = WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    )

    tokenizer.train(["wp_train.txt"], trainer)
    tokenizer.save(save_path)

    return tokenizer

# Train the English WordPiece tokenizer
wordpiece_tokenizer_en = train_wordpiece_tokenizer(df_en["text"], save_path="wordpiece_tokenizer_en.json")

# Train the Vietnamese WordPiece tokenizer
wordpiece_tokenizer_vi = train_wordpiece_tokenizer(df_vi["text"], save_path="wordpiece_tokenizer_vi.json")

# Load the WordPiece tokenizer
from tokenizers import Tokenizer
wordpiece_tokenizer_en = Tokenizer.from_file("wordpiece_tokenizer_en.json")

# Word tokenization generates tokens_str_wordpiece
df_wp_en = tokenize_with_method(df_en, wordpiece_tokenizer_en, "wordpiece")
plot_zipf(df_wp_en["tokens_str_wordpiece"], "WordPiece - English")

acc_clean_en_wp = run_tfidf_lr(df_wp_en, "tokens_str_wordpiece")

# add noise
df_wp_en_noisy = df_en.copy()
df_wp_en_noisy["text"] = df_wp_en_noisy["text"].apply(lambda x: add_typo_noise_English(x, error_rate=0.05))

# Rework the new word tokenization using the WordPiece tokenizer for English
df_wp_en_noisy = tokenize_with_method(df_wp_en_noisy, wordpiece_tokenizer_en, "wordpiece_noisy")

acc_noisy_en_wp = run_tfidf_lr(df_wp_en_noisy, "tokens_str_wordpiece_noisy")
drop_en_wp = acc_clean_en_wp - acc_noisy_en_wp

print(f"\n WordPiece (English) Accuracy Drop: {round(drop_en_wp * 100, 2)}%")

# Word tokenization generates tokens_str_wordpiece for Vietnamese
wordpiece_tokenizer_vi = train_wordpiece_tokenizer(df_vi["text"], save_path="wordpiece_tokenizer_vi.json")
df_wp_vi = tokenize_with_method(df_vi, wordpiece_tokenizer_vi, "wordpiece")
plot_zipf(df_wp_vi["tokens_str_wordpiece"], "WordPiece - Vietnamese")
acc_clean_vi_wp = run_tfidf_lr(df_wp_vi, "tokens_str_wordpiece")

# Rework the new word tokenization using the WordPiece tokenizer for Vietnamese
df_wp_vi_noisy = df_vi.copy()
df_wp_vi_noisy["text"] = df_wp_vi_noisy["text"].apply(lambda x: add_typo_noise_vietnamese(x, error_rate=0.05))

df_wp_vi_noisy = tokenize_with_method(df_wp_vi_noisy, wordpiece_tokenizer_vi, "wordpiece_noisy")

acc_noisy_vi_wp = run_tfidf_lr(df_wp_vi_noisy, "tokens_str_wordpiece_noisy")

drop_vi_wp = acc_clean_vi_wp - acc_noisy_vi_wp
print(f"\n WordPiece (Vietnamese) Accuracy Drop: {round(drop_vi_wp * 100, 2)}%")


############################### Implement SentencePiece-BPE Tokenizer ###############################

import sentencepiece as spm
import os

def train_sentencepiece_tokenizer(texts, model_prefix="spm_model", vocab_size=10000, model_type="bpe"):
    """
    Train the SentencePiece tokenizer (BPE or Unigram) to generate model and vocab files.
    """
    # Write into the training corpus
    input_file = f"{model_prefix}_train.txt"
    with open(input_file, "w", encoding="utf-8") as f:
        for line in texts:
            f.write(line.strip() + "\n")

    # train
    spm.SentencePieceTrainer.train(
        input=input_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=1.0,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
        user_defined_symbols=["[SEP]"]
    )

    # Load trianed model
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

def tokenize_with_sentencepiece(df, sp_model, method_name):
    df_copy = df.copy()
    df_copy[f"tokens_str_{method_name}"] = df_copy["text"].apply(
        lambda x: " ".join(sp_model.encode(x, out_type=str))
    )
    return df_copy

# Train the SentencePiece-BPE tokenizer for English
sp_bpe_en = train_sentencepiece_tokenizer(df_en["text"], model_prefix="sp_bpe_en", model_type="bpe", vocab_size=10000)
df_sp_bpe_en = tokenize_with_sentencepiece(df_en, sp_bpe_en, "sp_bpe")

plot_zipf(df_sp_bpe_en["tokens_str_sp_bpe"], "SentencePiece-BPE English")

acc_clean_en_spbpe = run_tfidf_lr(df_sp_bpe_en, "tokens_str_sp_bpe")

# Add noise and train again
df_en_noisy = df_en.copy()
df_en_noisy["text"] = df_en_noisy["text"].apply(lambda x: add_typo_noise_English(x, error_rate=0.05))
df_sp_bpe_en_noisy = tokenize_with_sentencepiece(df_en_noisy, sp_bpe_en, "sp_bpe_noisy")
acc_noisy_en_spbpe  = run_tfidf_lr(df_sp_bpe_en_noisy, "tokens_str_sp_bpe_noisy")
drop_en_spbpe = acc_clean_en_spbpe - acc_noisy_en_spbpe
print(f"SentencePiece-BPE (English) Accuracy drop: {round(drop_en_spbpe*100, 2)}%")


# Train the SentencePiece-BPE tokenizer for Vietnamese
sp_bpe_vi = train_sentencepiece_tokenizer(
    df_vi["text"],
    model_prefix="sp_bpe_vi",
    vocab_size=8000,
    model_type="bpe"
)

df_sp_bpe_vi = tokenize_with_sentencepiece(df_vi, sp_bpe_vi, "sp_bpe")
plot_zipf(df_sp_bpe_vi["tokens_str_sp_bpe"], "SentencePiece-BPE Vietnamese")

acc_clean_vi_spbpe = run_tfidf_lr(df_sp_bpe_vi, "tokens_str_sp_bpe")

# Add noise and train again
df_sp_bpe_vi_noisy = df_vi.copy()
df_sp_bpe_vi_noisy["text"] = df_sp_bpe_vi_noisy["text"].apply(lambda x: add_typo_noise_vietnamese(x, error_rate=0.05))

df_sp_bpe_vi_noisy = tokenize_with_sentencepiece(df_sp_bpe_vi_noisy, sp_bpe_vi, "sp_bpe_noisy")
acc_noisy_vi_spbpe = run_tfidf_lr(df_sp_bpe_vi_noisy, "tokens_str_sp_bpe_noisy")

drop_vi_spbpe = acc_clean_vi_spbpe - acc_noisy_vi_spbpe
print(f"\n SentencePiece-BPE (Vietnamese) Accuracy Drop: {round(drop_vi_spbpe * 100, 2)}%")


############################### Implement SentencePiece-Unigram Tokenizer ###############################

# Train the SentencePiece-Unigram tokenizer for english
sp_unigram_en = train_sentencepiece_tokenizer(
    df_en["text"],
    model_prefix="sp_unigram_en",
    vocab_size=8000,
    model_type="unigram"
)

df_sp_unigram_en = tokenize_with_sentencepiece(df_en, sp_unigram_en, "sp_unigram")
plot_zipf(df_sp_unigram_en["tokens_str_sp_unigram"], "SentencePiece-Unigram English")

acc_clean_en_spuni = run_tfidf_lr(df_sp_unigram_en, "tokens_str_sp_unigram")

# Add noise and train again
df_sp_unigram_en_noisy = df_en.copy()
df_sp_unigram_en_noisy["text"] = df_sp_unigram_en_noisy["text"].apply(lambda x: add_typo_noise_English(x, error_rate=0.05))
df_sp_unigram_en_noisy = tokenize_with_sentencepiece(df_sp_unigram_en_noisy, sp_unigram_en, "sp_unigram_noisy")

acc_noisy_en_spuni = run_tfidf_lr(df_sp_unigram_en_noisy, "tokens_str_sp_unigram_noisy")

drop_en_spuni = acc_clean_en_spuni - acc_noisy_en_spuni
print(f"\n SentencePiece-Unigram (English) Accuracy Drop: {round(drop_en_spuni * 100, 2)}%")

# Train the SentencePiece-Unigram tokenizer for Vietnamese
sp_unigram_vi = train_sentencepiece_tokenizer(
    df_vi["text"],
    model_prefix="sp_unigram_vi",
    vocab_size=8000,
    model_type="unigram"
)
df_sp_unigram_vi = tokenize_with_sentencepiece(df_vi, sp_unigram_vi, "sp_unigram")
plot_zipf(df_sp_unigram_vi["tokens_str_sp_unigram"], "SentencePiece-Unigram Vietnamese")

acc_clean_vi_spuni = run_tfidf_lr(df_sp_unigram_vi, "tokens_str_sp_unigram")

# Add noise and train again
df_sp_unigram_vi_noisy = df_vi.copy()
df_sp_unigram_vi_noisy["text"] = df_sp_unigram_vi_noisy["text"].apply(lambda x: add_typo_noise_vietnamese(x, error_rate=0.05))
df_sp_unigram_vi_noisy = tokenize_with_sentencepiece(df_sp_unigram_vi_noisy, sp_unigram_vi, "sp_unigram_noisy")

acc_noisy_vi_spuni = run_tfidf_lr(df_sp_unigram_vi_noisy, "tokens_str_sp_unigram_noisy")

drop_vi_spuni = acc_clean_vi_spuni - acc_noisy_vi_spuni
print(f"\n SentencePiece-Unigram (Vietnamese) Accuracy Drop: {round(drop_vi_spuni * 100, 2)}%")


############################### Compare the Results ###############################
import pandas as pd

results = [
    ["BPE", "English", acc_clean_en_bpe, acc_noisy_en_bpe, drop_en_bpe],
    ["BPE", "Vietnamese", acc_clean_vi_bpe, acc_noisy_vi_bpe, drop_vi_bpe],
    ["WordPiece", "English", acc_clean_en_wp, acc_noisy_en_wp, drop_en_wp],
    ["WordPiece", "Vietnamese", acc_clean_vi_wp, acc_noisy_vi_wp, drop_vi_wp],
    ["SentencePiece-BPE", "English", acc_clean_en_spbpe, acc_noisy_en_spbpe, drop_en_spbpe],
    ["SentencePiece-BPE", "Vietnamese", acc_clean_vi_spbpe, acc_noisy_vi_spbpe, drop_vi_spbpe],
    ["SentencePiece-Unigram", "English", acc_clean_en_spuni, acc_noisy_en_spuni, drop_en_spuni],
    ["SentencePiece-Unigram", "Vietnamese", acc_clean_vi_spuni, acc_noisy_vi_spuni, drop_vi_spuni],
]

df_results = pd.DataFrame(results, columns=["Tokenizer", "Language", "Clean Acc", "Noisy Acc", "Drop (%)"])
df_results["Drop (%)"] = df_results["Drop (%)"].apply(lambda x: round(x * 100, 2))

print(df_results)

import matplotlib.pyplot as plt
import numpy as np

tokenizers = df_results["Tokenizer"].unique()
x = np.arange(len(tokenizers))

clean_en = df_results[df_results["Language"] == "English"]["Clean Acc"].values
noisy_en = df_results[df_results["Language"] == "English"]["Noisy Acc"].values
clean_vi = df_results[df_results["Language"] == "Vietnamese"]["Clean Acc"].values
noisy_vi = df_results[df_results["Language"] == "Vietnamese"]["Noisy Acc"].values

bar_width = 0.2

plt.figure(figsize=(10, 6))
plt.bar(x - 1.5*bar_width, clean_en, width=bar_width, label="English - Clean")
plt.bar(x - 0.5*bar_width, noisy_en, width=bar_width, label="English - Noisy")
plt.bar(x + 0.5*bar_width, clean_vi, width=bar_width, label="Vietnamese - Clean")
plt.bar(x + 1.5*bar_width, noisy_vi, width=bar_width, label="Vietnamese - Noisy")

plt.xticks(x, tokenizers, rotation=20)
plt.ylabel("Accuracy")
plt.title("Tokenizer Performance for XNLI")
plt.ylim(0.38, 0.46)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("accuracy_comparison_XNLI.png", dpi=300)
plt.close()