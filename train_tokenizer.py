import os
import argparse
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace

def train_tokenizer(config):
    """
    Trains a tokenizer based on the provided configuration.
    Handles BPE, WordPiece, Unigram (via tokenizers lib) and
    SentencePiece (via its own library).
    """
    # Check if the corpus file exists
    if not os.path.exists(config.corpus_file):
        print(f"Error: Corpus file not found at {config.corpus_file}")
        return

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(config.output_path)
    os.makedirs(output_dir, exist_ok=True)

    model_type = config.model_type.lower()
    print(f"Starting training for {model_type.upper()} tokenizer...")

    if model_type in ['bpe', 'wordpiece', 'unigram']:
        # --- Handle training for BPE, WordPiece, Unigram ---
        if model_type == 'bpe':
            model = BPE(unk_token="[UNK]")
            trainer = BpeTrainer(vocab_size=config.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        elif model_type == 'wordpiece':
            model = WordPiece(unk_token="[UNK]")
            trainer = WordPieceTrainer(vocab_size=config.vocab_size, special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
        elif model_type == 'unigram':
            model = Unigram()
            trainer = UnigramTrainer(vocab_size=config.vocab_size, unk_token="[UNK]", special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])

        tokenizer = Tokenizer(model)
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([config.corpus_file], trainer)
        # Save as a single JSON file
        tokenizer.save(config.output_path)
        print(f"Tokenizer saved to {config.output_path}")

    elif model_type == 'sentencepiece':
        # --- Handle training for SentencePiece ---
        # SentencePiece trains directly from the corpus and saves a .model and .vocab file.
        model_prefix = os.path.splitext(config.output_path)[0]

        # Command-line style string for training.
        train_command = (
            f"--input={config.corpus_file} --model_prefix={model_prefix} "
            f"--vocab_size={config.vocab_size} --model_type=bpe "
            f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 "
            f"--user_defined_symbols=[CLS],[SEP],[MASK]"
        )
        spm.SentencePieceTrainer.train(train_command)
        print(f"SentencePiece model and vocab saved with prefix {model_prefix}")
        print(f"Main model file: {model_prefix}.model")

    else:
        print(f"Error: Invalid model type '{model_type}'")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom tokenizer.")
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=['bpe', 'wordpiece', 'unigram', 'sentencepiece'],
        help="Type of tokenizer model to train."
    )
    parser.add_argument(
        "--corpus_file",
        type=str,
        required=True,
        help="Path to the raw text corpus file for training."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path/Prefix to save the trained tokenizer. For sentencepiece, this is a prefix."
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=30000,
        help="The desired size of the vocabulary."
    )

    args = parser.parse_args()
    train_tokenizer(args)