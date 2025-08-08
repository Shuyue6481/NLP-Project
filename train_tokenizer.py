import os
import argparse
import sentencepiece as spm
from tokenizers import Tokenizer
from tokenizers.models import BPE, WordPiece, Unigram
from tokenizers.trainers import BpeTrainer, WordPieceTrainer, UnigramTrainer
from tokenizers.pre_tokenizers import Whitespace
# Import the normalizers to match your teammate's script
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence as NormalizerSequence

def train_tokenizer(config):
    """
    Trains a tokenizer based on the provided configuration, now including normalization.
    """
    if not os.path.exists(config.corpus_file):
        print(f"Error: Corpus file not found at {config.corpus_file}")
        return

    output_dir = os.path.dirname(config.output_path)
    os.makedirs(output_dir, exist_ok=True)

    model_type = config.model_type.lower()
    print(f"Starting training for {model_type.upper()} tokenizer...")

    if model_type in ['bpe', 'wordpiece', 'unigram']:
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
        
        # --- THIS IS THE NEW, IMPORTANT STEP ---
        # Add the normalizer to match your teammate's process
        tokenizer.normalizer = NormalizerSequence([NFD(), Lowercase(), StripAccents()])
        # -----------------------------------------

        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.train([config.corpus_file], trainer)
        tokenizer.save(config.output_path)
        print(f"Tokenizer saved to {config.output_path}")

    elif model_type.startswith('sentencepiece'):
        sp_model_type = model_type.split('-')[-1]
        model_prefix = os.path.splitext(config.output_path)[0]
        
        # SentencePiece handles normalization via command line flags,
        # the default behavior is very similar (NFC, lowercasing can be added).
        # We will stick to the standard, robust SentencePiece training command.
        train_command = (
            f"--input={config.corpus_file} --model_prefix={model_prefix} "
            f"--vocab_size={config.vocab_size} --model_type={sp_model_type} "
            f"--unk_id=0 --bos_id=1 --eos_id=2 --pad_id=3 "
            f"--user_defined_symbols=[CLS],[SEP],[MASK]"
        )
        spm.SentencePieceTrainer.train(train_command)
        print(f"SentencePiece model and vocab saved with prefix {model_prefix}")

    else:
        print(f"Error: Invalid model type '{model_type}'")
        return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a custom tokenizer.")
    parser.add_argument(
        "--model_type", type=str, required=True,
        choices=['bpe', 'wordpiece', 'unigram', 'sentencepiece-bpe', 'sentencepiece-unigram'],
        help="Type of tokenizer model to train."
    )
    parser.add_argument(
        "--corpus_file", type=str, required=True,
        help="Path to the raw text corpus file for training."
    )
    parser.add_argument(
        "--output_path", type=str, required=True,
        help="Path/Prefix to save the trained tokenizer."
    )
    parser.add_argument(
        "--vocab_size", type=int, default=10000,
        help="The desired size of the vocabulary."
    )

    args = parser.parse_args()
    train_tokenizer(args)