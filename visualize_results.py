# visualize_results.py (Final Version with Drop Rate Chart)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def parse_tokenizer_info(df):
    """
    Takes the DataFrame with the messy Tokenizer column and creates
    clean 'Tokenizer Name' and 'Vocab Size' columns.
    """
    def get_vocab_size(tokenizer_string):
        if "-2K" in tokenizer_string.upper():
            return 2000
        else:
            return 10000
            
    def get_clean_name(tokenizer_string):
        name_upper = tokenizer_string.upper()
        if "SENTENCEPIECE-BPE" in name_upper: return "SentencePiece-BPE"
        if "SENTENCEPIECE-UNIGRAM" in name_upper: return "SentencePiece-Unigram"
        if "WORDPIECE" in name_upper: return "WordPiece"
        if "BPE" in name_upper: return "BPE"
        return tokenizer_string

    df['Vocab Size'] = df['Tokenizer'].apply(get_vocab_size)
    df['Tokenizer Name'] = df['Tokenizer'].apply(get_clean_name)
    df['Configuration'] = df['Tokenizer Name'] + '-' + (df['Vocab Size'] // 1000).astype(str) + 'k'
    return df

def visualize_all_results(results_path="results/full_results.csv"):
    """
    Reads the final results CSV and creates two charts:
    1. A grouped bar chart for performance.
    2. A sorted bar chart for robustness (drop rate).
    """
    if not os.path.exists(results_path):
        print(f"Results file not found at {results_path}. Please run the experiments first.")
        return

    df = pd.read_csv(results_path)
    df = parse_tokenizer_info(df)
    
    # --- CHART 1: PERFORMANCE (CLEAN VS. NOISY ACCURACY) ---
    print("Generating Performance Chart (Clean vs. Noisy Accuracy)...")
    df_perf = df.sort_values(by=['Tokenizer Name', 'Vocab Size']).reset_index()
    
    labels = df_perf['Configuration']
    clean_acc = df_perf['Clean Acc']
    noisy_acc = df_perf['Noisy Acc']

    x = np.arange(len(labels))
    width = 0.35

    fig1, ax1 = plt.subplots(figsize=(14, 8))
    rects1 = ax1.bar(x - width/2, clean_acc, width, label='Clean Accuracy', color='steelblue')
    rects2 = ax1.bar(x + width/2, noisy_acc, width, label='Noisy Accuracy (5% error)', color='lightcoral')

    ax1.set_ylabel('Accuracy')
    ax1.set_title('Tokenizer Performance and Robustness on IMDb Dataset')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha="right")
    ax1.legend()
    ax1.set_ylim([0.80, 0.95])
    
    ax1.bar_label(rects1, padding=3, fmt='%.4f')
    ax1.bar_label(rects2, padding=3, fmt='%.4f')
    fig1.tight_layout()

    perf_chart_path = "results/final_performance_chart.png"
    plt.savefig(perf_chart_path)
    print(f"Performance chart saved to: {perf_chart_path}")
    plt.close(fig1) # Close the figure to prepare for the next one

    # --- CHART 2: ROBUSTNESS (ACCURACY DROP %) ---
    print("\nGenerating Robustness Chart (Accuracy Drop)...")
    df_robust = df.sort_values(by='Drop (%)', ascending=True).reset_index()

    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bars = ax2.bar(df_robust['Configuration'], df_robust['Drop (%)'], color='seagreen')

    ax2.set_ylabel('Accuracy Drop (%)')
    ax2.set_title('Robustness to Noise (Lower is Better)')
    ax2.tick_params(axis='x', rotation=45)
    
    ax2.bar_label(bars, fmt='%.2f%%')
    
    fig2.tight_layout()

    robust_chart_path = "results/robustness_drop_chart.png"
    plt.savefig(robust_chart_path)
    print(f"Robustness chart saved to: {robust_chart_path}")
    plt.show()


if __name__ == "__main__":
    visualize_all_results()