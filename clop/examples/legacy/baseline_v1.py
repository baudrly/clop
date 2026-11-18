#!/usr/bin/env python3

import argparse
import gzip
import re
from collections import Counter

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def parse_chromosample_header(header: str) -> tuple[str, str, str]:
    """
    Parses headers from the format: >species|biotype|description
    Returns: (species, biotype, description)
    Simplified from the main clop.py script.
    """
    if header.startswith('>'):
        header = header[1:]
    
    parts = header.split('|', 2)
    
    if len(parts) >= 2:
        species = parts[0].strip() or "unknown"
        biotype = parts[1].strip() or "unknown"
        description = parts[2].strip() if len(parts) > 2 else ""
    else:
        species = "unknown"
        biotype = "unknown"
        description = header.strip()
        
    return species, biotype, description


def parse_fasta(filepath: str) -> tuple[list[str], list[str]]:
    """
    Parses a FASTA file to extract sequences and biotypes.
    
    Returns:
        A tuple containing (list_of_sequences, list_of_biotypes).
    """
    sequences = []
    biotypes = []
    open_func = gzip.open if filepath.endswith(".gz") else open
    
    current_seq_chunks = []
    current_biotype = "unknown"

    with open_func(filepath, "rt", errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_seq_chunks:
                    sequences.append("".join(current_seq_chunks))
                    biotypes.append(current_biotype)
                
                _, biotype, _ = parse_chromosample_header(line)
                current_biotype = biotype
                current_seq_chunks = []
            else:
                current_seq_chunks.append(line.upper())

        if current_seq_chunks:
            sequences.append("".join(current_seq_chunks))
            biotypes.append(current_biotype)
            
    print(f"Parsed {len(sequences)} sequences from {filepath}")
    return sequences, biotypes

def calculate_gc_content(sequence: str) -> float:
    """Calculates the GC content of a DNA sequence."""
    if not sequence:
        return 0.0
    g_count = sequence.count('G')
    c_count = sequence.count('C')
    return (g_count + c_count) / len(sequence)

def main(args):
    """Main function to run the baseline evaluation."""

    sequences, labels = parse_fasta(args.input_file)
    if not sequences:
        print("Error: No sequences loaded. Check the input file path.")
        return

    print("\n--- Step 1: Engineering Features ---")
    
    print(f"Calculating {args.kmer_k}-mer frequencies...")

    vectorizer = CountVectorizer(
        analyzer='char', 
        ngram_range=(args.kmer_k, args.kmer_k),
        lowercase=False
    )
    X_kmers = vectorizer.fit_transform(sequences)
    print(f"K-mer feature matrix shape: {X_kmers.shape}")

    print("Calculating GC-content...")
    X_gc = np.array([calculate_gc_content(seq) for seq in sequences]).reshape(-1, 1)
    print(f"GC-content feature matrix shape: {X_gc.shape}")

    X_combined = hstack([X_kmers, X_gc]).tocsr() 
    print(f"Combined feature matrix shape: {X_combined.shape}")

    print("\n--- Step 2: Splitting Data ---")
    print(f"Creating a {args.validation_split:.0%} validation set with seed {args.seed}...")
    
    # stratification to ensure class distribution is similar in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, 
        labels, 
        test_size=args.validation_split, 
        random_state=args.seed,
        stratify=labels
    )
    print(f"Training samples: {X_train.shape[0]}, Validation samples: {X_test.shape[0]}")

    print("\n--- Step 3: Training Baseline Model ---")
    print("Training a Logistic Regression classifier...")
    model = LogisticRegression(
        random_state=args.seed,
        max_iter=1000, 
        solver='saga' # for large datasets and sparse data
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    # 5. Evaluation
    print("\n--- Step 4: Evaluating ---")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n==============================================")
    print("    Baseline Evaluation Report")
    print("==============================================")
    print(f"\nClassifier: Logistic Regression")
    print(f"Features: GC-Content + {args.kmer_k}-mer Frequencies")
    print("----------------------------------------------")
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy:.2%})")
    print("----------------------------------------------")
    print("\nDetailed Classification Report:\n")
    print(report)
    print("==============================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a baseline evaluation for biotype classification using simple sequence features.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--input_file", 
        type=str, 
        required=True,
        help="Path to the input FASTA file (e.g., filtered_sampled_dataset.fa)."
    )
    parser.add_argument(
        "--kmer_k", 
        type=int, 
        default=6,
        help="Size of k-mers to use as features."
    )
    parser.add_argument(
        "--validation_split", 
        type=float, 
        default=0.1,
        help="Fraction of the data to use for the validation set."
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed for reproducibility of the train/test split."
    )
    
    args = parser.parse_args()
    main(args)