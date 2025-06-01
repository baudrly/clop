#!/usr/bin/env python3
import argparse
import os
import sys
import gzip
import re
import json
from collections import Counter, defaultdict
from typing import List, Tuple, Dict, Any, Optional, Union, Callable, Type
import tempfile
import shutil
import unittest
import io 
import logging
import math
import warnings
import contextlib
import subprocess # For TensorFlow.js conversion
import unittest.mock # For TFJS export test mocking
import time # For timing sections
import urllib.parse # For GFF attribute parsing

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd
try:
    import polars as pl
except ImportError:
    pl = None 

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix, classification_report as sk_classification_report,
                             silhouette_score, davies_bouldin_score)
from sklearn.neighbors import KNeighborsClassifier # Moved to top for clarity
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap 
except ImportError:
    umap = None 

import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
from fpdf import FPDF
try:
    import PIL.Image
except ImportError:
    PIL = None

try:
    import onnx
    import onnxruntime
except ImportError:
    onnx = None
    onnxruntime = None

# Optional TF related imports for TF.js export
try:
    import tensorflow # For type hinting and conditional logic
except ImportError:
    tensorflow = None
try:
    from onnx_tf.backend import prepare as onnx_tf_prepare
except ImportError:
    # Try tf2onnx.convert if onnx_tf is not available (alternative path)
    # This script currently uses onnx_tf, so onnx_tf_prepare is primary.
    try:
        # tf2onnx is generally used for TF -> ONNX, not ONNX -> TF.
        # For ONNX -> TF, onnx-tf is the more direct tool.
        # If onnx_tf_prepare is None, TF.js export will be skipped if it relies on it.
        pass
    except ImportError:
        pass
    onnx_tf_prepare = None # Explicitly None if primary import failed
try:
    import tensorflowjs # For type hinting and conditional logic
except ImportError:
    tensorflowjs = None


# --- Global Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)

DNA_CHARACTERS = ['A', 'C', 'G', 'T', 'N'] 
VALID_DNA_CHARS_SET = set(DNA_CHARACTERS)
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
DEFAULT_KMER_K = 6
DEFAULT_CHAR_DNA_VOCAB = {PAD_TOKEN: 0, UNK_TOKEN: 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, 'N': 6}
DEFAULT_EMBEDDING_DIM = 128
DEFAULT_HIDDEN_DIM = 256
DEFAULT_NUM_LAYERS = 1
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_EPOCHS = 10
DEFAULT_TEMPERATURE = 0.07
ONNX_OPSET_VERSION = 12
DEFAULT_GFF_FEATURE_TYPES = ['gene', 'pseudogene', 'ncRNA_gene', 'lnc_RNA', 'miRNA', 'mRNA', 'transcript', 'rRNA', 'snoRNA', 'snRNA', 'tRNA', 'antisense_RNA']

# Blacklist of biotypes to exclude (instead of whitelist)
EXCLUDED_BIOTYPES = frozenset({
    "unknown_biotype", "unknown", "N/A", "random_dna", 
    "unknown_biotype_fasta_header", "unknown_species_fasta_header",
    "unknown_biotype_bed_ref", "unknown_bed_feature", "unknown_gff_species"
})

# --- Helper Functions ---
def reverse_complement(dna_sequence: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    complement_map = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N': 'N',
                      'R': 'Y', 'Y': 'R', 'S': 'S', 'W': 'W', 'K': 'M', 'M': 'K',
                      'B': 'V', 'D': 'H', 'H': 'D', 'V': 'B'}
    return "".join(complement_map.get(base, 'N') for base in reversed(dna_sequence.upper()))

def parse_gff_attributes(attributes_str: str) -> Dict[str, str]:
    """Parses a GFF3 attributes string into a dictionary."""
    attributes = {}
    if not attributes_str or attributes_str == '.':
        return attributes
    for part in attributes_str.split(';'):
        if not part.strip():
            continue
        if '=' in part:
            key, value = part.split('=', 1)
            attributes[key.strip()] = urllib.parse.unquote(value.strip())
        else: # Flag attribute or malformed (store as is, or with a default value like True)
            attributes[part.strip()] = "" # Storing as empty string if no value
    return attributes

def parse_chromosample_header(header: str) -> Tuple[str, str, str]:
    """
    Parse headers from chromosample.py output format: >species|biotype|description
    Returns: (species, biotype, description)
    """
    # Remove '>' if present
    if header.startswith('>'):
        header = header[1:]
    
    # Split by '|' - expecting format: species|biotype|description
    parts = header.split('|', 2)  # Split into at most 3 parts
    
    if len(parts) >= 3:
        species = parts[0].strip()
        biotype = parts[1].strip()
        description = parts[2].strip()
    elif len(parts) == 2:
        species = parts[0].strip()
        biotype = parts[1].strip()
        description = ""
    elif len(parts) == 1:
        # Try to extract any useful info from single part
        species = "unknown_species"
        biotype = "unknown_biotype"
        description = parts[0].strip()
    else:
        species = "unknown_species"
        biotype = "unknown_biotype"
        description = ""
    
    # Clean up common issues
    if species in ["unknown_species_fasta_header", ""]:
        species = "unknown_species"
    if biotype in ["unknown_biotype_fasta_header", ""]:
        biotype = "unknown_biotype"
    
    return species, biotype, description

# --- Tokenizers ---
class BaseTokenizer:
    """Base class for tokenizers."""
    def __init__(self, vocab: Optional[Dict[str, int]] = None, k: Optional[int] = None): # k for DNA kmer
        self.k = k 
        _user_vocab = vocab.copy() if vocab else {}

        final_vocab_staging = {}
        
        # Ensure PAD and UNK are present and correctly indexed
        self.pad_token_id = 0
        self.unk_token_id = 1
        final_vocab_staging[PAD_TOKEN] = self.pad_token_id
        final_vocab_staging[UNK_TOKEN] = self.unk_token_id

        # Remove PAD/UNK from user_vocab as they are handled
        _user_vocab.pop(PAD_TOKEN, None)
        _user_vocab.pop(UNK_TOKEN, None)

        # Add tokens from user_vocab, respecting their indices if possible
        # Store tokens that need re-indexing separately
        tokens_to_reindex_clash_pad_unk = {}
        tokens_to_reindex_clash_other = {} # Stores token: original_idx
        
        occupied_indices = {self.pad_token_id, self.unk_token_id}

        # First pass: add non-conflicting tokens and identify conflicts
        for token, idx in _user_vocab.items():
            if idx == self.pad_token_id or idx == self.unk_token_id:
                tokens_to_reindex_clash_pad_unk[token] = idx # Will be re-indexed
            elif idx in occupied_indices: # User token's index is already taken by another user token
                tokens_to_reindex_clash_other[token] = idx # Will be re-indexed
            else: # Index is fine for now
                final_vocab_staging[token] = idx
                occupied_indices.add(idx)
        
        # Combine all tokens that need re-indexing
        all_tokens_to_reindex = {}
        all_tokens_to_reindex.update(tokens_to_reindex_clash_pad_unk)
        all_tokens_to_reindex.update(tokens_to_reindex_clash_other)
            
        # Re-index the problematic tokens deterministically
        next_available_idx = 2 # Start re-indexing from 2
        
        # Sort tokens for deterministic re-indexing if their original indices were problematic
        for token in sorted(all_tokens_to_reindex.keys()):
            while next_available_idx in occupied_indices:
                next_available_idx += 1
            final_vocab_staging[token] = next_available_idx
            occupied_indices.add(next_available_idx) # Mark new index as occupied
        
        self.vocab = final_vocab_staging


    def tokenize(self, text_or_sequence: str) -> List[str]:
        raise NotImplementedError("Subclasses must implement tokenize")

    def encode(self, text_or_sequence: str, max_length: Optional[int] = None) -> Tuple[List[int], int]:
        tokens = self.tokenize(text_or_sequence)
        
        effective_length = len(tokens)
        if max_length is not None and max_length > 0:
            if effective_length > max_length:
                tokens = tokens[:max_length]
                effective_length = max_length
        
        encoded_ids = [self.vocab.get(token, self.unk_token_id) for token in tokens]
        
        if max_length is not None and max_length > 0:
            padding_needed = max_length - len(encoded_ids)
            if padding_needed > 0:
                encoded_ids.extend([self.pad_token_id] * padding_needed)
        
        if not encoded_ids and max_length is not None and max_length > 0:
            encoded_ids = [self.pad_token_id] * max_length
            effective_length = 1 
        elif not encoded_ids: 
            encoded_ids = [self.unk_token_id] 
            effective_length = 1

        if effective_length == 0 and len(tokens) == 0: 
             effective_length = 1 

        return encoded_ids, effective_length

    def get_vocab_size(self) -> int:
        return len(self.vocab)

    def save_vocab(self, filepath: str):
        try:
            config_to_save = {'vocab': self.vocab}
            if self.k is not None: 
                config_to_save['k'] = self.k
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(config_to_save, f, indent=2)
            logger.info(f"Tokenizer vocabulary saved to {filepath}")
        except IOError as e:
            logger.error(f"Failed to save tokenizer vocabulary to {filepath}: {e}")

    @classmethod
    def load_vocab(cls, filepath: str) -> 'BaseTokenizer':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            loaded_vocab = data.get('vocab', {})
            if loaded_vocab.get(PAD_TOKEN) != 0 or loaded_vocab.get(UNK_TOKEN) != 1:
                logger.warning(f"Loaded vocab from {filepath} has non-standard PAD/UNK token indices. These will be enforced to 0 and 1 respectively during instantiation.")
            
            k_val = data.get('k')
            # Pass k explicitly to constructor if it's relevant for the class
            if k_val is not None and hasattr(cls, '__init__') and 'k' in cls.__init__.__code__.co_varnames:
                instance = cls(vocab=loaded_vocab, k=k_val)
            else:
                instance = cls(vocab=loaded_vocab)

            logger.info(f"Tokenizer vocabulary loaded from {filepath} for {cls.__name__}")
            return instance
        except FileNotFoundError: 
            logger.error(f"Tokenizer vocabulary file not found: {filepath}"); raise
        except Exception as e: 
            logger.error(f"Error loading/parsing tokenizer vocabulary {filepath}: {e}"); raise


class KmerDNATokenizer(BaseTokenizer):
    """Tokenizes DNA sequences into k-mers."""
    def __init__(self, k: int = DEFAULT_KMER_K, vocab: Optional[Dict[str, int]] = None):
        if k <= 0: raise ValueError("k-mer size k must be positive.")
        super().__init__(vocab=vocab, k=k)

    def tokenize(self, sequence: str) -> List[str]:
        sequence = sequence.upper()
        if not sequence or len(sequence) < self.k:
            return [UNK_TOKEN]

        kmers = []
        for i in range(len(sequence) - self.k + 1):
            kmer = sequence[i:i+self.k]
            if all(c in VALID_DNA_CHARS_SET for c in kmer): # Looser check, allow N in k-mers
                 kmers.append(kmer)
        return kmers if kmers else [UNK_TOKEN]

class CharDNATokenizer(BaseTokenizer):
    """Tokenizes DNA sequences into characters."""
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        super().__init__(vocab=vocab if vocab else DEFAULT_CHAR_DNA_VOCAB.copy())

    def tokenize(self, sequence: str) -> List[str]:
        sequence = sequence.upper()
        if not sequence:
            return [UNK_TOKEN]
        # Keep N, but filter out other non-standard DNA chars
        tokens = [char for char in sequence if char in VALID_DNA_CHARS_SET]
        return tokens if tokens else [UNK_TOKEN]


class TextTokenizer(BaseTokenizer):
    """Simple text tokenizer."""
    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        super().__init__(vocab=vocab)

    def _clean_text(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s_'\-\.:]", "", text) # Allow period and colon
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        if not text or not text.strip(): 
            return [UNK_TOKEN]
        cleaned_text = self._clean_text(text)
        tokens = cleaned_text.split()
        return tokens if tokens else [UNK_TOKEN]


def build_vocab_from_data(
    texts: List[str], 
    tokenizer_cls: Type[BaseTokenizer], 
    min_freq: int = 1, 
    tokenizer_args: Optional[Dict[str, Any]] = None
) -> BaseTokenizer:
    
    tokenizer_name = tokenizer_cls.__name__
    logger.info(f"Building {tokenizer_name} vocab from {len(texts)} texts with min_freq={min_freq}" + 
                (f", args={tokenizer_args}" if tokenizer_args else ""))
    
    build_start_time = time.time()

    if tokenizer_cls == CharDNATokenizer and not tokenizer_args: 
        logger.info(f"Using fixed vocab for {tokenizer_name}.")
        return CharDNATokenizer()

    _tokenizer_args = tokenizer_args if tokenizer_args else {}
    temp_tokenizer_instance = tokenizer_cls(vocab={}, **_tokenizer_args) 

    token_counts = Counter()
    for i, text_item in enumerate(texts):
        if (i+1) % 100000 == 0: 
            logger.info(f"Processed {i+1}/{len(texts)} for vocab building...")
        tokens = temp_tokenizer_instance.tokenize(text_item)
        token_counts.update(tokens)
    logger.info(f"Token counting finished in {time.time() - build_start_time:.2f}s.")

    # vocab_dict_for_builder will not include PAD/UNK yet.
    # BaseTokenizer constructor will handle PAD/UNK and re-indexing if necessary.
    vocab_dict_for_builder = {}
    current_idx_for_builder = 2 # Start from 2, as BaseTokenizer will handle 0 and 1 for PAD/UNK
    
    for token, count in token_counts.most_common():
        if count >= min_freq: # Don't add PAD/UNK here, BaseTokenizer handles them.
            if token != PAD_TOKEN and token != UNK_TOKEN:
                 vocab_dict_for_builder[token] = current_idx_for_builder
                 current_idx_for_builder +=1
    
    logger.info(f"Built vocab with {len(vocab_dict_for_builder) + 2} effective tokens for {tokenizer_name} (including PAD/UNK). Total time: {time.time() - build_start_time:.2f}s")
    # Pass this potentially sparse vocab to the main constructor which will normalize it.
    return tokenizer_cls(vocab=vocab_dict_for_builder, **_tokenizer_args)


# --- Model Architecture ---
class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class TransformerEncoder(nn.Module):
    """Transformer-based sequence encoder"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, num_heads: int = 8, dropout: float = 0.1, 
                 pad_idx: int = 0, max_seq_length: int = 5000):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.pos_encoder = PositionalEncoding(embedding_dim, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final projection to ensure output dimension matches what CLIP expects
        self.fc = nn.Linear(embedding_dim, embedding_dim)
        self.pad_idx = pad_idx

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # Create padding mask
        batch_size, seq_len = x.shape
        mask = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len) >= lengths.unsqueeze(1)
        
        # Embedding and positional encoding
        x = self.embedding(x) * math.sqrt(self.embedding.size(-1))
        x = x.transpose(0, 1)  # (batch, seq, embed) -> (seq, batch, embed)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        x = x.transpose(0, 1)  # Back to (batch, seq, embed)
        
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=mask)
        
        # Global average pooling over sequence dimension (excluding padding)
        # Create a mask for non-padded positions
        lengths_expanded = lengths.unsqueeze(1).unsqueeze(2).float()
        x_sum = x.sum(dim=1)
        x_mean = x_sum / lengths_expanded.clamp(min=1)
        
        # Project and normalize
        x_mean = self.fc(x_mean)
        return F.normalize(x_mean, p=2, dim=1)


class LSTMEncoder(nn.Module):
    """Original LSTM-based sequence encoder"""
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, 
                 num_layers: int, dropout: float = 0.1, pad_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim) 
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        lengths_cpu = lengths.cpu().clamp(min=1)
        embedded = self.dropout_layer(self.embedding(x))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths_cpu, batch_first=True, enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed_embedded)
        hidden_fwd = hidden[-2,:,:] 
        hidden_bwd = hidden[-1,:,:]
        hidden_combined = torch.cat((hidden_fwd, hidden_bwd), dim=1)
        projected = self.fc(hidden_combined) 
        return F.normalize(projected, p=2, dim=1)


class DNAClipModel(nn.Module):
    def __init__(self, dna_vocab_size: int, text_vocab_size: int,
                 embedding_dim: int = DEFAULT_EMBEDDING_DIM, hidden_dim: int = DEFAULT_HIDDEN_DIM, 
                 num_layers: int = DEFAULT_NUM_LAYERS, dropout: float = 0.1,
                 initial_temperature: float = DEFAULT_TEMPERATURE,
                 dna_pad_idx: int = 0, text_pad_idx: int = 0,
                 encoder_type: str = "lstm", num_heads: int = 8, max_seq_length: int = 5000):
        super().__init__()
        
        # Select encoder type
        if encoder_type == "transformer":
            self.dna_encoder = TransformerEncoder(
                dna_vocab_size, embedding_dim, hidden_dim, num_layers, 
                num_heads, dropout, dna_pad_idx, max_seq_length
            )
            self.text_encoder = TransformerEncoder(
                text_vocab_size, embedding_dim, hidden_dim, num_layers, 
                num_heads, dropout, text_pad_idx, max_seq_length
            )
        else:  # Default to LSTM
            self.dna_encoder = LSTMEncoder(
                dna_vocab_size, embedding_dim, hidden_dim, num_layers, 
                dropout, dna_pad_idx
            )
            self.text_encoder = LSTMEncoder(
                text_vocab_size, embedding_dim, hidden_dim, num_layers, 
                dropout, text_pad_idx
            )
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / initial_temperature))
        self.encoder_type = encoder_type

    def encode_dna(self, dna_tokens: torch.Tensor, dna_lengths: torch.Tensor) -> torch.Tensor:
        return self.dna_encoder(dna_tokens, dna_lengths)

    def encode_text(self, text_tokens: torch.Tensor, text_lengths: torch.Tensor) -> torch.Tensor:
        return self.text_encoder(text_tokens, text_lengths)

    def forward(self, dna_tokens: torch.Tensor, dna_lengths: torch.Tensor,
                text_tokens: torch.Tensor, text_lengths: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dna_embeds = self.encode_dna(dna_tokens, dna_lengths)
        text_embeds = self.encode_text(text_tokens, text_lengths)
        logit_scale_clamped = torch.clamp(self.logit_scale, max=math.log(100.)) 
        logits_per_dna = (dna_embeds @ text_embeds.t()) * logit_scale_clamped.exp()
        logits_per_text = logits_per_dna.t() 
        return logits_per_dna, logits_per_text, logit_scale_clamped.exp()


# --- Data Handling ---
class DNADataset(Dataset):
    def __init__(self, sequences: List[str], annotations: List[str],
                 dna_tokenizer: BaseTokenizer, text_tokenizer: TextTokenizer,
                 max_dna_len: Optional[int], max_text_len: Optional[int], 
                 metadata: Optional[List[Dict[str, Any]]] = None):
        self.sequences = sequences
        self.annotations = annotations
        self.dna_tokenizer = dna_tokenizer
        self.text_tokenizer = text_tokenizer
        self.max_dna_len = max_dna_len if max_dna_len is not None and max_dna_len > 0 else None
        self.max_text_len = max_text_len if max_text_len is not None and max_text_len > 0 else None
        self.metadata = metadata if metadata else [{} for _ in range(len(sequences))]
        
        if not (len(sequences) == len(annotations) == len(self.metadata)):
            raise ValueError("Sequences, annotations, and metadata must have the same length.")

    def __len__(self) -> int: 
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        seq, ann, meta = self.sequences[idx], self.annotations[idx], self.metadata[idx]

        dna_encoded_ids, dna_effective_len = self.dna_tokenizer.encode(seq, self.max_dna_len)
        text_encoded_ids, text_effective_len = self.text_tokenizer.encode(ann, self.max_text_len)
        
        dna_effective_len = max(1, dna_effective_len)
        text_effective_len = max(1, text_effective_len)

        if not dna_encoded_ids:
             dna_encoded_ids = [self.dna_tokenizer.pad_token_id] * (self.max_dna_len if self.max_dna_len else 1)
        if not text_encoded_ids:
             text_encoded_ids = [self.text_tokenizer.pad_token_id] * (self.max_text_len if self.max_text_len else 1)

        return {
            "dna_tokens": torch.tensor(dna_encoded_ids, dtype=torch.long), 
            "dna_lengths": torch.tensor(dna_effective_len, dtype=torch.long),
            "text_tokens": torch.tensor(text_encoded_ids, dtype=torch.long), 
            "text_lengths": torch.tensor(text_effective_len, dtype=torch.long),
            "raw_sequence": seq, "raw_sequence_len": len(seq), 
            "raw_annotation": ann, "raw_annotation_len": len(ann.split()), # Approx words
            "metadata": meta
        }

def custom_collate_fn(batch: List[Dict[str, Any]], dna_pad_id: int, text_pad_id: int) -> Dict[str, Any]:
    if not batch: return {}

    keys_to_pad = ['dna_tokens', 'text_tokens']
    pad_ids = {'dna_tokens': float(dna_pad_id), 'text_tokens': float(text_pad_id)}
    
    collated_batch = {}
    for key in keys_to_pad:
        tokens_list = [item[key] for item in batch]
        if all(t.size(0) == tokens_list[0].size(0) for t in tokens_list): # Already same length (e.g. fixed max_len)
            collated_batch[key] = torch.stack(tokens_list)
        else: # Needs padding
            collated_batch[key] = pad_sequence(tokens_list, batch_first=True, padding_value=pad_ids[key])
            
    for key in batch[0].keys():
        if key not in keys_to_pad: # Handle other keys
            if isinstance(batch[0][key], torch.Tensor):
                 collated_batch[key] = torch.stack([item[key] for item in batch])
            elif isinstance(batch[0][key], (int, float)): # For lengths, etc.
                 collated_batch[key] = torch.tensor([item[key] for item in batch], dtype=torch.long if 'length' in key else torch.float)
            else: # Lists of strings (raw_sequence, raw_annotation, metadata)
                 collated_batch[key] = [item[key] for item in batch]
    
    # Ensure lengths are tensors
    if 'dna_lengths' in collated_batch and not isinstance(collated_batch['dna_lengths'], torch.Tensor):
        collated_batch['dna_lengths'] = torch.tensor(collated_batch['dna_lengths'], dtype=torch.long)
    if 'text_lengths' in collated_batch and not isinstance(collated_batch['text_lengths'], torch.Tensor):
        collated_batch['text_lengths'] = torch.tensor(collated_batch['text_lengths'], dtype=torch.long)

    return collated_batch


def create_dataloader(dataset: DNADataset, batch_size: int, num_workers: int, shuffle: bool, 
                      dna_pad_id: int, text_pad_id: int, pin_memory: bool = True, drop_last: bool = False) -> DataLoader:
    collate_wrapper = lambda batch_items: custom_collate_fn(batch_items, dna_pad_id, text_pad_id)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      collate_fn=collate_wrapper, pin_memory=pin_memory, drop_last=drop_last)

FASTA_HEADER_PATTERN = re.compile(r">([^\s|]+)(?:[|\s]*(.*))?") # ID is group 1, rest is group 2

def parse_fasta(filepath: str, use_blacklist: bool = True) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Parses a FASTA file.
    Returns a list of tuples: (id, sequence, metadata_dict).
    Metadata includes 'id', 'original_header', 'species', 'biotype', 'description'.
    
    If use_blacklist is True, filters out sequences with blacklisted biotypes.
    """
    data = []
    open_func = gzip.open if filepath.endswith((".gz", ".fa.gz")) else open
    current_seq_chunks, current_id, current_meta = [], None, {}
    
    skipped_blacklisted = 0
    
    try:
        with open_func(filepath, "rt", errors='ignore') as f: 
            for line in f:
                line = line.strip()
                if not line: continue
                if line.startswith(">"):
                    if current_seq_chunks and current_id: 
                        # Check blacklist before adding
                        if use_blacklist and current_meta.get("biotype", "unknown") in EXCLUDED_BIOTYPES:
                            skipped_blacklisted += 1
                        else:
                            data.append((current_id, "".join(current_seq_chunks), current_meta))
                    
                    # Parse header for chromosample format
                    header = line[1:]  # Remove '>'
                    species, biotype, description = parse_chromosample_header(header)
                    
                    # Also try to extract ID from header
                    header_match = FASTA_HEADER_PATTERN.match(line)
                    if header_match:
                        current_id = header_match.group(1).strip()
                    else:
                        current_id = header.split()[0] if header else f"seq_{len(data)+1}"
                    
                    current_meta = {
                        "id": current_id, 
                        "original_header": header,
                        "species": species,
                        "biotype": biotype,
                        "description": description
                    }
                    
                    current_seq_chunks = []
                else: 
                    current_seq_chunks.append(line.upper()) # Ensure sequence is uppercase

            if current_seq_chunks and current_id: 
                # Check blacklist for last sequence
                if use_blacklist and current_meta.get("biotype", "unknown") in EXCLUDED_BIOTYPES:
                    skipped_blacklisted += 1
                else:
                    data.append((current_id, "".join(current_seq_chunks), current_meta))
                    
    except FileNotFoundError: 
        logger.error(f"FASTA file not found: {filepath}"); raise
    except Exception as e: 
        logger.error(f"Error parsing FASTA file {filepath}: {e}", exc_info=True); raise
    
    if use_blacklist and skipped_blacklisted > 0:
        logger.info(f"Filtered out {skipped_blacklisted} sequences with blacklisted biotypes")
    
    logger.info(f"Parsed {len(data)} sequences from FASTA: {filepath}")
    return data

def parse_bed(filepath: str, reference_fasta_path: str) -> List[Tuple[str, str, Dict[str, Any], str]]:
    """
    Parses a BED file, extracting sequences from a reference FASTA.
    Returns a list of tuples: (name, sequence_extracted, metadata_dict, annotation_text).
    """
    ref_genome_sequences = {}
    ref_genome_metadata = {} # To store metadata from FASTA headers per chromosome
    logger.info(f"Loading reference genome from {reference_fasta_path} for BED processing...")
    try:
        parsed_fasta_ref = parse_fasta(reference_fasta_path)
        for header_id, seq, meta_dict in parsed_fasta_ref: 
            chrom_name = header_id.split()[0] # Use first part of header as chrom name
            ref_genome_sequences[chrom_name] = seq
            ref_genome_metadata[chrom_name] = meta_dict
            # Add variants (e.g., chr1 vs 1)
            if chrom_name.startswith("chr"): ref_genome_sequences[chrom_name[3:]] = seq; ref_genome_metadata[chrom_name[3:]] = meta_dict
            else: ref_genome_sequences[f"chr{chrom_name}"] = seq; ref_genome_metadata[f"chr{chrom_name}"] = meta_dict
        logger.info(f"Loaded {len(ref_genome_sequences)} chromosome/contig entries from reference FASTA for BED.")
    except Exception as e: 
        logger.error(f"Error parsing reference FASTA {reference_fasta_path} for BED: {e}", exc_info=True); raise
    
    data = [] # List of (name, seq_extracted, meta, annotation_text)
    skipped_lines_coord = 0
    skipped_lines_chrom = 0

    try:
        open_func = gzip.open if filepath.endswith(".gz") else open
        with open_func(filepath, "rt", errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith(("#", "track", "browser")): continue
                fields = line.split("\t")
                if len(fields) < 3: 
                    logger.warning(f"Skipping malformed BED line {i+1} (fields<3): {line}"); continue
                
                chrom, start_str, end_str = fields[0], fields[1], fields[2]
                name = fields[3] if len(fields) > 3 and fields[3] != '.' else f"region_{chrom}_{start_str}_{end_str}"
                
                try: start, end = int(start_str), int(end_str)
                except ValueError: 
                    logger.warning(f"Skipping BED line {i+1} (invalid coords): {line}"); continue

                resolved_chrom = next((c for c in [chrom, f"chr{chrom}"] + ([chrom[3:]] if chrom.startswith("chr") else []) if c in ref_genome_sequences), None)
                
                if not resolved_chrom: 
                    skipped_lines_chrom +=1
                    if skipped_lines_chrom <= 10 or skipped_lines_chrom % 1000 == 0: # Log first few and then periodically
                         logger.warning(f"Chromosome '{chrom}' (BED line {i+1}) not found in reference genome. Skipping. (Total such skips: {skipped_lines_chrom})"); 
                    continue
                
                chrom_seq = ref_genome_sequences[resolved_chrom]
                if not (0 <= start < end <= len(chrom_seq)): 
                    skipped_lines_coord +=1
                    if skipped_lines_coord <=10 or skipped_lines_coord % 1000 == 0:
                         logger.warning(f"Skipping BED line {i+1} (invalid range for chr '{resolved_chrom}'): {start}-{end}, ref len {len(chrom_seq)}. (Total such skips: {skipped_lines_coord})"); 
                    continue
                
                seq_extracted = chrom_seq[start:end].upper()
                strand = fields[5] if len(fields) > 5 else '+'
                if strand == '-':
                    seq_extracted = reverse_complement(seq_extracted)

                meta = {"id": name, "chr": resolved_chrom, "start": start, "end": end, "strand": strand, "source_format": "bed"}
                meta["species"] = ref_genome_metadata.get(resolved_chrom, {}).get('species', 'unknown_species_bed_ref')
                
                attributes_str = ""
                if len(fields) > 8 and fields[8] and fields[8] != '.': # Optional 9th field (attributes)
                    attributes_str = fields[8]
                elif len(fields) > 3 and fields[3] and fields[3] != '.' and any(c in fields[3] for c in "=;"): # Check if name field contains attributes
                     attributes_str = fields[3] # Use name field if it looks like attributes

                parsed_desc_meta = {}
                if attributes_str:
                    parsed_desc_meta = parse_gff_attributes(attributes_str) # Re-use GFF attribute parser

                meta["biotype"] = parsed_desc_meta.get("biotype", parsed_desc_meta.get("gene_biotype", ref_genome_metadata.get(resolved_chrom, {}).get('biotype', 'unknown_bed_feature')))
                meta.update(parsed_desc_meta) # Add all parsed attributes to meta
                meta["raw_bed_fields"] = "\t".join(fields[3:]) if len(fields) > 3 else ""
                
                annotation_text_parts = [meta['biotype'], name, f"from {resolved_chrom}:{start}-{end}"]
                if 'description' in meta: annotation_text_parts.append(meta['description'])
                elif meta["raw_bed_fields"] and meta["raw_bed_fields"] != name : 
                    annotation_text_parts.append(f"Attributes: {meta['raw_bed_fields']}")
                
                annotation_text = ", ".join(filter(None, annotation_text_parts)).strip()
                annotation_text = re.sub(r'\s+', ' ', annotation_text) # Clean up multiple spaces
                
                data.append((name, seq_extracted, meta, annotation_text))
    except FileNotFoundError: 
        logger.error(f"BED file not found: {filepath}"); raise
    except Exception as e: 
        logger.error(f"Error parsing BED file {filepath}: {e}", exc_info=True); raise
    
    if skipped_lines_chrom > 0: logger.warning(f"Total BED lines skipped due to chromosome not in reference: {skipped_lines_chrom}")
    if skipped_lines_coord > 0: logger.warning(f"Total BED lines skipped due to invalid coordinates for reference: {skipped_lines_coord}")
    logger.info(f"Parsed {len(data)} regions from BED file: {filepath}")
    return data

def parse_gff3(filepath: str, reference_fasta_path: str, 
               target_feature_types: List[str]) -> List[Tuple[str, str, Dict[str, Any], str]]:
    """
    Parses a GFF3 file, extracting sequences for specified feature types from a reference FASTA.
    Returns a list of tuples: (id, sequence_extracted, metadata_dict, annotation_text).
    """
    ref_genome_sequences = {}
    ref_genome_metadata = {}
    logger.info(f"Loading reference genome from {reference_fasta_path} for GFF3 processing...")
    try:
        parsed_fasta_ref = parse_fasta(reference_fasta_path)
        for header_id, seq, meta_dict in parsed_fasta_ref:
            chrom_name = header_id.split()[0]
            ref_genome_sequences[chrom_name] = seq
            ref_genome_metadata[chrom_name] = meta_dict
            # Add variants for chromosome naming (e.g. '1' vs 'chr1')
            if chrom_name.startswith("chr") and chrom_name[3:] not in ref_genome_sequences : 
                ref_genome_sequences[chrom_name[3:]] = seq
                ref_genome_metadata[chrom_name[3:]] = meta_dict
            elif not chrom_name.startswith("chr") and f"chr{chrom_name}" not in ref_genome_sequences:
                ref_genome_sequences[f"chr{chrom_name}"] = seq
                ref_genome_metadata[f"chr{chrom_name}"] = meta_dict

        logger.info(f"Loaded {len(ref_genome_sequences)} chromosome/contig entries from reference FASTA for GFF3.")
    except Exception as e:
        logger.error(f"Error parsing reference FASTA {reference_fasta_path} for GFF3: {e}", exc_info=True)
        raise

    data = []
    gff_feature_type_counts = Counter()
    processed_count = 0
    skipped_lines_target_type = 0
    skipped_lines_coord = 0
    skipped_lines_chrom = 0

    open_func = gzip.open if filepath.endswith((".gz", ".gff3.gz", ".gff.gz")) else open
    try:
        with open_func(filepath, "rt", errors='ignore') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                fields = line.split("\t")
                if len(fields) != 9:
                    logger.warning(f"Skipping malformed GFF3 line {i+1} (fields != 9): {line}")
                    continue

                chrom, source, feat_type, start_str, end_str, score, strand, phase, attributes_str = fields
                gff_feature_type_counts[feat_type] += 1

                if target_feature_types and feat_type not in target_feature_types:
                    skipped_lines_target_type += 1
                    continue
                
                try:
                    start = int(start_str) - 1 # GFF is 1-based, Python is 0-based start
                    end = int(end_str)       # GFF is 1-based, Python is 0-based exclusive end
                except ValueError:
                    logger.warning(f"Skipping GFF3 line {i+1} (invalid coords {start_str}-{end_str}): {line}")
                    continue

                resolved_chrom = chrom
                if chrom not in ref_genome_sequences: # Try common variations like 'chr1' vs '1'
                    alt_chrom = f"chr{chrom}" if not chrom.startswith("chr") else chrom[3:]
                    if alt_chrom in ref_genome_sequences:
                        resolved_chrom = alt_chrom
                    else:
                        skipped_lines_chrom +=1
                        if skipped_lines_chrom <= 10 or skipped_lines_chrom % 1000 == 0:
                            logger.warning(f"Chromosome '{chrom}' (GFF3 line {i+1}) not found in reference genome (tried '{alt_chrom}'). Skipping. (Total such skips: {skipped_lines_chrom})")
                        continue
                
                chrom_seq = ref_genome_sequences[resolved_chrom]
                if not (0 <= start < end <= len(chrom_seq)):
                    skipped_lines_coord += 1
                    if skipped_lines_coord <= 10 or skipped_lines_coord % 1000 == 0:
                         logger.warning(f"Skipping GFF3 line {i+1} (invalid range for chr '{resolved_chrom}'): {start+1}-{end}, ref len {len(chrom_seq)}. (Total such skips: {skipped_lines_coord})")
                    continue
                
                seq_extracted = chrom_seq[start:end].upper()
                if strand == '-':
                    seq_extracted = reverse_complement(seq_extracted)

                attributes = parse_gff_attributes(attributes_str)
                
                # Construct ID: prefer ID, then Name, then a generated one
                gff_id = attributes.get('ID', attributes.get('Name', f"{feat_type}_{resolved_chrom}_{start+1}_{end}"))
                
                # Construct biotype: prefer 'biotype', then 'gene_biotype', then GFF 'type', then ref FASTA biotype
                biotype = attributes.get('biotype', attributes.get('gene_biotype', attributes.get('transcript_biotype', feat_type)))
                
                # Species from reference FASTA metadata if available
                species = ref_genome_metadata.get(resolved_chrom, {}).get('species', 'unknown_gff_species')

                meta = {
                    "id": gff_id, "chr": resolved_chrom, "start": start + 1, "end": end, # Store original 1-based for meta
                    "strand": strand, "source_format": "gff3", "gff_feature_type": feat_type,
                    "biotype": biotype, "species": species,
                    "gff_source_column": source, "gff_score": score, "gff_phase": phase,
                    "gff_attributes_dict": attributes # Store all parsed attributes
                }

                # Construct annotation text
                ann_parts = [meta['biotype']]
                if 'Name' in attributes: ann_parts.append(attributes['Name'])
                elif meta['id'] != meta['biotype']: ann_parts.append(meta['id']) # Add ID if different from biotype and no Name
                
                ann_parts.append(f"from {species} chromosome {resolved_chrom}:{meta['start']}-{meta['end']}({strand})")

                if 'description' in attributes: ann_parts.append(f"Description: {attributes['description']}")
                if 'product' in attributes: ann_parts.append(f"Product: {attributes['product']}")
                if 'gene' in attributes: ann_parts.append(f"Gene: {attributes['gene']}")
                
                # Add a few other common/useful attributes if present
                other_attrs_to_include = ['gene_id', 'transcript_id', 'note']
                for attr_key in other_attrs_to_include:
                    if attr_key in attributes and attributes[attr_key]:
                        ann_parts.append(f"{attr_key.replace('_',' ').title()}: {attributes[attr_key]}")

                annotation_text = ". ".join(filter(None, ann_parts)).strip()
                annotation_text = re.sub(r'\s+', ' ', annotation_text) # Clean up multiple spaces
                annotation_text = re.sub(r'\.\.+', '.', annotation_text) # Clean up multiple periods

                data.append((gff_id, seq_extracted, meta, annotation_text))
                processed_count += 1
                if processed_count % 10000 == 0:
                    logger.info(f"Processed {processed_count} target GFF3 features...")

    except FileNotFoundError:
        logger.error(f"GFF3 file not found: {filepath}"); raise
    except Exception as e:
        logger.error(f"Error parsing GFF3 file {filepath}: {e}", exc_info=True); raise

    logger.info(f"Finished parsing GFF3 file: {filepath}")
    logger.info(f"Total GFF3 lines processed for target types: {processed_count}")
    if skipped_lines_target_type > 0: logger.info(f"GFF3 lines skipped because feature type not in target list: {skipped_lines_target_type}")
    if skipped_lines_chrom > 0: logger.warning(f"Total GFF3 lines skipped due to chromosome not in reference: {skipped_lines_chrom}")
    if skipped_lines_coord > 0: logger.warning(f"Total GFF3 lines skipped due to invalid coordinates for reference: {skipped_lines_coord}")
    
    # Log GFF feature type counts summary
    logger.info("GFF3 Feature Type Counts (all types in file):")
    for f_type, count in gff_feature_type_counts.most_common(10): # Log top 10
        logger.info(f"  {f_type}: {count}")
    if len(gff_feature_type_counts) > 10:
        logger.info(f"  ... and {len(gff_feature_type_counts) - 10} other feature types.")
    
    # Store overall GFF feature counts for reporting
    if 'dataset_summary_temp_storage' not in globals(): # Create if not exists
        global dataset_summary_temp_storage
        dataset_summary_temp_storage = {}
    dataset_summary_temp_storage['gff_feature_type_counts_raw'] = dict(gff_feature_type_counts)

    return data


# --- Training and Evaluation --- 
def contrastive_loss(logits_per_source: torch.Tensor, logits_per_target: torch.Tensor, device: torch.device) -> torch.Tensor:
    batch_size = logits_per_source.shape[0]
    if batch_size == 0: return torch.tensor(0.0, device=device, requires_grad=True)
    labels = torch.arange(batch_size, device=device, dtype=torch.long)
    loss_source = F.cross_entropy(logits_per_source, labels)
    loss_target = F.cross_entropy(logits_per_target, labels)
    return (loss_source + loss_target) / 2.0

def calculate_mrr(similarity_matrix: np.ndarray) -> Tuple[float, float]:
    n_samples = similarity_matrix.shape[0]
    if n_samples == 0: return 0.0, 0.0
    mrr_d2t, mrr_t2d = 0.0, 0.0
    sorted_indices_dna_to_text = np.argsort(-similarity_matrix, axis=1)
    for i in range(n_samples):
        rank_list = np.where(sorted_indices_dna_to_text[i] == i)[0]
        if rank_list.size > 0: mrr_d2t += 1.0 / (rank_list[0] + 1)
    sorted_indices_text_to_dna = np.argsort(-similarity_matrix.T, axis=1)
    for i in range(n_samples):
        rank_list = np.where(sorted_indices_text_to_dna[i] == i)[0]
        if rank_list.size > 0: mrr_t2d += 1.0 / (rank_list[0] + 1)
    return (mrr_d2t / n_samples if n_samples > 0 else 0.0), \
           (mrr_t2d / n_samples if n_samples > 0 else 0.0)

def train_epoch(model: DNAClipModel, dataloader: DataLoader, optimizer: optim.Optimizer, device: torch.device, 
                epoch_num: int, num_epochs: int, use_amp: bool = False, scaler: Optional[torch.cuda.amp.GradScaler] = None) -> Tuple[float, float, float]:
    model.train()
    total_loss, total_temp_val, total_grad_norm = 0.0, 0.0, 0.0
    batches_processed = 0
    num_total_batches = len(dataloader)
    if num_total_batches == 0: return 0.0, model.logit_scale.exp().item(), 0.0
    
    epoch_start_time = time.time()

    for i, batch in enumerate(dataloader):
        dna_tokens = batch["dna_tokens"].to(device, non_blocking=True)
        dna_lengths = batch["dna_lengths"] # Stays on CPU for pack_padded_sequence
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        text_lengths = batch["text_lengths"] # Stays on CPU for pack_padded_sequence
        
        batches_processed += 1
        optimizer.zero_grad(set_to_none=True)
        current_grad_norm = 0.0
        
        autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp) if device.type == 'cuda' and use_amp else contextlib.nullcontext()
        
        with autocast_ctx:
            logits_dna, logits_text, temp = model(dna_tokens, dna_lengths, text_tokens, text_lengths)
            loss = contrastive_loss(logits_dna, logits_text, device)
        
        if torch.isnan(loss) or torch.isinf(loss):
            logger.warning(f"NaN/Inf loss encountered at Epoch {epoch_num+1}, Batch {i+1}. Skipping update for this batch.")
        else:
            if use_amp and device.type == 'cuda' and scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer) # Unscale before clipping
                current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                scaler.step(optimizer)
                scaler.update()
            else: 
                loss.backward()
                current_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
                optimizer.step()

            total_loss += loss.item()
            total_temp_val += temp.item()
            if not (np.isnan(current_grad_norm) or np.isinf(current_grad_norm)):
                total_grad_norm += current_grad_norm
        
        if (i + 1) % (max(1, num_total_batches // 10)) == 0 or i == num_total_batches -1 : 
            current_loss_val = loss.item() if not (torch.isnan(loss) or torch.isinf(loss)) else float('nan')
            logger.info(f"Epoch [{epoch_num+1}/{num_epochs}], Batch [{i+1}/{num_total_batches}], Loss: {current_loss_val:.4f}, Temp_eff: {temp.item():.4f}, GradNorm: {current_grad_norm:.4f}")

    epoch_duration = time.time() - epoch_start_time
    logger.info(f"Train epoch {epoch_num+1} completed in {epoch_duration:.2f}s.")
    if batches_processed == 0: return 0.0, model.logit_scale.exp().item(), 0.0
    return total_loss/batches_processed, total_temp_val/batches_processed, total_grad_norm/batches_processed

@torch.no_grad()
def evaluate(model: DNAClipModel, dataloader: DataLoader, device: torch.device, report_dir: str, epoch: Optional[int], metrics_agg: Dict, is_final_eval: bool = False) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray], List[str], List[str], List[str], List[str]]:
    model.eval()
    all_dna_embeds_list, all_text_embeds_list = [], []
    all_biotypes_eval, all_species_eval, raw_seqs_eval, raw_anns_eval = [], [], [], []
    total_loss = 0.0; batches_processed = 0
    num_total_batches = len(dataloader)

    if num_total_batches == 0: 
        logger.warning("Evaluation dataloader is empty.")
        return {}, None, None, [], [], [], []

    epoch_str = f"epoch {epoch+1}" if epoch is not None else "Final Evaluation"
    logger.info(f"Starting evaluation for {epoch_str}...")
    eval_start_time = time.time()
    
    # Check if autocast was enabled during training for consistency (though not strictly necessary for eval)
    autocast_enabled_for_eval = device.type == 'cuda' and torch.is_autocast_enabled() # Check if it was enabled outside

    for i, batch in enumerate(dataloader):
        dna_tokens = batch["dna_tokens"].to(device, non_blocking=True)
        dna_lengths = batch["dna_lengths"] 
        text_tokens = batch["text_tokens"].to(device, non_blocking=True)
        text_lengths = batch["text_lengths"] 
        batches_processed +=1

        autocast_ctx_eval = torch.cuda.amp.autocast(enabled=True) if autocast_enabled_for_eval else contextlib.nullcontext()

        with autocast_ctx_eval: # Use AMP for eval if it was used in training for consistency
            logits_dna, logits_text, _ = model(dna_tokens, dna_lengths, text_tokens, text_lengths)
            loss = contrastive_loss(logits_dna, logits_text, device)
            # Get embeddings for other metrics
            dna_embeds = model.encode_dna(dna_tokens, dna_lengths)
            text_embeds = model.encode_text(text_tokens, text_lengths)

        if not (torch.isnan(loss) or torch.isinf(loss)):
            total_loss += loss.item()
        else:
            logger.warning(f"NaN/Inf loss encountered in evaluation batch {i+1} for {epoch_str}.")

        all_dna_embeds_list.append(dna_embeds.cpu().numpy(force=True)) # force=True if from autocast non-blocking
        all_text_embeds_list.append(text_embeds.cpu().numpy(force=True))
        
        # Ensure metadata is a list of dicts
        current_batch_metadata = batch["metadata"]
        if isinstance(current_batch_metadata, dict) and not any(isinstance(v, list) for v in current_batch_metadata.values()):
             # If metadata wasn't properly batched as a list of dicts, try to reconstruct
             try:
                 num_items_in_batch = dna_tokens.size(0)
                 reconstructed_meta = [{} for _ in range(num_items_in_batch)]
                 for k_meta, v_meta_list in current_batch_metadata.items():
                     if isinstance(v_meta_list, list) and len(v_meta_list) == num_items_in_batch:
                         for item_idx in range(num_items_in_batch):
                             reconstructed_meta[item_idx][k_meta] = v_meta_list[item_idx]
                 current_batch_metadata = reconstructed_meta
             except Exception as e_meta_recon:
                 logger.error(f"Error reconstructing metadata in evaluation: {e_meta_recon}. Metadata might be incomplete.")
                 current_batch_metadata = [{} for _ in range(dna_tokens.size(0))] # Fallback
        
        all_biotypes_eval.extend([m.get("biotype", "unknown") for m in current_batch_metadata])
        all_species_eval.extend([m.get("species", "unknown") for m in current_batch_metadata])

        if is_final_eval: # Only collect raw texts for final detailed report to save memory
            raw_seqs_eval.extend(batch["raw_sequence"])
            raw_anns_eval.extend(batch["raw_annotation"])

        if (i + 1) % (max(1, num_total_batches // 5)) == 0 or i == num_total_batches -1:
            logger.info(f"Evaluation batch [{i+1}/{num_total_batches}] for {epoch_str} processed.")

    if batches_processed == 0: 
        logger.warning(f"No data processed during evaluation for {epoch_str}.")
        return {}, None, None, [], [], [], []

    avg_loss = total_loss / batches_processed
    metrics_agg['eval_loss'].append(avg_loss)
    logger.info(f"Evaluation for {epoch_str} - Average Loss: {avg_loss:.4f}")

    all_dna_embeds = np.concatenate(all_dna_embeds_list, axis=0)
    all_text_embeds = np.concatenate(all_text_embeds_list, axis=0)
    
    current_metrics = {"avg_loss": avg_loss}

    # Similarity and Retrieval Metrics
    similarity_matrix = all_dna_embeds @ all_text_embeds.T
    num_samples_eval = similarity_matrix.shape[0]

    mrr_d2t, mrr_t2d = calculate_mrr(similarity_matrix)
    current_metrics.update({"mrr_dna_to_text": mrr_d2t, "mrr_text_to_dna": mrr_t2d})
    metrics_agg['mrr_dna_to_text'].append(mrr_d2t)
    metrics_agg['mrr_text_to_dna'].append(mrr_t2d)
    logger.info(f"MRR DNA->Text: {mrr_d2t:.3f}, Text->DNA: {mrr_t2d:.3f} for {epoch_str}")

    recall_ks = [1, 5, 10] # Common recall values
    recalls_dna_to_text, recalls_text_to_dna = {}, {}
    if num_samples_eval > 0 :
        valid_recall_ks = [k for k in recall_ks if k <= num_samples_eval]
        valid_recall_ks = valid_recall_ks if valid_recall_ks else ([1] if num_samples_eval >= 1 else [])

        top_k_indices_dna = np.argsort(-similarity_matrix, axis=1)
        for k_val in valid_recall_ks:
            recalls_dna_to_text[f"dna_to_text_R@{k_val}"] = np.mean([i in top_k_indices_dna[i, :k_val] for i in range(num_samples_eval)])
        
        top_k_indices_text = np.argsort(-similarity_matrix.T, axis=1) # Transpose for text-to-dna
        for k_val in valid_recall_ks:
            recalls_text_to_dna[f"text_to_dna_R@{k_val}"] = np.mean([i in top_k_indices_text[i, :k_val] for i in range(num_samples_eval)])
    
    for k_met, v_met in recalls_dna_to_text.items(): metrics_agg[k_met].append(v_met)
    for k_met, v_met in recalls_text_to_dna.items(): metrics_agg[k_met].append(v_met)
    current_metrics.update(recalls_dna_to_text)
    current_metrics.update(recalls_text_to_dna)
    logger.info(f"Retrieval Recalls ({epoch_str}): {{**recalls_dna_to_text, **recalls_text_to_dna}}")

    # k-NN Classification on Biotypes (if enough diverse data)
    unique_biotypes_for_knn = sorted(list(set(all_biotypes_eval)))
    if len(unique_biotypes_for_knn) >= 2 and all_dna_embeds.shape[0] > len(unique_biotypes_for_knn):
        try:
            # Ensure enough samples per class for stratified split
            biotype_counts_for_knn = Counter(all_biotypes_eval)
            # Filter out classes with only 1 sample for stratification
            valid_indices_knn = [i for i, bio in enumerate(all_biotypes_eval) if biotype_counts_for_knn[bio] >= 2] # Need at least 2 for split
            
            if len(valid_indices_knn) >= max(10, 2 * len(unique_biotypes_for_knn) + 2) : # Heuristic: need enough data overall
                knn_X_embeds = all_dna_embeds[valid_indices_knn]
                knn_Y_labels = [all_biotypes_eval[i] for i in valid_indices_knn]
                
                label_counts_for_split = Counter(knn_Y_labels)
                # Check again if after filtering, we still have >1 class with >=2 samples each for stratification
                if all(c >= 2 for c in label_counts_for_split.values()) and len(label_counts_for_split) >= 2:
                    X_train_knn, X_test_knn, y_train_knn, y_test_knn = train_test_split(
                        knn_X_embeds, knn_Y_labels, test_size=0.3, stratify=knn_Y_labels, random_state=42
                    )
                    if len(X_train_knn)>0 and len(X_test_knn)>0 and len(set(y_train_knn))>1: # Ensure train/test are usable
                        n_neighbors_knn = min(5, len(X_train_knn), min(Counter(y_train_knn).values())) # Adapt k
                        n_neighbors_knn = max(1, n_neighbors_knn) # k must be at least 1

                        knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors_knn).fit(X_train_knn, y_train_knn)
                        y_pred_knn = knn_classifier.predict(X_test_knn)
                        
                        report_labels = sorted(list(set(y_test_knn) | set(y_pred_knn))) # All labels in test or pred

                        clf_report_dict = sk_classification_report(y_test_knn, y_pred_knn, output_dict=True, zero_division=0, labels=report_labels, target_names=report_labels)
                        if "accuracy" not in clf_report_dict: clf_report_dict["accuracy"] = accuracy_score(y_test_knn,y_pred_knn) # Ensure accuracy is there

                        current_metrics.update({
                            "knn_biotype_accuracy": clf_report_dict["accuracy"],
                            "knn_biotype_f1_macro": clf_report_dict["macro avg"]["f1-score"]
                        })
                        metrics_agg['knn_biotype_accuracy'].append(current_metrics["knn_biotype_accuracy"])
                        metrics_agg['knn_biotype_f1_macro'].append(current_metrics["knn_biotype_f1_macro"])
                        metrics_agg['knn_classification_reports_biotype'].append(clf_report_dict)
                        logger.info(f"k-NN Biotype Classification ({epoch_str}): Accuracy {current_metrics['knn_biotype_accuracy']:.3f}, F1-macro {current_metrics['knn_biotype_f1_macro']:.3f}")

                        # Plot Confusion Matrix for specific evaluations (final, best, or last epoch)
                        plot_cm_condition = is_final_eval or \
                                            (epoch is not None and metrics_agg.get('best_epoch_idx', -1) == epoch) or \
                                            (epoch is not None and epoch == metrics_agg.get('num_epochs_trained', 0) - 1)

                        if plot_cm_condition:
                            cm = confusion_matrix(y_test_knn, y_pred_knn, labels=report_labels)
                            plt.figure(figsize=(max(8, len(report_labels) * 0.6), max(6, len(report_labels) * 0.5)))
                            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=report_labels, yticklabels=report_labels)
                            plt.xlabel("Predicted Labels"); plt.ylabel("True Labels")
                            epoch_tag_cm = f"epoch_{epoch+1}" if epoch is not None else "final_model"
                            plt.title(f"k-NN Biotype Confusion Matrix ({epoch_tag_cm.replace('_',' ').title()})")
                            plt.tight_layout()
                            cm_filename = f"knn_biotype_confusion_matrix_{epoch_tag_cm}.png"
                            cm_filepath = os.path.join(report_dir, cm_filename)
                            try: 
                                plt.savefig(cm_filepath); plt.close()
                                if 'plot_paths' not in metrics_agg: metrics_agg['plot_paths'] = defaultdict(list)
                                # Avoid duplicate filenames if called multiple times for same epoch (e.g. final eval = best epoch)
                                if not metrics_agg['plot_paths']['knn_biotype_cm'] or metrics_agg['plot_paths']['knn_biotype_cm'][-1] != cm_filename:
                                    metrics_agg['plot_paths']['knn_biotype_cm'].append(cm_filename)
                            except Exception as e_cm_save:
                                logger.error(f"Failed to save k-NN CM plot {cm_filepath}: {e_cm_save}"); plt.close()
                else:
                    logger.info(f"Not enough data or classes with sufficient samples for stratified k-NN Biotype Classification for {epoch_str}.")
            else:
                logger.info(f"Not enough valid samples for k-NN Biotype Classification for {epoch_str}.")
        except ImportError:
            logger.warning("scikit-learn (for KNeighborsClassifier) not available. Skipping k-NN Biotype Classification.")
        except Exception as e_knn:
            logger.error(f"Error during k-NN Biotype Classification for {epoch_str}: {e_knn}", exc_info=True)
    
    # Clustering Quality Metrics (Silhouette, Davies-Bouldin)
    for label_type_cluster, labels_list_cluster in [("biotype", all_biotypes_eval), ("species", all_species_eval)]:
        unique_labels_for_clustering = sorted(list(set(labels_list_cluster)))
        if len(unique_labels_for_clustering) >= 2 and all_dna_embeds.shape[0] > len(unique_labels_for_clustering) : # Need at least 2 clusters and more samples than clusters
            try:
                silhouette = silhouette_score(all_dna_embeds, labels_list_cluster)
                davies_bouldin = davies_bouldin_score(all_dna_embeds, labels_list_cluster)
                current_metrics[f"silhouette_score_{label_type_cluster}"] = silhouette
                current_metrics[f"davies_bouldin_index_{label_type_cluster}"] = davies_bouldin
                metrics_agg[f"silhouette_score_{label_type_cluster}"].append(silhouette)
                metrics_agg[f"davies_bouldin_index_{label_type_cluster}"].append(davies_bouldin)
                logger.info(f"Silhouette Score ({label_type_cluster}, {epoch_str}): {silhouette:.3f}")
                logger.info(f"Davies-Bouldin Index ({label_type_cluster}, {epoch_str}): {davies_bouldin:.3f}")
            except ValueError as e_cluster_metric: # Typically "Number of labels is N but should be between 2 and n_samples - 1"
                logger.warning(f"Could not compute clustering metrics for {label_type_cluster} ({epoch_str}): {e_cluster_metric}")
            except Exception as e_gen_cluster:
                 logger.error(f"Unexpected error computing clustering metrics for {label_type_cluster} ({epoch_str}): {e_gen_cluster}", exc_info=True)
        else:
            logger.info(f"Not enough distinct labels or samples for clustering metrics on {label_type_cluster} for {epoch_str}.")

    eval_duration = time.time() - eval_start_time
    logger.info(f"Evaluation for {epoch_str} complete in {eval_duration:.2f}s.")
    return current_metrics, all_dna_embeds, all_text_embeds, all_biotypes_eval, all_species_eval, raw_seqs_eval, raw_anns_eval

# --- Plotting and Reporting --- 
def plot_embeddings(embeddings: np.ndarray, labels: List[str], title: str, filepath: str, method: str = "pca"):
    n_samples = embeddings.shape[0]
    if n_samples < 2: 
        logger.warning(f"Not enough samples ({n_samples}) for embedding plot: {title}. Skipping.")
        return False
    
    # If method is umap/tsne but samples are too few, default to PCA
    if n_samples < 5 and method in ["tsne", "umap"]:
        logger.info(f"Number of samples ({n_samples}) is too small for {method.upper()}, defaulting to PCA for plot: {title}.")
        method = "pca"
    
    if method == "umap" and not umap:
        logger.info(f"UMAP module not installed, defaulting to PCA for plot: {title}.")
        method = "pca"

    logger.info(f"Generating embedding plot: {title} using {method.upper()}...")
    embeddings_2d = None
    try:
        if embeddings.shape[1] < 2: # Input embeddings are 1D
            if embeddings.shape[1] == 1:
                 embeddings_2d = np.hstack((embeddings, np.zeros_like(embeddings))) # Make it 2D by adding zeros
            else: # embeddings.shape[1] == 0, should not happen with valid model
                logger.warning(f"Embeddings have dimension {embeddings.shape[1]} for {title}, cannot plot.")
                return False
        elif method == "pca": 
            # Ensure n_components is valid
            pca_n_components = min(2, embeddings.shape[0], embeddings.shape[1])
            if pca_n_components < 2 and pca_n_components ==1: # Can only do 1D PCA
                 pca_result = PCA(n_components=1, random_state=42, svd_solver='full').fit_transform(embeddings)
                 embeddings_2d = np.hstack((pca_result, np.zeros_like(pca_result))) # Make 2D
            elif pca_n_components < 1:
                 logger.warning(f"Cannot perform PCA with {pca_n_components} components for {title}. Skipping plot.")
                 return False
            else: # Standard 2D PCA
                embeddings_2d = PCA(n_components=2, random_state=42, svd_solver='full').fit_transform(embeddings)
        elif method == "tsne":
            # Adjust perplexity and n_iter based on sample size
            tsne_perplexity = min(30.0, float(max(1, n_samples - 2))) # Perplexity must be less than n_samples - 1
            tsne_n_iter = max(250, n_samples * 4) # More iterations for larger N
            tsne_init_method = 'pca' if n_samples > 2 and embeddings.shape[1] >=2 else 'random' # PCA init needs >2 components possible
            embeddings_2d = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity, 
                                 n_iter=tsne_n_iter, init=tsne_init_method, learning_rate='auto').fit_transform(embeddings)
        elif method == "umap" and umap: # umap is guaranteed to be available here by prior check
            umap_n_neighbors = min(15, max(2, n_samples -1 if n_samples >1 else 2)) # n_neighbors must be less than n_samples
            umap_min_dist = 0.1 if n_samples > 15 else 0.5 # Adjust min_dist for small N
            embeddings_2d = umap.UMAP(n_components=2, random_state=42, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist).fit_transform(embeddings)
        
        if embeddings_2d is None: # Fallback if a method was chosen but failed for some reason (e.g. UMAP not installed and logic error)
             pca_n_components_fb = min(2, embeddings.shape[0], embeddings.shape[1])
             if pca_n_components_fb < 1: logger.error("Fallback PCA failed due to insufficient components."); return False
             logger.warning(f"Dimensionality reduction failed for {title}, attempting fallback PCA."); 
             embeddings_2d = PCA(n_components=pca_n_components_fb, random_state=42, svd_solver='full').fit_transform(embeddings)
             if pca_n_components_fb == 1: embeddings_2d = np.hstack((embeddings_2d, np.zeros_like(embeddings_2d)))

    except Exception as e_reduce:
        logger.error(f"Dimensionality reduction ({method.upper()}) failed for {title}: {e_reduce}. Skipping plot.", exc_info=True)
        return False

    if embeddings_2d is None or embeddings_2d.shape[1] < 2 :
        logger.error(f"Dimensionality reduction resulted in < 2 dimensions for {title}. Skipping plot.")
        return False

    plt.figure(figsize=(12, 10))
    unique_plot_labels = sorted(list(set(l for l in labels if l and str(l).strip()))) # Filter out None or empty string labels

    if not unique_plot_labels or len(unique_plot_labels) == 1: # Only one type of label or no labels
        label_name = unique_plot_labels[0] if unique_plot_labels else "Data Points"
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7, label=label_name)
    else:
        try:
            cmap = plt.get_cmap('gist_rainbow', len(unique_plot_labels))
        except ValueError: # If len(unique_plot_labels) is too small for gist_rainbow
            cmap = plt.get_cmap('viridis', len(unique_plot_labels))
            
        for i, label_val in enumerate(unique_plot_labels):
            indices = [j for j, current_lab in enumerate(labels) if current_lab == label_val]
            if indices:
                plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], 
                            label=str(label_val), alpha=0.7, 
                            color=cmap(i / len(unique_plot_labels)))
        
        # Adjust legend position to prevent overlap, especially with many labels
        box = plt.gca().get_position()
        plt.gca().set_position([box.x0, box.y0 + box.height * 0.1, box.width, box.height * 0.9])
        num_legend_cols = min(5, (len(unique_plot_labels) // 10) + 1 if len(unique_plot_labels) > 1 else 1) # Adapt columns
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), 
                   fancybox=True, shadow=True, ncol=num_legend_cols, title="Labels")

    plt.title(title)
    plt.xlabel(f"{method.upper()} Component 1")
    plt.ylabel(f"{method.upper()} Component 2")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    try:
        plt.savefig(filepath, bbox_inches='tight')
        plt.close()
        logger.info(f"Embedding plot saved: {filepath}")
        return True
    except Exception as e_save:
        logger.error(f"Failed to save embedding plot {filepath}: {e_save}", exc_info=True)
        plt.close()
        return False

def plot_metric_trends(metrics_history: Dict, report_dir: str):
    num_epochs_trained = metrics_history.get('num_epochs_trained', 0)
    if num_epochs_trained == 0:
        logger.info("No epochs trained or metrics history empty, skipping metric trend plots.")
        return {}
        
    epochs_axis = range(1, num_epochs_trained + 1)
    generated_plot_paths = {}

    metric_groups_to_plot = {
        "loss_curves": (["train_loss", "eval_loss"], "Training & Validation Loss Curves", "Loss Value"),
        "learning_rate": (["learning_rate"], "Learning Rate Schedule", "Learning Rate"),
        "temperature": (["temperature"], "Learned Temperature (1 / Logit Scale)", "Temperature"),
        "gradient_norm": (["avg_grad_norm"], "Average Gradient Norm (During Training)", "L2 Norm"),
        "retrieval_metrics": ([m_key for m_key in metrics_history if "R@" in m_key or "mrr_" in m_key], "Retrieval Metrics (Validation Set)", "Score"),
        "knn_biotype_metrics": ([m_key for m_key in metrics_history if "knn_biotype_" in m_key and "report" not in m_key], "k-NN Biotype Classification Metrics (Validation Set)", "Score"),
        "clustering_metrics": ([m_key for m_key in metrics_history if "silhouette_" in m_key or "davies_bouldin_" in m_key], "Clustering Quality Metrics (Validation Set)", "Score")
    }

    for plot_filename_key, (metric_keys_in_group, plot_title, y_axis_label) in metric_groups_to_plot.items():
        plt.figure(figsize=(10, 6))
        data_plotted_flag = False
        
        actual_metrics_in_group = [mk for mk in metric_keys_in_group if mk in metrics_history and len(metrics_history[mk]) > 0]

        for metric_name in actual_metrics_in_group:
            metric_values = metrics_history[metric_name][:num_epochs_trained] # Ensure correct length
            if metric_values: # Check if list is not empty
                current_epochs_axis = epochs_axis[:len(metric_values)] # Align x-axis with available data points
                plt.plot(current_epochs_axis, metric_values, marker='o', linestyle='-', label=metric_name.replace("_", " ").title())
                data_plotted_flag = True
        
        if data_plotted_flag:
            plt.title(plot_title)
            plt.xlabel("Epoch Number")
            plt.ylabel(y_axis_label)
            num_legend_cols_trend = max(1, len(actual_metrics_in_group) // 2 if len(actual_metrics_in_group) > 2 else 1)
            plt.legend(loc='best', ncol=num_legend_cols_trend)
            plt.grid(True)
            plt.tight_layout()
            plot_filepath = os.path.join(report_dir, f"{plot_filename_key}.png")
            try:
                plt.savefig(plot_filepath)
                plt.close()
                generated_plot_paths[plot_filename_key] = os.path.basename(plot_filepath)
            except Exception as e_trend_save:
                logger.error(f"Failed to save metric trend plot {plot_filepath}: {e_trend_save}", exc_info=True)
                plt.close()
        else:
            plt.close() # Close the figure if no data was plotted
            logger.debug(f"Skipping plot for '{plot_title}': No data found or mismatched epoch counts.")
            
    return generated_plot_paths

def plot_similarity_matrix_snippet(similarity_matrix: Optional[np.ndarray], report_dir: str, epoch_identifier: str, max_elements_display: int = 20):
    if similarity_matrix is None or similarity_matrix.size == 0:
        logger.warning("Similarity matrix is empty or None. Skipping snippet plot.")
        return None

    num_to_display = min(similarity_matrix.shape[0], similarity_matrix.shape[1], max_elements_display)
    if num_to_display == 0:
        logger.warning("Cannot display 0x0 similarity matrix snippet. Skipping plot.")
        return None

    plt.figure(figsize=(10, 8))
    show_annotations = num_to_display <= 10 # Only annotate if small enough
    annotation_format = ".2f" if show_annotations else ""

    sns.heatmap(similarity_matrix[:num_to_display, :num_to_display], 
                annot=show_annotations, cmap="viridis", cbar=True, fmt=annotation_format,
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title(f"Similarity Matrix Snippet ({num_to_display}x{num_to_display}) - {epoch_identifier}")
    plt.xlabel("Text Sample Index")
    plt.ylabel("DNA Sample Index")
    plt.tight_layout()
    
    safe_epoch_id = re.sub(r'[^\w\s-]', '', epoch_identifier.lower()).strip().replace(' ', '_')
    filename = f"similarity_matrix_snippet_{safe_epoch_id}.png"
    filepath = os.path.join(report_dir, filename)
    
    try:
        plt.savefig(filepath)
        plt.close()
        logger.info(f"Similarity matrix snippet plot saved: {filepath}")
        return os.path.basename(filepath)
    except Exception as e_sim_save:
        logger.error(f"Failed to save similarity matrix plot {filepath}: {e_sim_save}", exc_info=True)
        plt.close()
        return None

def plot_sequence_length_distributions(dataset_summary: Dict, report_dir: str):
    generated_plot_paths = {}
    length_data_keys = {
        "dna_lengths_raw": "Raw DNA Sequence Lengths (bases/k-mers)", # Clarify if k-mers later if needed
        "text_lengths_raw_words": "Raw Annotation Lengths (words)"
    }

    for data_key, plot_title_suffix in length_data_keys.items():
        if data_key in dataset_summary and dataset_summary[data_key]:
            lengths_array = np.array(dataset_summary[data_key])
            if lengths_array.size == 0: continue # Skip if no data

            plt.figure(figsize=(10,6))
            num_unique_vals = len(np.unique(lengths_array))
            # Determine appropriate number of bins
            hist_bins = min(50, max(10, num_unique_vals // 2 if num_unique_vals > 20 else num_unique_vals if num_unique_vals > 0 else 10))
            
            plt.hist(lengths_array, bins=hist_bins, color='skyblue', edgecolor='black', density=True, alpha=0.7)
            
            mean_len = np.mean(lengths_array)
            median_len = np.median(lengths_array)
            std_dev_len = np.std(lengths_array)
            q25, q75 = np.percentile(lengths_array, [25, 75])

            plt.axvline(mean_len, color='red', linestyle='dashed', linewidth=1.5, label=f'Mean: {mean_len:.1f}')
            plt.axvline(median_len, color='green', linestyle='dashed', linewidth=1.5, label=f'Median: {median_len:.1f}')
            plt.axvline(q25, color='purple', linestyle='dotted', linewidth=1.2, label=f'25th Pctl: {q25:.1f}')
            plt.axvline(q75, color='orange', linestyle='dotted', linewidth=1.2, label=f'75th Pctl: {q75:.1f}')
            
            plot_title = f"Distribution of {plot_title_suffix}"
            plt.title(plot_title)
            plt.xlabel("Length")
            plt.ylabel("Density") # Since density=True
            plt.legend(title=f"Stats (StdDev: {std_dev_len:.1f})")
            plt.grid(axis='y', linestyle='--', alpha=0.75)
            plt.tight_layout()

            plot_filename = f"{data_key}_distribution.png"
            plot_filepath = os.path.join(report_dir, plot_filename)
            try:
                plt.savefig(plot_filepath)
                plt.close()
                generated_plot_paths[f"{data_key}_dist_plot"] = os.path.basename(plot_filepath)
            except Exception as e_len_dist_save:
                logger.error(f"Failed to save length distribution plot {plot_filepath}: {e_len_dist_save}", exc_info=True)
                plt.close()
    return generated_plot_paths

def plot_gff_feature_type_distribution(dataset_summary: Dict, report_dir: str):
    """Plots the distribution of GFF3 feature types if available."""
    if 'gff_feature_type_counts_raw' not in dataset_summary or not dataset_summary['gff_feature_type_counts_raw']:
        return None # No GFF data or no counts

    counts = Counter(dataset_summary['gff_feature_type_counts_raw'])
    if not counts: return None

    # Sort by count descending, take top N or all if fewer
    top_n = 20
    labels, values = zip(*counts.most_common(top_n))
    
    plt.figure(figsize=(max(12, len(labels) * 0.5), 7)) # Dynamic width
    bars = plt.bar(labels, values, color='cornflowerblue')
    plt.xlabel("GFF3 Feature Type")
    plt.ylabel("Count")
    plt.title(f"Distribution of GFF3 Feature Types (Top {min(top_n, len(labels))})")
    plt.xticks(rotation=45, ha="right", fontsize=9) # Rotate labels for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Add counts on top of bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01 * max(values), # Adjust offset
                 f'{int(yval)}', ha='center', va='bottom', fontsize=8)

    filename = "gff_feature_type_distribution.png"
    filepath = os.path.join(report_dir, filename)
    try:
        plt.savefig(filepath)
        plt.close()
        logger.info(f"GFF3 feature type distribution plot saved: {filepath}")
        return os.path.basename(filepath)
    except Exception as e_gff_plot_save:
        logger.error(f"Failed to save GFF3 feature type distribution plot {filepath}: {e_gff_plot_save}", exc_info=True)
        plt.close()
        return None


class PDFReport(FPDF):
    def __init__(self, orientation='P', unit='mm', format='A4', title="Training Report"):
        super().__init__(orientation, unit, format)
        self.report_title_text = title 
        self.set_auto_page_break(auto=True, margin=15)
        self.setup_fonts()
        self.add_page()
        self.set_font(self.font_family_main, "B", 16)
        self.cell(0, 10, self.report_title_text, 0, 1, "C")
        self.ln(5)

    def setup_fonts(self):
        self.font_family_main = "Helvetica"
        self.font_family_mono = "Courier"
        # Ensure fonts are available or use core fonts
        try:
            self.add_font(self.font_family_main, "", "Helvetica.ttf", uni=True) # Example, ensure ttf path if needed
            self.add_font(self.font_family_main, "B", "Helvetica-Bold.ttf", uni=True)
            self.add_font(self.font_family_main, "I", "Helvetica-Oblique.ttf", uni=True)
            self.add_font(self.font_family_mono, "", "Courier.ttf", uni=True)
        except RuntimeError: # Font file not found, FPDF falls back to core fonts if names match
            logger.debug("FPDF custom font loading skipped, will use core fonts.")
            pass # Core fonts (Helvetica, Courier) should be fine.

    def header(self):
        if self.page_no() == 1: return # No header on title page
        current_font_family = self.font_family
        current_font_style = self.font_style
        current_font_size = self.font_size_pt

        self.set_font(self.font_family_main, "I", 8)
        self.set_y(10) # Position from top
        self.cell(0, 10, f"{self.report_title_text} - Page {self.page_no()}/{{nb}}", 0, 0, "C")
        self.ln(10) # Move below header
        
        # Restore font
        self.set_font(current_font_family, current_font_style, current_font_size)


    def chapter_title(self, title: str):
        self.set_font(self.font_family_main, "B", 12)
        self.set_fill_color(230, 230, 230) # Light grey background
        self.cell(0, 8, title, 0, 1, "L", fill=True)
        self.ln(4)

    def chapter_body(self, text: str, font_style="", is_code=False):
        font_family = self.font_family_mono if is_code else self.font_family_main
        self.set_font(font_family, font_style, 10)
        self.multi_cell(0, 5, text, border=0, align="L")
        self.ln(2)

    def add_plot(self, image_path: str, caption: str = "", width_percent: float = 0.85):
        if not os.path.exists(image_path):
            self.chapter_body(f"[Plot Image Missing: {os.path.basename(image_path)}]", font_style="I")
            return
        if not PIL:
            self.chapter_body(f"[PIL/Pillow not installed, cannot add plot: {os.path.basename(image_path)}]", font_style="I")
            return

        page_width_mm = self.w - 2 * self.l_margin # Usable page width
        img_display_width_mm = page_width_mm * width_percent
        img_display_height_mm = 0 # Will be calculated

        try:
            with PIL.Image.open(image_path) as img_pil:
                orig_w_px, orig_h_px = img_pil.size
            aspect_ratio = orig_h_px / orig_w_px if orig_w_px > 0 else 1.0
            img_display_height_mm = img_display_width_mm * aspect_ratio
        except Exception as e_pil:
            logger.warning(f"PIL error opening or reading image {image_path}: {e_pil}. Using default aspect ratio.")
            img_display_height_mm = img_display_width_mm * (3/4) # Default aspect ratio

        # Check if image fits, if not, add new page
        if self.get_y() + img_display_height_mm + 20 > self.page_break_trigger : # 20 for caption and spacing
            self.add_page()

        if caption:
            self.set_font(self.font_family_main, "I", 9)
            self.multi_cell(0, 5, caption, 0, "C") # Centered caption
            self.ln(1)
        
        x_pos_mm = (page_width_mm - img_display_width_mm) / 2 + self.l_margin # Center image

        try:
            self.image(image_path, x=x_pos_mm, w=img_display_width_mm, h=img_display_height_mm)
        except RuntimeError as e_fpdf_img: # FPDF error (e.g. unsupported image format by FPDF directly)
            logger.error(f"FPDF error placing image {image_path}: {e_fpdf_img}")
            self.chapter_body(f"[Error displaying plot: {os.path.basename(image_path)} - {e_fpdf_img}]", font_style="I")

        self.ln(max(3, img_display_height_mm * 0.05)) # Spacing after image


    def add_table(self, header_cols: List[str], data_rows: List[List[Any]], title: str = "", 
                  col_widths_proportions: Optional[List[float]] = None, header_fill: bool = False):
        if title: self.chapter_title(title)
        
        self.set_font(self.font_family_main, "B", 9)
        page_width_usable = self.w - 2 * self.l_margin
        num_columns = len(header_cols)

        if col_widths_proportions and len(col_widths_proportions) == num_columns:
            effective_col_widths = [page_width_usable * prop for prop in col_widths_proportions]
        else:
            effective_col_widths = [page_width_usable / num_columns if num_columns > 0 else 0] * num_columns
        
        line_height = 7 # mm

        if header_fill: self.set_fill_color(200, 220, 255) # Light blue for header

        for i, header_text in enumerate(header_cols):
            self.cell(effective_col_widths[i], line_height, str(header_text), border=1, ln=0, align="C", fill=header_fill)
        self.ln(line_height)

        self.set_font(self.font_family_main, "", 8) # Smaller font for data
        for row_data in data_rows:
            # Check for page break before drawing a row
            if self.get_y() + line_height > self.page_break_trigger:
                self.add_page()
                # Redraw header on new page
                self.set_font(self.font_family_main, "B", 9)
                if header_fill: self.set_fill_color(200, 220, 255)
                for i, header_text in enumerate(header_cols):
                    self.cell(effective_col_widths[i], line_height, str(header_text), border=1, ln=0, align="C", fill=header_fill)
                self.ln(line_height)
                self.set_font(self.font_family_main, "", 8)

            for i, cell_item in enumerate(row_data):
                item_str = str(cell_item)
                # Truncate long strings to fit cell (approximate)
                # Char width in mm is roughly font_size_pt * 0.35 * 0.35 (pt to mm)
                # For font size 8, char_width_approx = 8 * 0.35 * 0.35 * 0.3 = 0.2 mm.
                # Max chars = col_width_mm / char_width_approx
                max_chars_in_cell = int(effective_col_widths[i] / 1.8) if effective_col_widths[i] > 0 else 10 # Heuristic, adjust as needed
                if len(item_str) > max_chars_in_cell:
                    item_str = item_str[:max_chars_in_cell-3] + "..."
                
                try:
                    self.cell(effective_col_widths[i], line_height, item_str, border=1, ln=0, align="L")
                except Exception as e_cell: # Catch potential FPDF errors with strange characters
                    logger.error(f"PDF table cell write error: {e_cell}, item (truncated): {item_str[:30]}")
                    self.cell(effective_col_widths[i], line_height, "[ERR]", border=1, ln=0, align="L")

            self.ln(line_height)
        self.ln(5) # Extra space after table

    def add_classification_report_table(self, report_dict: Dict, title: str):
        self.chapter_title(title)
        header = ["Class", "Precision", "Recall", "F1-Score", "Support"]
        col_widths_props = [0.3, 0.175, 0.175, 0.175, 0.175] # Proportions of page width
        data_rows = []

        # Sort classes alphabetically, but put averages at the end
        class_items = sorted([
            (label, metrics) for label, metrics in report_dict.items() 
            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']
        ], key=lambda x: str(x[0])) # Sort by class name

        average_items = []
        for avg_key in ['macro avg', 'weighted avg']: # Ensure consistent order for averages
            if avg_key in report_dict and isinstance(report_dict[avg_key], dict):
                average_items.append((avg_key, report_dict[avg_key]))
        
        for label, metrics_dict in class_items + average_items:
            row = [
                str(label),
                f"{metrics_dict.get('precision', 0.0):.3f}",
                f"{metrics_dict.get('recall', 0.0):.3f}",
                f"{metrics_dict.get('f1-score', 0.0):.3f}",
                str(metrics_dict.get('support', 'N/A'))
            ]
            data_rows.append(row)
        
        if 'accuracy' in report_dict and isinstance(report_dict['accuracy'], (float, int)):
            # Calculate total support for accuracy row if not directly available
            total_support_for_accuracy = sum(int(class_metrics.get('support',0)) for _, class_metrics in class_items) if class_items else 'N/A'
            accuracy_row = [
                "Accuracy", 
                f"{report_dict['accuracy']:.3f}", 
                "", # P, R, F1 not applicable for overall accuracy in this format
                "", 
                str(total_support_for_accuracy)
            ]
            data_rows.append(accuracy_row)

        if data_rows:
            self.add_table(header, data_rows, col_widths_proportions=col_widths_props, header_fill=True)
        else:
            self.chapter_body("No classification report data available to display.")

def generate_report(args: argparse.Namespace, metrics_history: Dict, report_dir: str, dataset_summary: Dict, 
                    final_similarity_matrix: Optional[np.ndarray] = None, 
                    final_eval_epoch_descriptor: str = "Final Model Evaluation"):
    logger.info("Generating training and evaluation reports...")
    report_gen_start_time = time.time()

    if 'plot_paths' not in metrics_history: metrics_history['plot_paths'] = defaultdict(list)

    # Generate plots and collect their paths
    trend_plot_paths = plot_metric_trends(metrics_history, report_dir)
    for key, path_val in trend_plot_paths.items(): metrics_history['plot_paths'][key] = [path_val] # Store as list

    dist_plot_paths = plot_sequence_length_distributions(dataset_summary, report_dir)
    for key, path_val in dist_plot_paths.items(): metrics_history['plot_paths'][key] = [path_val]

    gff_dist_plot_path = plot_gff_feature_type_distribution(dataset_summary, report_dir)
    if gff_dist_plot_path: metrics_history['plot_paths']['gff_feature_type_dist_plot'] = [gff_dist_plot_path]


    # --- Markdown Report ---
    md_report_content = f"# DNA-Clip Model Training Report\n\n"
    md_report_content += f"Report Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    input_file_display = os.path.basename(args.input_file) if args.input_file else "N/A (e.g., resumed without new input)"
    md_report_content += f"Input Data File: `{input_file_display}` (Type: `{args.input_type}`)\n"
    if args.input_type in ['bed', 'gff3'] and args.reference_fasta:
        md_report_content += f"Reference FASTA: `{os.path.basename(args.reference_fasta)}`\n"
    md_report_content += f"Main Output Directory: `{args.output_dir}`\n\n"

    md_report_content += f"## 1. Experiment Configuration\n\n| Parameter | Value |\n|---|---|\n"
    for arg_name, arg_val in sorted(vars(args).items()):
        md_report_content += f"| `{arg_name}` | `{arg_val}` |\n"
    
    md_report_content += f"\n## 2. Dataset Overview\n\n| Statistic | Value |\n|---|---|\n"
    dataset_stat_keys_ordered = [
        "total_samples_in_file", "training_samples", "validation_samples",
        "dna_tokenizer_type", "kmer_k_effective", 
        "dna_vocab_size", "text_vocab_size",
        "max_dna_len_config", "max_text_len_config"
    ]
    for key in dataset_stat_keys_ordered:
        if key in dataset_summary:
            md_report_content += f"| {key.replace('_',' ').title()} | {dataset_summary[key]} |\n"
    md_report_content += "\n"

    # Dataset distribution plots
    for plot_key, plot_paths_list in metrics_history['plot_paths'].items():
        if ("dist_plot" in plot_key or "gff_feature_type_dist_plot" in plot_key) and plot_paths_list:
            plot_title = plot_key.replace("_dist_plot", "").replace("_plot","").replace("_", " ").title().replace("Dna", "DNA") + " Distribution"
            md_report_content += f"### {plot_title}\n![{plot_title}]({plot_paths_list[-1]})\n\n"

    # Top N Biotype/Species counts
    for count_type_key, count_title in [("biotype_counts", "Biotype"), ("species_counts", "Species"), ("gff_feature_type_counts_processed", "GFF Feature Type (Processed)")]:
        if count_type_key in dataset_summary and dataset_summary[count_type_key]:
            md_report_content += f"\n### {count_title} Distribution (Top 15):\n| {count_title} | Count |\n|---|---|\n"
            # Ensure it's a Counter or dict for .items()
            counts_data = dataset_summary[count_type_key]
            if isinstance(counts_data, Counter): sorted_counts = counts_data.most_common()
            elif isinstance(counts_data, dict): sorted_counts = sorted(counts_data.items(), key=lambda item: item[1], reverse=True)
            else: sorted_counts = [] # Should not happen

            for item_name, item_count in sorted_counts[:15]:
                md_report_content += f"| {item_name} | {item_count} |\n"
            if len(sorted_counts) > 15:
                md_report_content += "| ... (others) | ... |\n"
    md_report_content += "\n"
    
    num_epochs_completed = metrics_history.get('num_epochs_trained', args.epochs if hasattr(args, 'epochs') else 0)
    md_report_content += f"## 3. Training & Validation Metric Trends (Across {num_epochs_completed} Epochs)\n\n"
    # Metric trend plots
    for plot_key, plot_paths_list in metrics_history['plot_paths'].items():
        if "trend" in plot_key.lower() and plot_paths_list : # check "trend" in key name
             plot_title = plot_key.replace("_trend", "").replace("_", " ").title().replace("Dna", "DNA")
             md_report_content += f"### {plot_title}\n![{plot_title}]({plot_paths_list[-1]})\n\n"


    best_val_epoch_idx = metrics_history.get('best_epoch_idx', -1)
    md_report_content += f"\n## 4. Detailed Evaluation Metrics & Plots ({final_eval_epoch_descriptor})\n\n"
    md_report_content += f"### Summary of Key Metrics ({final_eval_epoch_descriptor}):\n| Metric Name | Value |\n|---|---|\n"
    metrics_to_showcase = [
        'eval_loss', 'mrr_dna_to_text', 'mrr_text_to_dna', 
        'dna_to_text_R@1', 'dna_to_text_R@5', 'dna_to_text_R@10',
        'text_to_dna_R@1', 'text_to_dna_R@5', 'text_to_dna_R@10',
        'knn_biotype_accuracy', 'knn_biotype_f1_macro',
        'silhouette_score_biotype', 'davies_bouldin_index_biotype',
        'silhouette_score_species', 'davies_bouldin_index_species'
    ]
    for metric_name in metrics_to_showcase:
        if metric_name in metrics_history and metrics_history[metric_name]:
            values_list = metrics_history[metric_name]
            # Use value from best_val_epoch if available and valid, else last value
            target_idx = best_val_epoch_idx if (0 <= best_val_epoch_idx < len(values_list)) else (len(values_list) - 1)
            if target_idx != -1: # Ensure index is valid
                metric_value = values_list[target_idx]
                formatted_value = f"{metric_value:.4f}" if isinstance(metric_value, (float, np.floating)) else str(metric_value)
                md_report_content += f"| {metric_name.replace('_', ' ').title()} | {formatted_value} |\n"
    md_report_content += "\n"

    # k-NN Classification Report
    knn_reports_list = metrics_history.get('knn_classification_reports_biotype', [])
    if knn_reports_list:
        report_idx = best_val_epoch_idx if (0 <= best_val_epoch_idx < len(knn_reports_list)) else (len(knn_reports_list) - 1)
        if report_idx != -1:
            final_knn_report = knn_reports_list[report_idx]
            md_report_content += f"### k-NN Biotype Classification Report ({final_eval_epoch_descriptor}):\n"
            md_report_content += "| Class | Precision | Recall | F1-Score | Support |\n|---|---|---|---|---|\n"
            
            report_class_items = sorted([
                (lbl, m_dict) for lbl, m_dict in final_knn_report.items() 
                if isinstance(m_dict, dict) and lbl not in ['accuracy', 'macro avg', 'weighted avg']
            ], key=lambda x: str(x[0]))
            report_avg_items = [(k, final_knn_report[k]) for k in ['macro avg', 'weighted avg'] if k in final_knn_report and isinstance(final_knn_report[k], dict)]

            for label, metrics_d in report_class_items + report_avg_items:
                md_report_content += f"| {label} | {metrics_d.get('precision',0):.3f} | {metrics_d.get('recall',0):.3f} | {metrics_d.get('f1-score',0):.3f} | {metrics_d.get('support','N/A')} |\n"
            if 'accuracy' in final_knn_report:
                total_sup_acc = sum(int(v.get('support',0)) for k,v in report_class_items if isinstance(v,dict))
                md_report_content += f"| **Accuracy** | **{final_knn_report['accuracy']:.3f}** | | | **{total_sup_acc}** |\n"
    md_report_content += f"\n### Visualizations ({final_eval_epoch_descriptor}):\n"

    # Final evaluation plots (CM, embeddings)
    plot_keys_for_final_report = ['knn_biotype_cm'] + \
                                 [f"embedding_plot_{pm}_{lt}" for pm in (args.plot_methods if hasattr(args, 'plot_methods') else ["pca","tsne"]) for lt in ["biotype", "species"]]

    for plot_key in plot_keys_for_final_report:
        if plot_key in metrics_history['plot_paths'] and metrics_history['plot_paths'][plot_key]:
            plot_paths = metrics_history['plot_paths'][plot_key]
            # Display the plot relevant to the 'best_val_epoch_idx' or the last one if not applicable/available
            plot_path_to_display_idx = best_val_epoch_idx if (0 <= best_val_epoch_idx < len(plot_paths)) else -1
            if plot_path_to_display_idx < len(plot_paths) and plot_path_to_display_idx >= -len(plot_paths) : # check valid index
                plot_path_to_display = plot_paths[plot_path_to_display_idx]
                plot_title_md = plot_key.replace("_", " ").title().replace("Dna", "DNA")
                md_report_content += f"#### {plot_title_md}\n![{plot_title_md}]({plot_path_to_display})\n\n"

    # Similarity Matrix Snippet
    if final_similarity_matrix is not None:
        sim_matrix_plot_path = plot_similarity_matrix_snippet(final_similarity_matrix, report_dir, epoch_identifier=final_eval_epoch_descriptor)
        if sim_matrix_plot_path:
            md_report_content += f"#### Similarity Matrix Snippet ({final_eval_epoch_descriptor})\n![Similarity Matrix]({sim_matrix_plot_path})\n\n"
            metrics_history['plot_paths']['final_similarity_matrix_snippet'] = [sim_matrix_plot_path]

    md_report_filepath = os.path.join(report_dir, "training_evaluation_report.md")
    try: 
        with open(md_report_filepath, "w", encoding='utf-8') as f: 
            f.write(md_report_content)
        logger.info(f"Markdown report saved to: {md_report_filepath}")
    except IOError as e_md_save:
        logger.error(f"Failed to write Markdown report: {e_md_save}", exc_info=True)

    # --- PDF Report ---
    if not PIL:
        logger.warning("PIL/Pillow library not found. Skipping PDF report generation.")
    else:
        pdf_report_title = f"DNA-Clip Report: {os.path.basename(args.input_file)}" if args.input_file else "DNA-Clip Report (Resumed)"
        pdf = PDFReport(title=pdf_report_title)
        pdf.alias_nb_pages() # Enable page numbering {nb}

        # Section 1: Config
        pdf.chapter_title("1. Experiment Configuration")
        config_data_pdf = [[arg_name, str(arg_val)] for arg_name, arg_val in sorted(vars(args).items())]
        pdf.add_table(["Parameter", "Value"], config_data_pdf, col_widths_proportions=[0.3, 0.7])

        # Section 2: Dataset Overview
        pdf.chapter_title("2. Dataset Overview")
        dataset_info_pdf = [[key.replace('_',' ').title(), str(dataset_summary.get(key, 'N/A'))] for key in dataset_stat_keys_ordered]
        pdf.add_table(["Statistic", "Value"], dataset_info_pdf, col_widths_proportions=[0.5, 0.5])

        for plot_key, plot_paths_list in metrics_history['plot_paths'].items():
            if ("dist_plot" in plot_key or "gff_feature_type_dist_plot" in plot_key) and plot_paths_list:
                plot_title_pdf = plot_key.replace("_dist_plot", "").replace("_plot","").replace("_", " ").title().replace("Dna", "DNA") + " Distribution"
                pdf.add_plot(os.path.join(report_dir, plot_paths_list[-1]), caption=plot_title_pdf)
        
        for count_type_key, count_title in [("biotype_counts", "Biotype"), ("species_counts", "Species"), ("gff_feature_type_counts_processed", "GFF Feature Type (Processed)")]:
            if count_type_key in dataset_summary and dataset_summary[count_type_key]:
                counts_data = dataset_summary[count_type_key]
                if isinstance(counts_data, Counter): sorted_counts_pdf = counts_data.most_common()
                elif isinstance(counts_data, dict): sorted_counts_pdf = sorted(counts_data.items(), key=lambda item: item[1], reverse=True)
                else: sorted_counts_pdf = []
                
                counts_data_pdf = [[name, count] for name, count in sorted_counts_pdf[:10]] # Top 10 for PDF
                if len(sorted_counts_pdf) > 10: counts_data_pdf.append(["... (others)", "..."])
                if counts_data_pdf: # Only add table if there's data
                     pdf.add_table([count_title, "Count"], counts_data_pdf, title=f"{count_title} Distribution (Top 10)", col_widths_proportions=[0.7, 0.3], header_fill=True)
        
        # Section 3: Metric Trends
        pdf.add_page() # Ensure trends start on a new page if needed
        pdf.chapter_title(f"3. Training & Validation Metric Trends (Across {num_epochs_completed} Epochs)")
        for plot_key, plot_paths_list in metrics_history['plot_paths'].items():
             if "trend" in plot_key.lower() and plot_paths_list:
                plot_title_pdf_trend = plot_key.replace("_trend", "").replace("_", " ").title().replace("Dna", "DNA")
                pdf.add_plot(os.path.join(report_dir, plot_paths_list[-1]), caption=plot_title_pdf_trend)
        
        # Section 4: Final Eval
        pdf.add_page()
        pdf.chapter_title(f"4. Detailed Evaluation Metrics & Visualizations ({final_eval_epoch_descriptor})")
        summary_metrics_pdf_data = []
        for metric_name in metrics_to_showcase:
            if metric_name in metrics_history and metrics_history[metric_name]:
                values_list_pdf = metrics_history[metric_name]
                target_idx_pdf = best_val_epoch_idx if (0 <= best_val_epoch_idx < len(values_list_pdf)) else (len(values_list_pdf) - 1)
                if target_idx_pdf != -1:
                    metric_value_pdf = values_list_pdf[target_idx_pdf]
                    formatted_val_pdf = f"{metric_value_pdf:.4f}" if isinstance(metric_value_pdf, (float, np.floating)) else str(metric_value_pdf)
                    summary_metrics_pdf_data.append([metric_name.replace('_', ' ').title(), formatted_val_pdf])
        if summary_metrics_pdf_data:
            pdf.add_table(["Metric Name", "Value"], summary_metrics_pdf_data, title="Summary of Key Metrics", col_widths_proportions=[0.6, 0.4], header_fill=True)

        if knn_reports_list and report_idx != -1 :
            pdf.add_classification_report_table(knn_reports_list[report_idx], title=f"k-NN Biotype Classification Report ({final_eval_epoch_descriptor})")

        for plot_key in plot_keys_for_final_report:
            if plot_key in metrics_history['plot_paths'] and metrics_history['plot_paths'][plot_key]:
                plot_paths_pdf = metrics_history['plot_paths'][plot_key]
                plot_path_to_display_idx_pdf = best_val_epoch_idx if (0 <= best_val_epoch_idx < len(plot_paths_pdf)) else -1
                if plot_path_to_display_idx_pdf < len(plot_paths_pdf) and plot_path_to_display_idx_pdf >= -len(plot_paths_pdf):
                    plot_path_to_display_pdf = plot_paths_pdf[plot_path_to_display_idx_pdf]
                    plot_title_pdf_final = plot_key.replace("_", " ").title().replace("Dna", "DNA")
                    pdf.add_plot(os.path.join(report_dir, plot_path_to_display_pdf), caption=plot_title_pdf_final)
        
        if 'final_similarity_matrix_snippet' in metrics_history['plot_paths'] and metrics_history['plot_paths']['final_similarity_matrix_snippet']:
            final_sim_plot_path = metrics_history['plot_paths']['final_similarity_matrix_snippet'][-1]
            pdf.add_plot(os.path.join(report_dir, final_sim_plot_path), caption=f"Similarity Matrix Snippet ({final_eval_epoch_descriptor})")

        pdf_report_filepath = os.path.join(report_dir, "training_evaluation_report.pdf")
        try:
            pdf.output(pdf_report_filepath, "F")
            logger.info(f"PDF report saved to: {pdf_report_filepath}")
        except Exception as e_pdf_save:
            logger.error(f"Failed to generate PDF report: {e_pdf_save}. PDF may be incomplete or corrupted.", exc_info=True)
    
    logger.info(f"Report generation finished in {time.time() - report_gen_start_time:.2f}s.")


def expand_fasta_curly_braces(fasta_content_str: str):
    """Expands sequences like {A*10} into AAAAAAAAAA in FASTA data."""
    def expand_matched_sequence(match_obj):
        base_char, repeat_count = match_obj.group(1), int(match_obj.group(2))
        return base_char * repeat_count

    expanded_lines = []
    for line_entry in fasta_content_str.splitlines(): # Use splitlines to handle \n correctly
        if line_entry.startswith('>'):
            expanded_lines.append(line_entry)
        else:
            # Repeatedly apply substitution until no more {X*Y} patterns exist
            processed_line = line_entry
            while re.search(r'\{(\w)\*(\d+)\}', processed_line):
                 processed_line = re.sub(r'\{(\w)\*(\d+)\}', expand_matched_sequence, processed_line)
            expanded_lines.append(processed_line)
    return '\n'.join(expanded_lines)

def export_onnx_to_tfjs(onnx_model_filepath: str, tfjs_output_dir: str):
    if not tensorflow or not onnx or not onnx_tf_prepare or not tensorflowjs:
        logger.error("Missing one or more libraries for TF.js export: tensorflow, onnx, onnx-tf, tensorflowjs. Skipping.")
        return
    if not os.path.exists(onnx_model_filepath):
        logger.error(f"ONNX model file not found: {onnx_model_filepath}. Skipping TF.js export.")
        return

    base_model_name = os.path.splitext(os.path.basename(onnx_model_filepath))[0]
    temp_saved_model_dir = None
    export_tfjs_start_time = time.time()

    try:
        logger.info(f"Starting TensorFlow.js export for {onnx_model_filepath} to {tfjs_output_dir}")
        
        onnx_model_proto = onnx.load(onnx_model_filepath)
        onnx.checker.check_model(onnx_model_proto) # Validate ONNX model
        
        logger.info("Converting ONNX model to TensorFlow representation...")
        tf_rep = onnx_tf_prepare(onnx_model_proto) # onnx_tf.backend.prepare
        
        temp_saved_model_dir = tempfile.mkdtemp(prefix=f"tf_sm_{base_model_name}_")
        logger.info(f"Exporting TensorFlow SavedModel to temporary directory: {temp_saved_model_dir}")
        tf_rep.export_graph(temp_saved_model_dir) # Exports to SavedModel format
        
        os.makedirs(tfjs_output_dir, exist_ok=True)
        logger.info(f"Converting TensorFlow SavedModel to TensorFlow.js format in {tfjs_output_dir}...")
        
        # Construct the command for tensorflowjs_converter
        # Using sys.executable to ensure the correct Python environment's tensorflowjs is used
        tfjs_converter_cmd = [
            sys.executable, "-m", "tensorflowjs.converters.tf_saved_model_conversion_cli",
            "--input_format=tf_saved_model",
            "--output_format=tfjs_graph_model",
            # Potentially add other options like --quantization_bytes, --weight_shard_size_bytes
            temp_saved_model_dir, # Input SavedModel directory
            tfjs_output_dir       # Output directory for TF.js model
        ]
        
        logger.info(f"Executing TensorFlow.js converter command: {' '.join(tfjs_converter_cmd)}")
        conversion_process = subprocess.run(tfjs_converter_cmd, capture_output=True, text=True, check=False) # check=False to handle errors manually
        
        if conversion_process.returncode == 0:
            logger.info(f"TensorFlow.js model exported successfully to {tfjs_output_dir}")
        else:
            logger.error(f"TensorFlow.js conversion failed with return code {conversion_process.returncode}.")
        
        if conversion_process.stdout: # Log STDOUT regardless of success for debugging
            logger.debug(f"TFJS Converter STDOUT:\n{conversion_process.stdout}")
        if conversion_process.stderr and conversion_process.returncode != 0 : # Log STDERR only on error
            logger.error(f"TFJS Converter STDERR:\n{conversion_process.stderr}")

    except Exception as e_tfjs:
        logger.error(f"An error occurred during TensorFlow.js export process: {e_tfjs}", exc_info=True)
    finally:
        if temp_saved_model_dir and os.path.exists(temp_saved_model_dir):
            try:
                shutil.rmtree(temp_saved_model_dir)
                logger.info(f"Cleaned up temporary SavedModel directory: {temp_saved_model_dir}")
            except Exception as e_cleanup:
                logger.error(f"Failed to clean up temporary directory {temp_saved_model_dir}: {e_cleanup}")
        logger.info(f"TensorFlow.js export process finished in {time.time() - export_tfjs_start_time:.2f}s.")

# --- Main Script Execution ---
def main():
    main_start_time = time.time()
    parser = argparse.ArgumentParser(description="Train DNA-Annotation CLIP-like model.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    io_group = parser.add_argument_group('Input/Output options')
    io_group.add_argument("--input_file",type=str,required=False,help="Path to FASTA/BED/GFF3 file (gzipped ok).")
    io_group.add_argument("--input_type",type=str,choices=['fasta','bed','gff3'],default='fasta',help="Input file type.")
    io_group.add_argument("--reference_fasta",type=str,help="Reference FASTA (required for BED and GFF3 input types).")
    io_group.add_argument("--output_dir",type=str,default="dna_clip_output",help="Directory for all outputs.")
    io_group.add_argument("--load_model_path",type=str,help="Path to load pre-trained model state_dict (weights only).")
    io_group.add_argument("--dna_tokenizer_type", type=str, default="kmer", choices=["kmer", "char"], help="Type of DNA tokenizer to use.")
    io_group.add_argument("--load_dna_tokenizer_path",type=str,help="Path to load pre-existing DNA tokenizer vocabulary.")
    io_group.add_argument("--load_text_tokenizer_path",type=str,help="Path to load pre-existing text tokenizer vocabulary.")
    io_group.add_argument("--resume_from_checkpoint",type=str,help="Path to a checkpoint file to resume full training state.")
    io_group.add_argument("--gff_feature_types", type=str, default=",".join(DEFAULT_GFF_FEATURE_TYPES), help="Comma-separated list of GFF3 feature types to process (e.g., gene,mRNA,ncRNA_gene). Used if --input_type is gff3.")

    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument("--kmer_k",type=int,default=DEFAULT_KMER_K,help="K-mer size (if using k-mer DNA tokenizer).")
    model_group.add_argument("--embedding_dim",type=int,default=DEFAULT_EMBEDDING_DIM,help="Dimension of embeddings.")
    model_group.add_argument("--hidden_dim",type=int,default=DEFAULT_HIDDEN_DIM,help="Dimension of LSTM hidden states.")
    model_group.add_argument("--num_layers",type=int,default=DEFAULT_NUM_LAYERS,help="Number of LSTM layers.")
    model_group.add_argument("--dropout",type=float,default=0.1,help="Dropout rate in model.")
    model_group.add_argument("--initial_temperature",type=float,default=DEFAULT_TEMPERATURE,help="Initial temperature for contrastive loss scaling.")

    train_group = parser.add_argument_group('Training Hyperparameters')
    train_group.add_argument("--epochs",type=int,default=DEFAULT_EPOCHS,help="Number of training epochs.")
    train_group.add_argument("--batch_size",type=int,default=DEFAULT_BATCH_SIZE,help="Batch size for training and evaluation.")
    train_group.add_argument("--learning_rate",type=float,default=DEFAULT_LEARNING_RATE,help="Initial learning rate.")
    train_group.add_argument("--max_dna_len",type=int,default=512,help="Maximum length for DNA sequences (tokens). 0 for dynamic (padded per batch).")
    train_group.add_argument("--max_text_len",type=int,default=128,help="Maximum length for text annotations (tokens). 0 for dynamic.")
    train_group.add_argument("--validation_split",type=float,default=0.1,help="Fraction of data for validation (0 to disable).")
    train_group.add_argument("--min_token_freq",type=int,default=1,help="Minimum frequency for tokens in vocabulary building.")
    train_group.add_argument("--optimizer_type",type=str,default="AdamW",choices=["AdamW","Adam","SGD"],help="Optimizer type.")
    train_group.add_argument("--weight_decay",type=float,default=0.01,help="Weight decay (L2 penalty), primarily for AdamW.")
    train_group.add_argument("--lr_scheduler_patience",type=int,default=3,help="Patience for ReduceLROnPlateau scheduler (epochs). 0 to disable.")
    train_group.add_argument("--lr_scheduler_factor",type=float,default=0.1,help="Factor by which LR is reduced by scheduler.")
    train_group.add_argument("--early_stopping_patience",type=int,default=5,help="Patience for early stopping (epochs). 0 to disable.")
    train_group.add_argument("--use_amp",action="store_true",help="Use Automatic Mixed Precision (AMP) for CUDA training.")
    train_group.add_argument("--use_torch_compile", action="store_true", help="Use torch.compile() for model optimization (PyTorch 2.0+).")

    export_group = parser.add_argument_group('Export Options')
    export_group.add_argument("--export_onnx",action="store_true",help="Export encoders to ONNX format after training.")
    export_group.add_argument("--export_tfjs", action="store_true", help="Export ONNX encoders to TensorFlow.js format after training (requires ONNX export).")
    export_group.add_argument("--export_embeddings_format",type=str,choices=["csv","parquet","both","none"],default="both",help="Format for exporting final embeddings.")
    
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument("--seed",type=int,default=42,help="Random seed for reproducibility.")
    misc_group.add_argument("--num_workers",type=int,default=0,help="Number of DataLoader workers. 0 for main process.")
    misc_group.add_argument("--run_dummy_example",action="store_true",help="Run with a small, internally generated dummy dataset for testing.")
    misc_group.add_argument("--test_suite",action="store_true",help="Run built-in unit tests and exit.")
    misc_group.add_argument("--plot_methods",nargs='+',default=["pca","tsne"],choices=["pca","tsne","umap"],help="Methods for embedding visualization.")
    misc_group.add_argument("--log_level",type=str,default="INFO",choices=["DEBUG","INFO","WARNING","ERROR","CRITICAL"],help="Logging level.")
    
    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))

    if args.test_suite:
        logger.info("Running unit test suite...")
        suite = unittest.TestSuite()
        suite.addTest(unittest.makeSuite(TestDNAClipComponents))
        runner = unittest.TextTestRunner(verbosity=2, stream=sys.stderr) # Use stderr for test output
        result = runner.run(suite)
        sys.exit(0 if result.wasSuccessful() else 1)

    if args.run_dummy_example:
        logger.info("Setting up for a dummy example run...")
        base_output_dir_dummy = args.output_dir # User specified or default "dna_clip_output"
        args.input_file = os.path.join(base_output_dir_dummy, "dummy_data.fasta.gz") # Store dummy data inside this base
        args.output_dir = os.path.join(base_output_dir_dummy, "dummy_model_outputs") # Outputs in a subfolder
        # Override some args for quick dummy run
        args.epochs=3; args.batch_size=4; args.max_dna_len=64; args.max_text_len=32
        args.embedding_dim=16; args.hidden_dim=32; args.validation_split=0.40
        args.early_stopping_patience=2; args.lr_scheduler_patience=1; args.num_workers = 0
        
        os.makedirs(base_output_dir_dummy, exist_ok=True) # Ensure base dir for dummy input exists
        
        # Dummy FASTA content (remains FASTA for simplicity, GFF3 dummy would be more complex)
        dummy_fasta_str_template = """>seq1|species=Human|biotype=GeneA|id=HS1\n{A*30}{C*30}\n>seq2|species=Mouse|biotype=GeneB|id=MM1\n{G*30}{T*30}\n>seq3|species=Human|biotype=GeneA|id=HS2\n{A*10}{T*10}{A*10}{T*10}{A*10}{T*10}\n>seq4|species=Mouse|biotype=GeneC|id=MM2\n{C*20}{G*20}{N*20}\n>seq5|species=Fly|biotype=GeneA|id=DM1\n{N*60}\n>seq6|species=Human|biotype=GeneB|id=HS3\n{A*15}{C*15}{G*15}{T*15}\n>seq7|species=Mouse|biotype=GeneC|id=MM3\n{T*60}\n>seq8|species=Fly|biotype=GeneB|id=DM2\n{G*20}{C*20}{A*20}\n>seq9|species=Human|biotype=GeneA|id=HS4\n{A*60}\n>seq10|species=Mouse|biotype=GeneB|id=MM4\n{C*60}\n>seq11|species=Fly|biotype=GeneC|id=DM3\n{G*60}\n>seq12|species=Human|biotype=GeneA|id=HS5\n{T*20}{N*20}{T*20}\n>seq13|species=Mouse|biotype=GeneB|id=MM5\n{A*20}{N*40}\n>seq14|species=Fly|biotype=GeneA|id=DM4\n{C*25}{G*35}\n>seq15|species=Human|biotype=GeneC|id=HS6\n{N*15}{A*15}{C*15}{G*15}\n>seq16|species=Mouse|biotype=GeneA|id=MM6\n{A*12}{C*12}{G*12}{T*12}{N*12}\n"""
        dummy_fasta_content_expanded = expand_fasta_curly_braces(dummy_fasta_str_template)
        with gzip.open(args.input_file, "wt", encoding='utf-8') as f_dummy:
            f_dummy.write(dummy_fasta_content_expanded)
        logger.info(f"Dummy FASTA data written to: {args.input_file}")
        logger.info(f"Dummy model outputs will be in: {args.output_dir}")
        if args.dna_tokenizer_type == "kmer" and args.kmer_k > 3:
            logger.info(f"For dummy run with k-mer tokenizer, reducing kmer_k to 3 for short sequences.")
            args.kmer_k = 3
        args.input_type = 'fasta' # Dummy run uses FASTA input

    elif not args.input_file and not args.resume_from_checkpoint:
        parser.error("Either --input_file or --resume_from_checkpoint must be provided for a non-dummy run.")

    # Create output directories
    try:
        os.makedirs(args.output_dir, exist_ok=True)
        report_dir = os.path.join(args.output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
    except OSError as e_mkdir:
        logger.critical(f"Failed to create output directories: {e_mkdir}"); sys.exit(1)

    # Reproducibility
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed) # For multi-GPU if used

    # Device and AMP setup
    if args.use_amp and not torch.cuda.is_available():
        logger.warning("AMP (Automatic Mixed Precision) was requested, but CUDA is not available. Disabling AMP.")
        args.use_amp = False
    
    if torch.cuda.is_available() and args.use_amp:
        torch.backends.cudnn.benchmark = True # Can speed up training if input sizes don't vary much
        logger.info("CUDA available. AMP enabled. CuDNN benchmark mode set to True.")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}. AMP active: {args.use_amp and device.type=='cuda'}")

    # Initialize training state variables
    start_epoch = 0; best_val_loss = float('inf'); epochs_without_improvement = 0
    metrics_history = defaultdict(list)
    metrics_history['plot_paths'] = defaultdict(list) # To store paths of generated plots
    metrics_history['knn_classification_reports_biotype'] = [] # Store full dicts for reports

    dna_tokenizer: Optional[BaseTokenizer] = None
    text_tokenizer: Optional[TextTokenizer] = None
    dataset_summary: Dict[str, Any] = {}
    train_dl: Optional[DataLoader] = None
    val_dl: Optional[DataLoader] = None
    
    data_loading_start_time = time.time()

    # Data loading and tokenizer setup
    if args.input_file: 
        logger.info(f"Loading data from {args.input_file} (type: {args.input_type})...")
        sequences_data: List[str] = []
        annotations_data: List[str] = []
        metadata_list: List[Dict[str,Any]] = []
        
        parsed_data_raw: List[Tuple[str, str, Dict[str, Any], Optional[str]]] = [] # (id, seq, meta, ann_text_opt)

        if args.input_type == 'fasta':
            parsed_fasta = parse_fasta(args.input_file)
            for id_val, seq_val, meta_val in parsed_fasta:
                parsed_data_raw.append((id_val, seq_val, meta_val, None)) # Annotation will be constructed
        elif args.input_type == 'bed':
            if not args.reference_fasta: 
                logger.critical("--reference_fasta is required for --input_type bed."); sys.exit(1)
            parsed_data_raw = parse_bed(args.input_file, args.reference_fasta)
        elif args.input_type == 'gff3':
            if not args.reference_fasta:
                logger.critical("--reference_fasta is required for --input_type gff3."); sys.exit(1)
            target_gff_types = [s.strip() for s in args.gff_feature_types.split(',') if s.strip()]
            logger.info(f"Processing GFF3 features of types: {target_gff_types}")
            parsed_data_raw = parse_gff3(args.input_file, args.reference_fasta, target_gff_types)

        # Populate sequences_data, annotations_data, metadata_list from parsed_data_raw
        for item_tuple in parsed_data_raw:
            id_val, seq_val, meta_val = item_tuple[0], item_tuple[1], item_tuple[2]
            ann_text_val = item_tuple[3] if len(item_tuple) > 3 and item_tuple[3] is not None else ""

            sequences_data.append(seq_val)
            metadata_list.append(meta_val)

            if ann_text_val: # If parser provided annotation (BED, GFF3)
                annotations_data.append(ann_text_val)
            elif args.input_type == 'fasta': # Construct annotation for FASTA
                biotype = meta_val.get("biotype", "unknown_biotype")
                species = meta_val.get("species", "unknown_species")
                identifier = meta_val.get("id", "unidentified_sequence")
                
                desc_parts = [meta_val.get(k) for k in ["description_freeform", "description", "note", "product", "gene", "function"] if meta_val.get(k)]
                description = " ".join(filter(None, desc_parts))
                
                ann_text_fasta = f"{biotype} from {species}, ID {identifier}. {description}".strip()
                annotations_data.append(re.sub(r'\s+', ' ', ann_text_fasta))
            else: # Should not happen if parsers are correct
                annotations_data.append("no annotation available")


        if not sequences_data:
            logger.critical(f"No sequences loaded from {args.input_file}. Check file, format, and GFF feature types if applicable."); sys.exit(1)
        
        logger.info(f"Successfully loaded {len(sequences_data)} sequence-annotation pairs in {time.time() - data_loading_start_time:.2f}s.")
        
        # Log a few examples
        logger.info("Example loaded data (first 3 entries):")
        for i_ex in range(min(3, len(sequences_data))):
            seq_snippet = sequences_data[i_ex][:50] + ("..." if len(sequences_data[i_ex]) > 50 else "")
            logger.info(f"  Example {i_ex+1}: Seq='{seq_snippet}', Ann='{annotations_data[i_ex][:100]}...'")
            logger.debug(f"  Metadata {i_ex+1}: {metadata_list[i_ex]}")


        # DNA Tokenizer
        dna_tokenizer_args = {}
        if args.dna_tokenizer_type == "kmer":
            dna_tokenizer_cls = KmerDNATokenizer
            dna_tokenizer_args['k'] = args.kmer_k
        elif args.dna_tokenizer_type == "char":
            dna_tokenizer_cls = CharDNATokenizer
        else: # Should be caught by argparse choices
            raise ValueError(f"Unsupported DNA tokenizer type: {args.dna_tokenizer_type}")

        if args.load_dna_tokenizer_path and os.path.exists(args.load_dna_tokenizer_path):
            logger.info(f"Loading DNA tokenizer from: {args.load_dna_tokenizer_path}")
            dna_tokenizer = dna_tokenizer_cls.load_vocab(args.load_dna_tokenizer_path)
        else:
            logger.info(f"Building DNA tokenizer ({args.dna_tokenizer_type}) from data...")
            dna_tokenizer = build_vocab_from_data(sequences_data, dna_tokenizer_cls, args.min_token_freq, dna_tokenizer_args)
        
        # Update kmer_k from loaded tokenizer if necessary
        if isinstance(dna_tokenizer, KmerDNATokenizer) and hasattr(dna_tokenizer, 'k') and dna_tokenizer.k != args.kmer_k:
            logger.info(f"K-mer size updated to {dna_tokenizer.k} from loaded DNA tokenizer.")
            args.kmer_k = dna_tokenizer.k

        # Text Tokenizer
        if args.load_text_tokenizer_path and os.path.exists(args.load_text_tokenizer_path):
            logger.info(f"Loading text tokenizer from: {args.load_text_tokenizer_path}")
            text_tokenizer = TextTokenizer.load_vocab(args.load_text_tokenizer_path)
        else:
            logger.info("Building text tokenizer from data...")
            text_tokenizer = build_vocab_from_data(annotations_data, TextTokenizer, args.min_token_freq)

        # Save tokenizers
        dna_tokenizer.save_vocab(os.path.join(args.output_dir, "dna_tokenizer_vocab.json"))
        text_tokenizer.save_vocab(os.path.join(args.output_dir, "text_tokenizer_vocab.json"))

        logger.info(f"DNA Tokenizer ({args.dna_tokenizer_type}): Vocab Size = {dna_tokenizer.get_vocab_size()}, PAD ID = {dna_tokenizer.pad_token_id}")
        if isinstance(dna_tokenizer, KmerDNATokenizer): logger.info(f"Effective K-mer k = {dna_tokenizer.k}")
        logger.info(f"Text Tokenizer: Vocab Size = {text_tokenizer.get_vocab_size()}, PAD ID = {text_tokenizer.pad_token_id}")

        # Train/Validation Split
        all_indices = list(range(len(sequences_data)))
        train_indices, val_indices = [], []

        if args.validation_split > 0 and args.validation_split < 1:
            try:
                # Attempt stratified split based on 'biotype' from metadata
                biotype_labels_for_split = [meta.get("biotype", "unknown") for meta in metadata_list]
                label_counts_split = Counter(biotype_labels_for_split)
                
                # Stratification requires at least 2 samples per class present in the split portion
                min_samples_per_class_for_stratify = 2 
                valid_for_stratify = all(c >= min_samples_per_class_for_stratify for c in label_counts_split.values()) \
                                     and len(label_counts_split) >= 2
                
                if valid_for_stratify:
                    train_indices, val_indices = train_test_split(all_indices, test_size=args.validation_split, 
                                                                  random_state=args.seed, stratify=biotype_labels_for_split)
                else:
                    logger.warning("Not enough samples or classes for stratified split based on biotype. Using random split.")
                    train_indices, val_indices = train_test_split(all_indices, test_size=args.validation_split, random_state=args.seed)
            except ValueError as e_split: # Catch errors from train_test_split itself
                logger.warning(f"Stratified split failed ({e_split}). Using random split.")
                train_indices, val_indices = train_test_split(all_indices, test_size=args.validation_split, random_state=args.seed)
        else: # No validation split
            train_indices = all_indices
            logger.info("Validation split is 0 or invalid, using all data for training (no validation set).")

        logger.info(f"Data split: {len(train_indices)} training samples, {len(val_indices)} validation samples.")

        # Create Datasets and DataLoaders
        ds_max_dna_len = args.max_dna_len if args.max_dna_len > 0 else None
        ds_max_text_len = args.max_text_len if args.max_text_len > 0 else None

        train_seqs = [sequences_data[i] for i in train_indices]
        train_anns = [annotations_data[i] for i in train_indices]
        train_meta = [metadata_list[i] for i in train_indices]
        train_ds = DNADataset(train_seqs, train_anns, dna_tokenizer, text_tokenizer, ds_max_dna_len, ds_max_text_len, train_meta)
        train_dl = create_dataloader(train_ds, args.batch_size, args.num_workers, shuffle=True, 
                                     dna_pad_id=dna_tokenizer.pad_token_id, text_pad_id=text_tokenizer.pad_token_id,
                                     pin_memory=(device.type=='cuda'), drop_last=True) # drop_last for training
        
        val_seqs, val_anns, val_meta = [], [], []
        if val_indices:
            val_seqs = [sequences_data[i] for i in val_indices]
            val_anns = [annotations_data[i] for i in val_indices]
            val_meta = [metadata_list[i] for i in val_indices]
            val_ds = DNADataset(val_seqs, val_anns, dna_tokenizer, text_tokenizer, ds_max_dna_len, ds_max_text_len, val_meta)
            val_dl = create_dataloader(val_ds, args.batch_size, args.num_workers, shuffle=False,
                                       dna_pad_id=dna_tokenizer.pad_token_id, text_pad_id=text_tokenizer.pad_token_id,
                                       pin_memory=(device.type=='cuda'), drop_last=False) # Do not drop_last for validation
        
        # Populate dataset_summary
        dataset_summary = {
            "total_samples_in_file": len(sequences_data),
            "training_samples": len(train_seqs),
            "validation_samples": len(val_seqs) if val_indices else 0,
            "dna_tokenizer_type": args.dna_tokenizer_type,
            "kmer_k_effective": dna_tokenizer.k if isinstance(dna_tokenizer, KmerDNATokenizer) else "N/A",
            "dna_vocab_size": dna_tokenizer.get_vocab_size(),
            "text_vocab_size": text_tokenizer.get_vocab_size(),
            "max_dna_len_config": args.max_dna_len,
            "max_text_len_config": args.max_text_len,
            "biotype_counts": Counter(m.get("biotype", "unknown") for m in metadata_list),
            "species_counts": Counter(m.get("species", "unknown") for m in metadata_list),
            "dna_lengths_raw": [len(s) for s in sequences_data], # Lengths before tokenization
            "text_lengths_raw_words": [len(a.split()) for a in annotations_data] # Approx word counts
        }
        if args.input_type == 'gff3':
            # Retrieve GFF feature counts stored by parse_gff3
            global dataset_summary_temp_storage
            if 'dataset_summary_temp_storage' in globals() and 'gff_feature_type_counts_raw' in dataset_summary_temp_storage:
                 dataset_summary['gff_feature_type_counts_raw'] = dataset_summary_temp_storage['gff_feature_type_counts_raw']
            # Count GFF feature types that were actually processed (after filtering by target_feature_types)
            dataset_summary['gff_feature_type_counts_processed'] = Counter(m.get("gff_feature_type", "unknown_gff_type") for m in metadata_list)


    elif args.resume_from_checkpoint: 
        logger.info("Resuming training. Tokenizers will be loaded based on checkpoint's context or --load paths.")
        checkpoint_dir = os.path.dirname(args.resume_from_checkpoint) # Assumes tokenizers are in same dir
        
        dna_tokenizer_cls_resume = KmerDNATokenizer if args.dna_tokenizer_type == "kmer" else CharDNATokenizer
        dna_tok_path_resume = args.load_dna_tokenizer_path or os.path.join(checkpoint_dir, "dna_tokenizer_vocab.json")
        if os.path.exists(dna_tok_path_resume):
            dna_tokenizer = dna_tokenizer_cls_resume.load_vocab(dna_tok_path_resume)
            if isinstance(dna_tokenizer, KmerDNATokenizer) and hasattr(dna_tokenizer, 'k'): args.kmer_k = dna_tokenizer.k
        else: logger.critical(f"DNA tokenizer not found at {dna_tok_path_resume} for resumed run. Exiting."); sys.exit(1)

        text_tok_path_resume = args.load_text_tokenizer_path or os.path.join(checkpoint_dir, "text_tokenizer_vocab.json")
        if os.path.exists(text_tok_path_resume):
            text_tokenizer = TextTokenizer.load_vocab(text_tok_path_resume)
        else: logger.critical(f"Text tokenizer not found at {text_tok_path_resume} for resumed run. Exiting."); sys.exit(1)
        
        if not args.input_file:
            logger.warning("Resuming without --input_file. DataLoaders (train_dl, val_dl) will be None. This is only suitable for loading a model for inference/export, not continued training.")
    else: # Should be caught by arg parsing logic earlier
        logger.critical("No input data source (file or checkpoint) and not a dummy run. Exiting."); sys.exit(1)

    if not dna_tokenizer or not text_tokenizer: # Should have been initialized by now
        logger.critical("Tokenizers not initialized. This indicates a logic error. Exiting."); sys.exit(1)

    # Model Initialization
    model_init_start_time = time.time()
    model = DNAClipModel(
        dna_vocab_size=dna_tokenizer.get_vocab_size(), 
        text_vocab_size=text_tokenizer.get_vocab_size(),
        embedding_dim=args.embedding_dim, 
        hidden_dim=args.hidden_dim, 
        num_layers=args.num_layers, 
        dropout=args.dropout, 
        initial_temperature=args.initial_temperature,
        dna_pad_idx=dna_tokenizer.pad_token_id,
        text_pad_idx=text_tokenizer.pad_token_id
    ).to(device)
    logger.info(f"Model initialized in {time.time() - model_init_start_time:.2f}s.")

    if args.use_torch_compile:
        if hasattr(torch, "compile") and sys.version_info >= (3,8): # torch.compile needs Py3.8+ generally
            logger.info("Applying torch.compile() to the model...")
            compile_start_time = time.time()
            try:
                # mode options: "default", "reduce-overhead", "max-autotune"
                model = torch.compile(model, mode="reduce-overhead") 
                logger.info(f"torch.compile() applied successfully in {time.time() - compile_start_time:.2f}s.")
            except Exception as e_compile:
                logger.warning(f"torch.compile() failed: {e_compile}. Proceeding without it.", exc_info=True)
        else:
            logger.warning("torch.compile requested but not available or fully supported in this PyTorch/Python version.")

    # Optimizer and Scheduler
    optimizer_params = {"lr": args.learning_rate}
    if args.optimizer_type == "AdamW": optimizer_params["weight_decay"] = args.weight_decay
    
    optimizer = getattr(optim, args.optimizer_type)(model.parameters(), **optimizer_params)
    scheduler = None
    if args.lr_scheduler_patience > 0 and val_dl : # Scheduler needs validation data
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_scheduler_factor, 
                                      patience=args.lr_scheduler_patience)
        logger.info(f"ReduceLROnPlateau learning rate scheduler active with patience {args.lr_scheduler_patience}.")
    elif args.lr_scheduler_patience > 0 and not val_dl:
        logger.warning("LR scheduler requested, but no validation data is available. Scheduler will be inactive.")

    # AMP GradScaler
    scaler = None
    if args.use_amp and device.type == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
        logger.info("CUDA AMP GradScaler initialized.")

    # Load from checkpoint or pre-trained weights
    if args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
        try:
            checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            if 'optimizer_state_dict' in checkpoint: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            start_epoch = checkpoint.get('epoch', -1) + 1 # Resume from next epoch
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            
            # Restore metrics history
            chkpt_metrics_history = checkpoint.get('metrics_history', {})
            for key, val_list in chkpt_metrics_history.items():
                if key == 'plot_paths': # defaultdict of lists
                    for sub_key, paths in val_list.items(): metrics_history[key][sub_key].extend(paths)
                elif key == 'knn_classification_reports_biotype': # list of dicts
                    metrics_history[key].extend(val_list)
                else: # simple list of values
                    metrics_history[key].extend(val_list)
            
            if scheduler and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict']:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            # If resuming without new input, try to load dataset_summary from checkpoint
            if not args.input_file and 'dataset_summary' in checkpoint:
                dataset_summary = checkpoint['dataset_summary']

            current_lr_resumed = optimizer.param_groups[0]['lr'] if 'optimizer_state_dict' in checkpoint else "N/A (Optimizer not loaded from chkpt)"
            logger.info(f"Resumed from epoch {start_epoch-1}. Best validation loss: {best_val_loss:.4f}. Current LR: {current_lr_resumed}")

        except Exception as e_chkpt_load:
            logger.error(f"Failed to load checkpoint from {args.resume_from_checkpoint}: {e_chkpt_load}. Starting fresh training.", exc_info=True)
            start_epoch = 0; best_val_loss = float('inf'); metrics_history = defaultdict(list); metrics_history['plot_paths'] = defaultdict(list); metrics_history['knn_classification_reports_biotype'] = [] # Reset
    
    elif args.load_model_path: # Load only model weights, not full training state
        logger.info(f"Loading model weights from: {args.load_model_path}")
        try:
            model.load_state_dict(torch.load(args.load_model_path, map_location=device), strict=True)
            logger.info("Model weights loaded successfully (strict=True).")
        except RuntimeError as e_strict_load: # If strict loading fails (e.g. different architecture)
            logger.warning(f"Strict loading of model weights failed: {e_strict_load}. Attempting non-strict loading (strict=False)...")
            try:
                model.load_state_dict(torch.load(args.load_model_path, map_location=device), strict=False)
                logger.info("Model weights loaded successfully (strict=False). Some layers might be unmatched or differently sized.")
            except Exception as e_non_strict_load:
                logger.error(f"Non-strict loading of model weights also failed: {e_non_strict_load}. Model will use fresh random weights.", exc_info=True)

    # --- Training Loop ---
    if not train_dl:
        logger.warning("Training DataLoader (train_dl) is not available. Skipping training loop.")
    else:
        logger.info(f"Starting training from epoch {start_epoch + 1} up to {args.epochs} epochs...")
        training_loop_start_time = time.time()
        best_epoch_idx = metrics_history.get('best_epoch_idx', -1) # From resumed history if any

        for epoch in range(start_epoch, args.epochs):
            metrics_history['num_epochs_trained'] = epoch + 1 # Record actual number of epochs run so far
            
            avg_train_loss, avg_temp, avg_grad_norm = train_epoch(
                model, train_dl, optimizer, device, epoch, args.epochs, 
                use_amp=(args.use_amp and device.type=='cuda'), scaler=scaler
            )
            metrics_history['train_loss'].append(avg_train_loss)
            metrics_history['temperature'].append(avg_temp)
            metrics_history['avg_grad_norm'].append(avg_grad_norm)
            metrics_history['learning_rate'].append(optimizer.param_groups[0]['lr'])

            logger.info(f"Epoch {epoch+1}/{args.epochs} Summary - Train Loss: {avg_train_loss:.4f}, Avg Temp: {avg_temp:.4f}, Avg Grad Norm: {avg_grad_norm:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")

            current_val_loss = float('inf') # Default if no validation
            if val_dl:
                val_epoch_metrics, dna_val_embeds, _, biotypes_val, species_val, _, _ = evaluate(
                    model, val_dl, device, report_dir, epoch, metrics_history, is_final_eval=False
                )
                current_val_loss = val_epoch_metrics.get("avg_loss", float('inf'))

                if not (np.isnan(current_val_loss) or np.isinf(current_val_loss)):
                    if scheduler: scheduler.step(current_val_loss) # LR scheduler step
                    
                    if current_val_loss < best_val_loss:
                        best_val_loss = current_val_loss
                        best_epoch_idx = epoch # Update best epoch index
                        metrics_history['best_epoch_idx'] = epoch
                        metrics_history['best_epoch_val_loss'] = best_val_loss
                        
                        best_model_checkpoint_path = os.path.join(args.output_dir, "best_model_checkpoint.pth")
                        checkpoint_data_to_save = {
                            'epoch': epoch, 'model_state_dict': model.state_dict(), 
                            'optimizer_state_dict': optimizer.state_dict(), 
                            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                            'scaler_state_dict': scaler.state_dict() if scaler else None,
                            'best_val_loss': best_val_loss, 'args': vars(args), 
                            'metrics_history': dict(metrics_history), # Save current state of metrics
                            'dataset_summary': dataset_summary # Save summary for context
                        }
                        try:
                            torch.save(checkpoint_data_to_save, best_model_checkpoint_path)
                        except Exception as e_save_best_chkpt:
                            logger.error(f"Failed to save best model checkpoint: {e_save_best_chkpt}", exc_info=True)
                        else:
                            logger.info(f"New best model checkpoint saved (Validation Loss: {best_val_loss:.4f} at Epoch {epoch+1}) to {best_model_checkpoint_path}")
                        epochs_without_improvement = 0

                        # Plot embeddings for the best model epoch
                        if dna_val_embeds is not None and dna_val_embeds.shape[0] > 1:
                            for plot_method_name in args.plot_methods:
                                if plot_method_name == "umap" and not umap: continue # Skip if UMAP not installed
                                
                                plot_filename_bio = f"dna_embeddings_{plot_method_name}_biotype_epoch_{epoch+1}.png"
                                plot_filepath_bio = os.path.join(report_dir, plot_filename_bio)
                                if plot_embeddings(dna_val_embeds, biotypes_val, f"DNA Embeddings ({plot_method_name.upper()}) by Biotype (Epoch {epoch+1})", plot_filepath_bio, method=plot_method_name):
                                    metrics_history['plot_paths'][f"embedding_plot_{plot_method_name}_biotype"].append(plot_filename_bio)

                                plot_filename_sp = f"dna_embeddings_{plot_method_name}_species_epoch_{epoch+1}.png"
                                plot_filepath_sp = os.path.join(report_dir, plot_filename_sp)
                                if plot_embeddings(dna_val_embeds, species_val, f"DNA Embeddings ({plot_method_name.upper()}) by Species (Epoch {epoch+1})", plot_filepath_sp, method=plot_method_name):
                                    metrics_history['plot_paths'][f"embedding_plot_{plot_method_name}_species"].append(plot_filename_sp)
                    else: # Validation loss did not improve
                        epochs_without_improvement += 1
                        logger.info(f"Validation loss did not improve for {epochs_without_improvement} epoch(s). Best validation loss remains: {best_val_loss:.4f} (from Epoch {best_epoch_idx+1})")
                else: # NaN/Inf validation loss
                    epochs_without_improvement += 1
                    logger.warning(f"Validation loss was NaN/Inf for epoch {epoch+1}. Counting as no improvement.")

                # Early stopping check
                if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without validation loss improvement.")
                    break # Exit training loop
            
            # Periodic checkpoint saving (e.g., every 5 epochs or last epoch)
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                periodic_checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pth")
                periodic_checkpoint_data = {
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                    'best_val_loss': best_val_loss, 'args': vars(args),
                    'metrics_history': dict(metrics_history), 
                    'dataset_summary': dataset_summary
                }
                try:
                    torch.save(periodic_checkpoint_data, periodic_checkpoint_path)
                except Exception as e_save_periodic_chkpt:
                     logger.error(f"Failed to save periodic checkpoint: {e_save_periodic_chkpt}", exc_info=True)
                else:
                    logger.info(f"Periodic checkpoint saved: {periodic_checkpoint_path}")
        
        logger.info(f"Training loop finished in {time.time() - training_loop_start_time:.2f}s.")

    # --- Final Evaluation and Export ---
    final_ops_start_time = time.time()
    best_epoch_idx = metrics_history.get('best_epoch_idx', -1) # Ensure it's up-to-date

    final_model_to_use_path = os.path.join(args.output_dir, "best_model_checkpoint.pth")
    descriptor_for_final_eval = f"Best Model (Epoch {best_epoch_idx+1} - Val Loss: {best_val_loss:.4f})" \
                                if best_epoch_idx != -1 and os.path.exists(final_model_to_use_path) \
                                else "Last Trained Model"

    if not (os.path.exists(final_model_to_use_path) and best_epoch_idx != -1):
        # Fallback to last completed epoch if best_model_checkpoint is not available
        last_completed_epoch = metrics_history.get('num_epochs_trained', start_epoch) 
        last_epoch_chkpt_num = last_completed_epoch if last_completed_epoch > 0 else start_epoch # Use start_epoch if no training happened
        
        final_model_to_use_path = os.path.join(args.output_dir, f"checkpoint_epoch_{last_epoch_chkpt_num}.pth")
        descriptor_for_final_eval = f"Model from Epoch {last_epoch_chkpt_num}"
        if not os.path.exists(final_model_to_use_path):
            final_model_to_use_path = None # No checkpoint found
            descriptor_for_final_eval = "Current In-Memory Model (No Checkpoint Loaded for Final Eval)"
    
    if final_model_to_use_path and os.path.exists(final_model_to_use_path):
        logger.info(f"Loading model from {final_model_to_use_path} for final evaluation and export ({descriptor_for_final_eval}).")
        try:
            final_checkpoint = torch.load(final_model_to_use_path, map_location=device)
            model.load_state_dict(final_checkpoint['model_state_dict'])
            # If dataset_summary wasn't populated (e.g. resume without input_file), try to get it from checkpoint
            if not dataset_summary and 'dataset_summary' in final_checkpoint : 
                dataset_summary = final_checkpoint['dataset_summary']
        except Exception as e_final_load:
            logger.error(f"Failed to load model from {final_model_to_use_path} for final ops: {e_final_load}. Using current in-memory model state.", exc_info=True)
            descriptor_for_final_eval = "Current In-Memory Model (Fallback due to load error)"
    else:
        logger.info(f"Using {descriptor_for_final_eval} (current in-memory state) for final evaluation and export.")
    
    model.eval() # Ensure model is in evaluation mode

    # Determine DataLoader for final evaluation (validation set if available, else training set)
    final_evaluation_dl = val_dl if val_dl else train_dl
    final_eval_dataset_name = "Validation Set" if val_dl else "Training Set (used for final eval)"
    
    final_dna_embeddings, final_text_embeddings, final_similarity_matrix_output = None, None, None

    if not final_evaluation_dl:
        logger.warning("No DataLoader available for final evaluation or embedding export. Skipping these steps.")
    else:
        logger.info(f"--- Performing Final Evaluation on {final_eval_dataset_name} using {descriptor_for_final_eval} ---")
        # Use a fresh metrics_agg for final eval to not mix with training history directly for this specific call
        final_eval_metrics_agg = defaultdict(list)
        final_eval_metrics_agg['plot_paths'] = defaultdict(list) # Needed by evaluate
        final_eval_metrics_agg['knn_classification_reports_biotype'] = [] # Needed by evaluate

        # Determine epoch number for plot filenames (best epoch or last trained epoch)
        epoch_for_final_eval_plots = best_epoch_idx if "Best Model" in descriptor_for_final_eval and best_epoch_idx != -1 \
                                     else metrics_history.get('num_epochs_trained', 0) -1 
        epoch_for_final_eval_plots = max(0, epoch_for_final_eval_plots) # Ensure non-negative

        _, final_dna_embeddings, final_text_embeddings, final_biotypes, final_species, final_raw_seqs, final_raw_anns = evaluate(
            model, final_evaluation_dl, device, report_dir, 
            epoch=epoch_for_final_eval_plots, # For consistent plot naming if this is the best epoch
            metrics_agg=final_eval_metrics_agg, # Use a temporary agg for this specific eval run
            is_final_eval=True # Enable full data collection
        )
        # Note: final_eval_metrics_agg will contain metrics for this run.
        # The main metrics_history is what's used for the report trends.
        # The evaluate function will save CM plots if conditions met.

        if final_dna_embeddings is not None and final_text_embeddings is not None:
            final_similarity_matrix_output = final_dna_embeddings @ final_text_embeddings.T # For report snippet

        # Export Embeddings
        if args.export_embeddings_format != "none" and final_dna_embeddings is not None and final_text_embeddings is not None:
            logger.info(f"Exporting {final_dna_embeddings.shape[0]} DNA and Text embeddings from {final_eval_dataset_name}...")
            num_embeddings = final_dna_embeddings.shape[0]
            
            # Make sure all lists are sliced to num_embeddings if final_evaluation_dl was subset/smaller
            final_ds_metadata = final_evaluation_dl.dataset.metadata[:num_embeddings]

            embeddings_df_data = {
                "id": [m.get("id", f"item_{i}") for i,m in enumerate(final_ds_metadata)],
                "biotype": final_biotypes[:num_embeddings], 
                "species": final_species[:num_embeddings],
                "raw_sequence": final_raw_seqs[:num_embeddings],
                "raw_annotation": final_raw_anns[:num_embeddings]
            }
            for i_emb_dim in range(final_dna_embeddings.shape[1]):
                embeddings_df_data[f"dna_emb_{i_emb_dim}"] = final_dna_embeddings[:, i_emb_dim]
            for i_emb_dim in range(final_text_embeddings.shape[1]):
                embeddings_df_data[f"text_emb_{i_emb_dim}"] = final_text_embeddings[:, i_emb_dim]
            
            embeddings_df = pd.DataFrame(embeddings_df_data)

            if args.export_embeddings_format in ["csv", "both"]:
                csv_filepath = os.path.join(args.output_dir, "final_embeddings.csv")
                embeddings_df.to_csv(csv_filepath, index=False)
                logger.info(f"Embeddings exported in CSV format to: {csv_filepath}")
            if args.export_embeddings_format in ["parquet", "both"]:
                if pl: # Polars is available
                    parquet_filepath = os.path.join(args.output_dir, "final_embeddings.parquet")
                    try:
                        pl_df = pl.from_pandas(embeddings_df)
                        pl_df.write_parquet(parquet_filepath)
                        logger.info(f"Embeddings exported in Parquet format to: {parquet_filepath}")
                    except Exception as e_parquet:
                        logger.error(f"Failed to export embeddings to Parquet: {e_parquet}", exc_info=True)
                else: # Polars not available
                    logger.warning("Polars library not installed. Skipping Parquet export for embeddings.")
        else:
            logger.info("Skipping embedding export (format='none' or embeddings not available).")

    # ONNX and TF.js Export
    # Use dummy input shapes based on max_len settings or defaults if not available
    export_dummy_dna_len = ds_max_dna_len if 'ds_max_dna_len' in locals() and ds_max_dna_len is not None else args.max_dna_len if args.max_dna_len > 0 else 128
    export_dummy_text_len = ds_max_text_len if 'ds_max_text_len' in locals() and ds_max_text_len is not None else args.max_text_len if args.max_text_len > 0 else 64
    
    dummy_dna_tokens_export = torch.randint(0, dna_tokenizer.get_vocab_size(), (1, export_dummy_dna_len), dtype=torch.long).to(device)
    dummy_dna_lengths_export = torch.tensor([export_dummy_dna_len // 2], dtype=torch.long).cpu() # Lengths on CPU for ONNX export
    dummy_text_tokens_export = torch.randint(0, text_tokenizer.get_vocab_size(), (1, export_dummy_text_len), dtype=torch.long).to(device)
    dummy_text_lengths_export = torch.tensor([export_dummy_text_len // 2], dtype=torch.long).cpu() # Lengths on CPU

    onnx_export_paths = {}
    if args.export_onnx:
        if not onnx: logger.warning("ONNX library not installed. Skipping ONNX export.")
        else:
            logger.info(f"Exporting model encoders to ONNX format (using model: {descriptor_for_final_eval})...");
            onnx_export_start_time = time.time()
            dna_encoder_onnx_path = os.path.join(args.output_dir, "dna_encoder.onnx")
            try:
                torch.onnx.export(model.dna_encoder, 
                                  (dummy_dna_tokens_export, dummy_dna_lengths_export), 
                                  dna_encoder_onnx_path, 
                                  input_names=['dna_tokens', 'dna_lengths'], 
                                  output_names=['dna_embedding'], 
                                  dynamic_axes={'dna_tokens': {0: 'batch_size', 1: 'dna_sequence_length'}, 
                                                'dna_lengths': {0: 'batch_size'}, 
                                                'dna_embedding': {0: 'batch_size'}}, 
                                  opset_version=ONNX_OPSET_VERSION, export_params=True, do_constant_folding=True)
                logger.info(f"DNA encoder exported to ONNX: {dna_encoder_onnx_path}")
                onnx_export_paths["dna_encoder"] = dna_encoder_onnx_path
            except Exception as e_onnx_dna:
                logger.error(f"Failed to export DNA encoder to ONNX: {e_onnx_dna}", exc_info=True)

            text_encoder_onnx_path = os.path.join(args.output_dir, "text_encoder.onnx")
            try:
                torch.onnx.export(model.text_encoder, 
                                  (dummy_text_tokens_export, dummy_text_lengths_export), 
                                  text_encoder_onnx_path, 
                                  input_names=['text_tokens', 'text_lengths'], 
                                  output_names=['text_embedding'], 
                                  dynamic_axes={'text_tokens': {0: 'batch_size', 1: 'text_sequence_length'}, 
                                                'text_lengths': {0: 'batch_size'}, 
                                                'text_embedding': {0: 'batch_size'}}, 
                                  opset_version=ONNX_OPSET_VERSION, export_params=True, do_constant_folding=True)
                logger.info(f"Text encoder exported to ONNX: {text_encoder_onnx_path}")
                onnx_export_paths["text_encoder"] = text_encoder_onnx_path
            except Exception as e_onnx_text:
                logger.error(f"Failed to export text encoder to ONNX: {e_onnx_text}", exc_info=True)
            logger.info(f"ONNX export finished in {time.time() - onnx_export_start_time:.2f}s.")

    if args.export_tfjs:
        if not args.export_onnx:
            logger.warning("TensorFlow.js export requested, but --export_onnx was not specified. Skipping TF.js export.")
        elif not onnx_export_paths: # If ONNX export failed or was skipped
            logger.warning("ONNX export seems to have failed or was skipped. Cannot proceed with TensorFlow.js export.")
        else:
            logger.info(f"Exporting ONNX models to TensorFlow.js format (using model: {descriptor_for_final_eval})...")
            if "dna_encoder" in onnx_export_paths:
                tfjs_dna_output_dir = os.path.join(args.output_dir, "tfjs_dna_encoder")
                export_onnx_to_tfjs(onnx_export_paths["dna_encoder"], tfjs_dna_output_dir)
            if "text_encoder" in onnx_export_paths:
                tfjs_text_output_dir = os.path.join(args.output_dir, "tfjs_text_encoder")
                export_onnx_to_tfjs(onnx_export_paths["text_encoder"], tfjs_text_output_dir)
    
    logger.info(f"Final evaluation and export operations finished in {time.time() - final_ops_start_time:.2f}s.")

    # Final Reporting
    # Ensure dataset_summary is available for the report
    if not dataset_summary and args.resume_from_checkpoint and os.path.exists(args.resume_from_checkpoint):
        # Try to load from the specific checkpoint used for resuming if dataset_summary is still empty
        chkpt_for_ds_summary = torch.load(args.resume_from_checkpoint, map_location='cpu')
        if 'dataset_summary' in chkpt_for_ds_summary:
            dataset_summary = chkpt_for_ds_summary['dataset_summary']
        else: # Fallback if not in checkpoint
            dataset_summary = {"info": "Dataset summary not available from checkpoint or current run."}
    elif not dataset_summary: # Fallback if no input file and no resume with summary
        dataset_summary = {"info": "Dataset summary not generated (e.g., no input data processed)."}

    generate_report(args, metrics_history, report_dir, dataset_summary, final_similarity_matrix_output, descriptor_for_final_eval)

    if args.run_dummy_example:
        logger.info(f"Dummy example run completed. Dummy input was: {args.input_file}. Dummy outputs are in: {args.output_dir}.")
        logger.info(f"To clean up, you can remove the base directory used for dummy data: {os.path.dirname(args.input_file)}")
    
    logger.info(f"All processing finished in {time.time() - main_start_time:.2f}s. Main outputs are in: {args.output_dir}")


# --- Unit Tests ---
class TestDNAClipComponents(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="dnaclip_test_")
        self.kmer_dna_k = 3
        # Vocab where PAD=0, UNK=1 is enforced by BaseTokenizer
        self.kmer_dna_vocab_dict = {"ATG":2, "CGC":3, "TAA":4} # User-provided part
        self.kmer_dna_tokenizer = KmerDNATokenizer(k=self.kmer_dna_k, vocab=self.kmer_dna_vocab_dict)
        
        self.char_dna_tokenizer = CharDNATokenizer() # Uses default vocab
        
        self.text_vocab_dict = {"gene":2, "is":3, "active":4} # User-provided part
        self.text_tokenizer = TextTokenizer(vocab=self.text_vocab_dict)

        self.dummy_fasta_content_braces = """>s1|species=H|biotype=G\n{A*5}{C*3}\n>s2 description\n{G*4}X{T*2}\n"""
        # Corrected expanded content: X is invalid, GGGGXT becomes GGGGT if X is filtered by kmer tokenizer
        # Kmer tokenizer allows N, but X is not N. So if ATGxNGC, it takes ATG, then if k=3, it might skip xNG if x invalidates it.
        # Let's assume GGGGXT -> GGGGT, if k=3 -> GGG, GGT.
        self.dummy_fasta_content_expanded = """>s1|species=H|biotype=G\nAAAAACCC\n>s2 description\nGGGGXT\n""" 
        self.test_fasta_path = os.path.join(self.temp_dir, "test_data.fasta")
        with open(self.test_fasta_path, "w", encoding='utf-8') as f: f.write(self.dummy_fasta_content_expanded)

        self.dummy_ref_fasta_content = ">chrTest1|species=TestSpecies|biotype=RefContig\nATGCGTATGCGTNNNATGCGTATGCGT\n>chrTest2\nCCCCCCCCCCCCCCCCCCCCCCCCC" # chrTest1 len 25
        self.test_ref_fasta_path = os.path.join(self.temp_dir, "test_ref.fasta")
        with open(self.test_ref_fasta_path, "w", encoding='utf-8') as f: f.write(self.dummy_ref_fasta_content)
        
        self.dummy_bed_content = "chrTest1\t0\t6\tfeat1\t100\t+\t.\tgene_id=g1;biotype=coding_bed\nchrTest1\t10\t15\tfeat2_coords_are_valid\nchrTestNonExistent\t0\t5\tfeat3_bad_chrom"
        self.test_bed_path = os.path.join(self.temp_dir, "test_data.bed")
        with open(self.test_bed_path, "w", encoding='utf-8') as f: f.write(self.dummy_bed_content)

        self.dummy_gff_content = """##gff-version 3
chrTest1\tTEST\tgene\t1\t6\t.\t+\t.\tID=gene1;Name=TestGene1;biotype=protein_coding_gff
chrTest1\tTEST\tmRNA\t10\t15\t.\t-\t.\tID=rna1;Parent=gene1;description=Test RNA on minus strand
chrTestNonExistent\tTEST\tgene\t1\t5\t.\t+\t.\tID=gene2;Name=BadChromGene
"""
        self.test_gff_path = os.path.join(self.temp_dir, "test_data.gff3")
        with open(self.test_gff_path, "w", encoding='utf-8') as f: f.write(self.dummy_gff_content)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def tearDown(self): shutil.rmtree(self.temp_dir)

    def test_kmer_dna_tokenizer(self):
        self.assertEqual(self.kmer_dna_tokenizer.tokenize("ATGCGC"), ["ATG", "TGC", "GCG", "CGC"]) # TGC, GCG will be UNK
        self.assertEqual(self.kmer_dna_tokenizer.tokenize("AT"), [UNK_TOKEN]) 
        # Kmer tokenizer's `all(c in VALID_DNA_CHARS_SET for c in kmer)` means 'ATG' is valid, 'TGX' is not if X not in set.
        # If sequence is ATGXCGC, and k=3: ATG (ok), TGX (skip if X invalid), GXC (skip if X invalid), XCGC (skip if X invalid), CGC (ok)
        # Current implementation: If "ATGxCGC", it would be ["ATG"] then "xCGC" is scanned. "xCG" fails, "CGC" is valid. So ["ATG", "CGC"]
        self.assertEqual(self.kmer_dna_tokenizer.tokenize("ATGxCGC"), ["ATG", "CGC"])
        self.assertEqual(self.kmer_dna_tokenizer.tokenize(""), [UNK_TOKEN]) 
        
        ids, length = self.kmer_dna_tokenizer.encode("ATGCGC", max_length=5)
        # Expected: ATG (2), TGC (UNK=1), GCG (UNK=1), CGC (3), PAD (0)
        self.assertEqual(ids, [self.kmer_dna_tokenizer.vocab["ATG"], 
                               self.kmer_dna_tokenizer.unk_token_id, 
                               self.kmer_dna_tokenizer.unk_token_id, 
                               self.kmer_dna_tokenizer.vocab["CGC"], 
                               self.kmer_dna_tokenizer.pad_token_id])
        self.assertEqual(length, 4) # Effective length is number of actual kmers before padding
        
        ids_trunc, length_trunc = self.kmer_dna_tokenizer.encode("ATGCGCTAA", max_length=2) # Kmers: ATG, TGC, GCG, CGC, GCT, CTA, TAA
        # Encoded (first 2): ATG (2), TGC (UNK=1)
        self.assertEqual(ids_trunc, [self.kmer_dna_tokenizer.vocab["ATG"], self.kmer_dna_tokenizer.unk_token_id]) 
        self.assertEqual(length_trunc, 2) # Max length truncates tokens

        ids_short, length_short = self.kmer_dna_tokenizer.encode("AT", max_length=3) # "AT" -> [UNK_TOKEN]
        # Encoded: UNK (1), PAD (0), PAD (0)
        self.assertEqual(ids_short, [self.kmer_dna_tokenizer.unk_token_id, self.kmer_dna_tokenizer.pad_token_id, self.kmer_dna_tokenizer.pad_token_id])
        self.assertEqual(length_short, 1) # Length of [UNK_TOKEN] is 1

    def test_char_dna_tokenizer(self):
        self.assertEqual(self.char_dna_tokenizer.tokenize("ATGC"), ['A', 'T', 'G', 'C'])
        self.assertEqual(self.char_dna_tokenizer.tokenize("ATGx C"), ['A', 'T', 'G', 'C']) # x and space are filtered
        self.assertEqual(self.char_dna_tokenizer.tokenize(""), [UNK_TOKEN])
        
        ids, length = self.char_dna_tokenizer.encode("ATGN", max_length=5)
        # DEFAULT_CHAR_DNA_VOCAB: {PAD_TOKEN: 0, UNK_TOKEN: 1, 'A': 2, 'C': 3, 'G': 4, 'T': 5, 'N': 6}
        expected_ids = [DEFAULT_CHAR_DNA_VOCAB['A'], DEFAULT_CHAR_DNA_VOCAB['T'], 
                        DEFAULT_CHAR_DNA_VOCAB['G'], DEFAULT_CHAR_DNA_VOCAB['N'], 
                        self.char_dna_tokenizer.pad_token_id]
        self.assertEqual(ids, expected_ids)
        self.assertEqual(length, 4)
        
        ids_trunc, length_trunc = self.char_dna_tokenizer.encode("ATGCGT", max_length=3)
        self.assertEqual(ids_trunc, [DEFAULT_CHAR_DNA_VOCAB['A'], DEFAULT_CHAR_DNA_VOCAB['T'], DEFAULT_CHAR_DNA_VOCAB['G']])
        self.assertEqual(length_trunc, 3)

    def test_text_tokenizer(self):
        self.assertEqual(self.text_tokenizer.tokenize("Gene is active!"), ["gene", "is", "active"]) # ! removed
        self.assertEqual(self.text_tokenizer.tokenize("  Multiple   spaces  "), ["multiple", "spaces"])
        self.assertEqual(self.text_tokenizer.tokenize(""), [UNK_TOKEN])
        
        ids, length = self.text_tokenizer.encode("Gene is active", max_length=4)
        # self.text_vocab_dict before BaseTokenizer: {"gene":2, "is":3, "active":4}
        # BaseTokenizer ensures PAD=0, UNK=1. If user indices don't clash, they are kept.
        # So, actual vocab values would be: "gene":2, "is":3, "active":4
        expected_ids = [self.text_tokenizer.vocab["gene"], self.text_tokenizer.vocab["is"], 
                        self.text_tokenizer.vocab["active"], self.text_tokenizer.pad_token_id]
        self.assertEqual(ids, expected_ids)
        self.assertEqual(length, 3)

    def test_vocab_building(self):
        dna_texts = ["ATGCG", "CGCGCG", "ATATAT"] # k=3: ATG,TGC,GCG; CGC,GCG,CGC; ATA,TAT,ATA
        kmer_tok_built = build_vocab_from_data(dna_texts, KmerDNATokenizer, min_freq=1, tokenizer_args={'k': 3})
        # Expected kmers: ATG, TGC, GCG, CGC, ATA, TAT. Plus PAD, UNK. Total 8.
        self.assertEqual(len(kmer_tok_built.vocab), 6 + 2) 
        self.assertIn("ATG", kmer_tok_built.vocab)
        
        char_tok_built = build_vocab_from_data(dna_texts, CharDNATokenizer, min_freq=1) # Uses fixed default vocab
        self.assertEqual(len(char_tok_built.vocab), len(DEFAULT_CHAR_DNA_VOCAB))
        self.assertIn('A', char_tok_built.vocab)

    def test_dataset_item_creation(self):
        seqs = ["ATGCGTAGC", "TTT"]; anns = ["active gene", "inactive"]; metadata = [{"id":"s1"}, {"id":"s2"}]
        # Kmer tokenizer with k=3. "ATGCGTAGC" -> ATG,TGC,GCG,CGT,GTA,TAG,AGC (7 kmers)
        dataset_kmer = DNADataset(seqs, anns, self.kmer_dna_tokenizer, self.text_tokenizer, 
                                  max_dna_len=5, max_text_len=3, metadata=metadata)
        item0_kmer = dataset_kmer[0] # "ATGCGTAGC" (7 kmers) -> truncated to 5 kmers. "active gene" (2 tokens)
        self.assertEqual(item0_kmer['dna_tokens'].shape, torch.Size([5])); self.assertEqual(item0_kmer['dna_lengths'].item(), 5)
        self.assertEqual(item0_kmer['text_tokens'].shape, torch.Size([3])); self.assertEqual(item0_kmer['text_lengths'].item(), 2)
        self.assertTrue(item0_kmer['text_tokens'][-1] == self.text_tokenizer.pad_token_id)
        
        # Char tokenizer. "TTT" (3 chars). "inactive" (1 token)
        dataset_char = DNADataset(seqs, anns, self.char_dna_tokenizer, self.text_tokenizer,
                                  max_dna_len=10, max_text_len=4, metadata=metadata)
        item1_char = dataset_char[1] 
        self.assertEqual(item1_char['dna_tokens'].shape, torch.Size([10])); self.assertEqual(item1_char['dna_lengths'].item(), 3) 
        self.assertTrue(all(t == self.char_dna_tokenizer.pad_token_id for t in item1_char['dna_tokens'][3:].tolist()))
        self.assertEqual(item1_char['text_tokens'].shape, torch.Size([4])); self.assertEqual(item1_char['text_lengths'].item(), 1) 

    def test_custom_collate_fn(self):
        batch_items_for_collate = [
            {"dna_tokens": torch.tensor([2,3,4]), "dna_lengths": torch.tensor(3), 
             "text_tokens": torch.tensor([2,3]), "text_lengths": torch.tensor(2), "metadata":{}},
            {"dna_tokens": torch.tensor([5]), "dna_lengths": torch.tensor(1), 
             "text_tokens": torch.tensor([4,5,6,7]), "text_lengths": torch.tensor(4), "metadata":{}}
        ]
        # Add other required keys that are not tensors (will be listified by collate)
        for item in batch_items_for_collate:
            item.update({"raw_sequence": "seq", "raw_annotation": "ann", "raw_sequence_len":1, "raw_annotation_len":1})

        collated = custom_collate_fn(batch_items_for_collate, 
                                     self.kmer_dna_tokenizer.pad_token_id, 
                                     self.text_tokenizer.pad_token_id)
        
        self.assertEqual(collated['dna_tokens'].shape, (2, 3))
        self.assertTrue(torch.equal(collated['dna_tokens'][1], torch.tensor([5, self.kmer_dna_tokenizer.pad_token_id, self.kmer_dna_tokenizer.pad_token_id])))
        
        self.assertEqual(collated['text_tokens'].shape, (2, 4))
        self.assertTrue(torch.equal(collated['text_tokens'][0], torch.tensor([2,3, self.text_tokenizer.pad_token_id, self.text_tokenizer.pad_token_id])))
        
        self.assertTrue(torch.equal(collated['dna_lengths'], torch.tensor([3,1])))
        self.assertTrue(torch.equal(collated['text_lengths'], torch.tensor([2,4])))

    def test_sequence_encoder_forward(self):
        vocab_size = 20; embed_dim = 16; hidden_dim = 24; num_layers = 1; pad_idx = 0
        encoder = SequenceEncoder(vocab_size, embed_dim, hidden_dim, num_layers, pad_idx=pad_idx).to(self.device)
        
        tokens = torch.tensor([[1,2,3,4,5], [6,7,8,pad_idx,pad_idx]], dtype=torch.long).to(self.device)
        lengths = torch.tensor([5, 3], dtype=torch.long) # Lengths must be on CPU for pack_padded_sequence
        
        output = encoder(tokens, lengths.cpu())
        self.assertEqual(output.shape, (2, embed_dim))
        self.assertFalse(torch.isnan(output).any())

        # Test with all padding (except one token to avoid empty sequence issues with pack_padded)
        tokens_all_pad = torch.tensor([[1,pad_idx,pad_idx],[6,pad_idx,pad_idx]], dtype=torch.long).to(self.device)
        lengths_all_pad = torch.tensor([1,1], dtype=torch.long) # Effective length 1 for each
        output_all_pad = encoder(tokens_all_pad, lengths_all_pad.cpu())
        self.assertEqual(output_all_pad.shape, (2, embed_dim))
        self.assertFalse(torch.isnan(output_all_pad).any())

    def test_dnaclipmodel_forward(self):
        dna_vocab_sz = self.kmer_dna_tokenizer.get_vocab_size()
        text_vocab_sz = self.text_tokenizer.get_vocab_size()
        embed_dim, hidden_dim, num_layers = 16, 24, 1
        
        model = DNAClipModel(dna_vocab_sz, text_vocab_sz, embed_dim, hidden_dim, num_layers,
                             dna_pad_idx=self.kmer_dna_tokenizer.pad_token_id,
                             text_pad_idx=self.text_tokenizer.pad_token_id).to(self.device)
        
        bs = 2; max_dna_len, max_text_len = 10, 8
        dna_toks = torch.randint(0, dna_vocab_sz, (bs, max_dna_len), dtype=torch.long).to(self.device)
        dna_lens = torch.tensor([max_dna_len, max_dna_len // 2], dtype=torch.long) # CPU
        text_toks = torch.randint(0, text_vocab_sz, (bs, max_text_len), dtype=torch.long).to(self.device)
        text_lens = torch.tensor([max_text_len // 2, max_text_len], dtype=torch.long) # CPU
        
        logits_dna, logits_text, temp = model(dna_toks, dna_lens.cpu(), text_toks, text_lens.cpu())
        self.assertEqual(logits_dna.shape, (bs, bs))
        self.assertEqual(logits_text.shape, (bs, bs))
        self.assertTrue(temp.item() > 0)

    def test_contrastive_loss(self):
        bs = 4
        sim_good = torch.eye(bs, device=self.device) * 10 # High diagonal similarity
        loss_good = contrastive_loss(sim_good, sim_good.t(), self.device)
        self.assertTrue(loss_good.item() < 1.0) 
        
        sim_bad = torch.rand(bs, bs, device=self.device) # Random similarity
        loss_bad = contrastive_loss(sim_bad, sim_bad.t(), self.device)
        self.assertTrue(loss_bad.item() > 0) 
        
        # Test with empty batch (should return 0)
        loss_empty = contrastive_loss(torch.empty(0,0, device=self.device), torch.empty(0,0, device=self.device), self.device)
        self.assertEqual(loss_empty.item(), 0.0)

    def test_calculate_mrr(self):
        sim_matrix_perfect = np.array([[1.0, 0.1, 0.2], [0.1, 1.0, 0.3], [0.2, 0.3, 1.0]])
        mrr_d2t_p, mrr_t2d_p = calculate_mrr(sim_matrix_perfect)
        self.assertAlmostEqual(mrr_d2t_p, 1.0)
        self.assertAlmostEqual(mrr_t2d_p, 1.0)
        
        sim_matrix_worst = np.array([[0.1, 0.8, 0.9], [0.8, 0.1, 0.7], [0.9, 0.7, 0.1]]) # Correct target is rank 3
        mrr_d2t_w, mrr_t2d_w = calculate_mrr(sim_matrix_worst)
        self.assertAlmostEqual(mrr_d2t_w, 1/3)
        self.assertAlmostEqual(mrr_t2d_w, 1/3)
        
        # Varied ranks:
        # D0 -> T0 (rank 1)
        # D1 -> T2 (rank 1), T1 is D1 (rank 3)
        # D2 -> T1 (rank 1), T2 is D2 (rank 2)
        sim_matrix_ranks_varied = np.array([[0.9,0.1,0.0],[0.1,0.0,0.9],[0.0,0.9,0.1]])
        # Ranks for D2T: D0->T0 (1st), D1->T2 (1st), D2->T1 (1st). MRR = 1.0
        # Ranks for T2D: T0->D0 (1st), T1->D2 (1st), T2->D1 (1st). MRR = 1.0
        # This example was tricky. Let's use a clearer one.
        # D0 correctly matches T0 (rank 1)
        # D1 correctly matches T1 (rank 2, T2 is 0.8, T1 is 0.7)
        # D2 correctly matches T2 (rank 3, T0 is 0.9, T1 is 0.8, T2 is 0.1)
        sim_matrix_varied = np.array([
            [1.0, 0.2, 0.1],  # D0 -> T0 (rank 1)
            [0.2, 0.7, 0.8],  # D1 -> T1 (rank 2)
            [0.9, 0.8, 0.1]   # D2 -> T2 (rank 3)
        ])
        mrr_d2t_v, _ = calculate_mrr(sim_matrix_varied)
        expected_mrr_d2t_v = (1.0/1 + 1.0/2 + 1.0/3) / 3.0
        self.assertAlmostEqual(mrr_d2t_v, expected_mrr_d2t_v)


    def test_expand_fasta_curly_braces(self):
        expanded = expand_fasta_curly_braces(self.dummy_fasta_content_braces)
        self.assertIn("AAAAACCC", expanded)
        self.assertIn("GGGGXT", expanded) # X is preserved by this function

    def test_parse_fasta(self):
        parsed = parse_fasta(self.test_fasta_path)
        self.assertEqual(len(parsed), 2)
        self.assertEqual(parsed[0][0], "s1") # ID
        self.assertEqual(parsed[0][1], "AAAAACCC") # Sequence
        self.assertEqual(parsed[0][2]['species'], "H")
        self.assertEqual(parsed[0][2]['biotype'], "G")
        
        self.assertEqual(parsed[1][0], "s2") # ID
        self.assertEqual(parsed[1][1], "GGGGXT") # Sequence
        self.assertEqual(parsed[1][2].get('description_freeform',"").strip(), "description")
        self.assertEqual(parsed[1][2]['species'], "unknown_species_fasta_header") # Default


    def test_parse_bed(self):
        # Ref: >chrTest1|species=TestSpecies|biotype=RefContig\nATGCGTATGCGTNNNATGCGTATGCGT
        # BED1: chrTest1 0-6 ('ATGCGT'), name 'feat1', biotype 'coding_bed' from attrs
        # BED2: chrTest1 10-15 ('NNNAT'), name 'feat2_coords_are_valid', biotype from ref ('RefContig')
        parsed = parse_bed(self.test_bed_path, self.test_ref_fasta_path)
        self.assertEqual(len(parsed), 2) # feat3_bad_chrom is skipped
        
        # First BED entry
        self.assertEqual(parsed[0][0], "feat1") # Name
        self.assertEqual(parsed[0][1], "ATGCGT") # Sequence
        self.assertEqual(parsed[0][2]['biotype'], "coding_bed") # From BED attributes
        self.assertEqual(parsed[0][2]['species'], "TestSpecies") # From ref FASTA header for chrTest1
        self.assertIn("gene_id=g1", parsed[0][3]) # Annotation text

        # Second BED entry
        self.assertEqual(parsed[1][0], "feat2_coords_are_valid") # Name
        self.assertEqual(parsed[1][1], "NNNAT") # Sequence from chrTest1[10:15]
        self.assertEqual(parsed[1][2]['biotype'], "RefContig") # Default from ref FASTA header for chrTest1
        self.assertEqual(parsed[1][2]['species'], "TestSpecies")
        self.assertIn("feat2_coords_are_valid from TestSpecies chromosome chrTest1:10-15", parsed[1][3])

    def test_parse_gff3(self):
        # Ref: >chrTest1|species=TestSpecies|biotype=RefContig\nATGCGTATGCGTNNNATGCGTATGCGT
        # GFF1: gene, chrTest1 1-6 ('ATGCGT'), Name TestGene1, biotype protein_coding_gff
        # GFF2: mRNA, chrTest1 10-15 ('NNNAT'), strand -, seq should be revcomp of NNNAT -> ATNNN
        target_types = ['gene', 'mRNA']
        parsed = parse_gff3(self.test_gff_path, self.test_ref_fasta_path, target_types)
        self.assertEqual(len(parsed), 2) # gene2 on BadChrom is skipped

        # First GFF entry (gene1)
        self.assertEqual(parsed[0][0], "gene1") # ID
        self.assertEqual(parsed[0][1], "ATGCGT") # Sequence
        self.assertEqual(parsed[0][2]['biotype'], "protein_coding_gff")
        self.assertEqual(parsed[0][2]['species'], "TestSpecies") # From ref FASTA for chrTest1
        self.assertEqual(parsed[0][2]['gff_feature_type'], "gene")
        self.assertIn("protein_coding_gff. TestGene1. from TestSpecies chromosome chrTest1:1-6(+).", parsed[0][3])

        # Second GFF entry (rna1)
        self.assertEqual(parsed[1][0], "rna1") # ID
        self.assertEqual(parsed[1][1], "ATNNN") # Reverse complement of NNNAT
        self.assertEqual(parsed[1][2]['biotype'], "mRNA") # Defaulted to feature type as no biotype attr
        self.assertEqual(parsed[1][2]['species'], "TestSpecies")
        self.assertEqual(parsed[1][2]['gff_feature_type'], "mRNA")
        self.assertIn("mRNA. rna1. from TestSpecies chromosome chrTest1:10-15(-). Description: Test RNA on minus strand.", parsed[1][3])


    @unittest.skipIf(onnx is None, "ONNX library not installed, skipping ONNX export test.")
    def test_onnx_export_conceptual(self):
        model = DNAClipModel(20, 20, 16, 24, 1).to(self.device); model.eval()
        dummy_tokens = torch.randint(0, 20, (1, 10), dtype=torch.long).to(self.device)
        dummy_lengths = torch.tensor([5], dtype=torch.long).cpu() # Lengths on CPU
        
        dna_encoder_path = os.path.join(self.temp_dir, "test_dna_encoder.onnx")
        try:
            torch.onnx.export(model.dna_encoder, 
                              (dummy_tokens, dummy_lengths), 
                              dna_encoder_path, 
                              input_names=['tokens', 'lengths'], output_names=['embedding'], 
                              dynamic_axes={'tokens': {0:'batch',1:'seq_len'}, 
                                            'lengths':{0:'batch'}, 
                                            'embedding':{0:'batch'}}, 
                              opset_version=ONNX_OPSET_VERSION)
            self.assertTrue(os.path.exists(dna_encoder_path))
            onnx.checker.check_model(dna_encoder_path) # Check validity
        except Exception as e: self.fail(f"ONNX export for DNA encoder failed: {e}")

    @unittest.skipIf(any(lib is None for lib in [tensorflow, onnx, onnx_tf_prepare, tensorflowjs]), 
                     "TensorFlow/ONNX/TFJS libraries not installed, skipping TF.js export test.")
    def test_tfjs_export_conceptual(self):
        # Create a minimal valid ONNX model for testing the export_onnx_to_tfjs pipeline
        mock_onnx_path = os.path.join(self.temp_dir, "mock_encoder.onnx")
        if onnx: # Guard for type hinting if onnx is None
            from onnx import helper, TensorProto
            # Create a simple graph (e.g., Identity operator)
            X = helper.make_tensor_value_info('X', TensorProto.FLOAT, [None, 3]) # Dynamic batch, 3 features
            Y = helper.make_tensor_value_info('Y', TensorProto.FLOAT, [None, 3])
            node_def = helper.make_node('Identity', ['X'], ['Y'])
            graph_def = helper.make_graph([node_def], 'dummy-model', [X], [Y])
            model_def = helper.make_model(graph_def, producer_name='unittest-tfjs')
            onnx.save(model_def, mock_onnx_path)
        
        tfjs_output_path = os.path.join(self.temp_dir, "tfjs_model_output")
        
        try:
            # Mock subprocess.run to avoid actual conversion, just check call
            with unittest.mock.patch('subprocess.run') as mock_subproc_run:
                # Simulate successful conversion
                mock_subproc_run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=0, stdout="mocked success", stderr=""
                )
                
                export_onnx_to_tfjs(mock_onnx_path, tfjs_output_path)
                
                mock_subproc_run.assert_called() # Check that the converter was called
                # Check if tfjs_output_dir was at least attempted to be created (mocked, so won't exist)
                # Instead, we can check the arguments to the call
                called_args = mock_subproc_run.call_args[0][0] # First arg of first call
                self.assertIn(tfjs_output_path, called_args) # Check if output path was in command

        except Exception as e: 
            self.fail(f"TF.js export function call failed: {e}")


if __name__ == "__main__":
    # Global storage for GFF specific stats if GFF parser is used
    dataset_summary_temp_storage = {} 
    main()
