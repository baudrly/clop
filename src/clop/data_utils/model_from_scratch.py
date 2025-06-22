import torch
import torch.nn as nn
import torch.optim as optim
from Bio import SeqIO
import string
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from transformers import BertModel, BertTokenizer
from torch.utils.data import Dataset, DataLoader
import zipfile
import gzip
import bz2


# Preprocessing utilities
def sequence_to_tensor(sequence, nucleotide_map):
    return torch.tensor(
        [nucleotide_map[nuc] for nuc in sequence], dtype=torch.long
    )


def annotation_to_tensor(annotation, tokenizer):
    return tokenizer.encode(
        annotation, return_tensors="pt", add_special_tokens=True
    ).squeeze(0)


# Enhanced Sequence Encoder
class DNASequenceEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(DNASequenceEncoder, self).__init__()
        self.embedding = nn.Embedding(
            5, embedding_dim
        )  # 4 nucleotides + 1 for padding
        self.conv_layers = nn.Sequential(
            nn.Conv1d(embedding_dim, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.bi_lstm = nn.LSTM(
            128, 128, num_layers=2, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(
            256, embedding_dim
        )  # 256 = 128 * 2 for bidirectional

    def forward(self, x, lengths):
        x = self.embedding(x).permute(
            0, 2, 1
        )  # [batch, embedding_dim, sequence_length]
        x = self.conv_layers(x).permute(
            0, 2, 1
        )  # [batch, sequence_length, channels]
        packed_x = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        packed_out, _ = self.bi_lstm(packed_x)
        out, _ = pad_packed_sequence(packed_out, batch_first=True)
        out = out[torch.arange(out.size(0)), lengths - 1]  # get last timestep
        out = self.fc(out)
        return out


# Enhanced Annotation Encoder with Transformer
class AnnotationEncoder(nn.Module):
    def __init__(self, embedding_dim):
        super(AnnotationEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc = nn.Linear(self.bert.config.hidden_size, embedding_dim)

    def forward(self, x, attention_mask):
        outputs = self.bert(x, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)


# DNA-Clip Model
class DNA_CLIP(nn.Module):
    def __init__(self, embedding_dim=128):
        super(DNA_CLIP, self).__init__()
        self.sequence_encoder = DNASequenceEncoder(embedding_dim)
        self.annotation_encoder = AnnotationEncoder(embedding_dim)

    def forward(self, sequence, annotation, seq_lengths, attention_mask):
        sequence_emb = self.sequence_encoder(sequence, seq_lengths)
        annotation_emb = self.annotation_encoder(annotation, attention_mask)
        return sequence_emb, annotation_emb


# Custom Dataset Class
class FastaDataset(Dataset):
    def __init__(self, file_path, nucleotide_map, tokenizer):
        (
            self.sequences,
            self.annotations,
            self.seq_lengths,
            self.attention_masks,
        ) = self.process_fasta(file_path, nucleotide_map, tokenizer)

    def process_fasta(self, file_path, nucleotide_map, tokenizer):
        sequences, annotations, seq_lengths, attention_masks = [], [], [], []
        open_func = open  # Default for uncompressed files

        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as z:
                fasta_path = z.extract(z.namelist()[0])
                open_func = open
                file_path = fasta_path
        elif file_path.endswith(".gz"):
            open_func = gzip.open
        elif file_path.endswith(".bz2"):
            open_func = bz2.open

        with open_func(file_path, "rt") as f:
            for record in SeqIO.parse(f, "fasta"):
                sequence = record.seq.upper()
                annotation = record.description
                sequences.append(sequence_to_tensor(sequence, nucleotide_map))
                encoded_annotation = annotation_to_tensor(
                    annotation, tokenizer
                )
                annotations.append(encoded_annotation)
                seq_lengths.append(len(sequence))
                attention_masks.append(
                    torch.tensor(
                        [1] * encoded_annotation.size(0), dtype=torch.long
                    )
                )

        return (
            pad_sequence(sequences, batch_first=True),
            pad_sequence(annotations, batch_first=True),
            torch.tensor(seq_lengths),
            pad_sequence(attention_masks, batch_first=True),
        )

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            self.sequences[idx],
            self.annotations[idx],
            self.seq_lengths[idx],
            self.attention_masks[idx],
        )


# Nucleotide and character mappings
nucleotide_map = {"A": 1, "C": 2, "G": 3, "T": 4, "N": 0}
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Example Usage
model = DNA_CLIP()
fasta_file_path = (
    "path_to_fasta_file.fasta.gz"  # Supports .zip, .gz, .bz2, or uncompressed
)
dataset = FastaDataset(fasta_file_path, nucleotide_map, tokenizer)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_function = nn.MSELoss()  # Example loss function, modify as needed

for sequences, annotations, seq_lengths, attention_masks in data_loader:
    optimizer.zero_grad()
    sequence_emb, annotation_emb = model(
        sequences, annotations, seq_lengths, attention_masks
    )

    # Compute loss, for example, using Mean Squared Error (modify according to your needs)
    loss = loss_function(
        sequence_emb, annotation_emb
    )  # Assuming a simplistic approach for demonstration
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model
