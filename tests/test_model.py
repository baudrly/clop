import json

import numpy as np
import pytest
import torch

from clop.baselines import canonical_index, kmer_profiles, knn_predict
from clop.data import CaptionTable, SequenceDataset, assign_splits, batch_captions, cap_per_cell, collate
from clop.demo import synthetic_records
from clop.evaluate import bootstrap_f1, hierarchical_metrics, label_bias, retrieval_metrics
from clop.io import Record, reverse_complement
from clop.losses import clip_loss, multi_positive_loss, sigmoid_loss
from clop.models import CLOP, ModelConfig, fcgr
from clop.pipeline import Pipeline, tile_starts
from clop.text import TextTokenizer, caption
from clop.tokenizers import encode, kmer_ids, pad_codes, reverse_complement_codes

ENCODERS = [
    dict(dna_encoder="cnn", width=16, depth=2),
    dict(dna_encoder="cnn", width=16, depth=2, pooling="meanmax"),
    dict(dna_encoder="transformer", width=16, k=3, layers=1, heads=2),
    dict(dna_encoder="lstm", width=16, k=3, layers=1),
    dict(dna_encoder="fcgr", width=16, k=4),
]


def _batch(seqs):
    codes, lengths = pad_codes([encode(s) for s in seqs])
    return torch.from_numpy(codes), torch.from_numpy(lengths)


def test_reverse_complement_codes_matches_strings():
    seqs = ["ACGTTGCAN", "GGGA", "T"]
    codes, lengths = _batch(seqs)
    rc = reverse_complement_codes(codes, lengths)
    for i, s in enumerate(seqs):
        assert rc[i, : len(s)].tolist() == encode(reverse_complement(s)).tolist()
        assert (rc[i, len(s) :] == 4).all()


def test_kmer_ids_handle_n_and_padding():
    codes, lengths = _batch(["ACGTA", "ANG"])
    ids, counts = kmer_ids(codes, lengths, 2)
    assert counts.tolist() == [4, 2]
    assert ids[1].tolist()[:2] == [1, 1] and ids[1, 2:].eq(0).all()


@pytest.mark.parametrize("kw", ENCODERS)
def test_encoders_shapes_padding_and_strand(kw):
    torch.manual_seed(0)
    model = CLOP(ModelConfig(text_vocab_size=20, embed_dim=8, **kw)).eval()
    seqs = ["ACGTACGTTTGACCA" * 4, "GGCATTACG" * 3]
    codes, lengths = _batch(seqs)
    emb = model.encode_dna(codes, lengths)
    assert emb.shape == (2, 8) and torch.allclose(emb.norm(dim=1), torch.ones(2), atol=1e-5)
    alone = model.encode_dna(*_batch(seqs[1:]))
    assert torch.allclose(emb[1], alone[0], atol=1e-4), "padding must not change embeddings"
    if model.cfg.rc:
        rc = model.encode_dna(*_batch([reverse_complement(s) for s in seqs]))
        assert torch.allclose(emb, rc, atol=1e-4), "embedding must be strand-invariant"


def test_cnn_moment_pooling_ignores_length():
    """A periodic sequence has the same interior statistics at any length: moment pooling gives identical features,
    while mean+max pooling, whose mean includes the edges, does not."""
    torch.manual_seed(0)
    pooled = {}
    for pooling in ("moments", "meanmax"):
        model = CLOP(ModelConfig(text_vocab_size=4, width=16, depth=2, pooling=pooling)).eval()
        with torch.no_grad():
            pooled[pooling] = [model.dna.features(*_batch(["ACGGT" * n])) for n in (40, 80, 160)]
    assert all(torch.allclose(pooled["moments"][0], p, atol=1e-5) for p in pooled["moments"][1:])
    assert not torch.allclose(pooled["meanmax"][0], pooled["meanmax"][2], atol=1e-3)


def test_tiled_embedding_of_long_inputs():
    assert tile_starts(100, 500).tolist() == [0]
    assert tile_starts(1000, 500).tolist() == [0, 250, 500]
    assert tile_starts(1100, 500).tolist() == [0, 250, 500, 600]
    torch.manual_seed(0)
    model = CLOP(ModelConfig(text_vocab_size=4, width=16, depth=2, embed_dim=8, window=200))
    pipe = Pipeline(model, TextTokenizer.build(["a"]))
    rng = np.random.default_rng(0)
    long, short = "".join(rng.choice(list("ACGT"), 450)), "ACGT" * 30
    got = pipe.embed([short, long, short])
    tiles = pipe.embed([long[s : s + 200] for s in tile_starts(450, 200)])
    mean = tiles.mean(0) / np.linalg.norm(tiles.mean(0))
    assert np.allclose(got[1], mean, atol=1e-5) and np.allclose(got[0], got[2], atol=1e-6)


def test_dataset_crop_range():
    records = synthetic_records(4, 300)
    ds = SequenceDataset(records, crop_lengths=[50, 200])
    sizes = [len(ds[i % len(ds)][0]) for i in range(400)]
    assert 50 <= min(sizes) < 70 and 180 < max(sizes) <= 200


def test_fcgr_counts_and_symmetry():
    codes, lengths = _batch(["AAAA", "TTTT"])
    image = fcgr(codes, lengths, 2, rc=True)
    assert image.shape == (2, 1, 4, 4)
    assert torch.allclose(image[0], image[1])


def test_losses_reward_correct_alignment():
    torch.manual_seed(0)
    text = torch.nn.functional.normalize(torch.randn(3, 8), dim=1)
    owner = torch.tensor([0, 0, 1, 2])
    aligned, scale = text[owner], torch.tensor(10.0)
    shuffled = text[torch.tensor([1, 2, 0, 0])]
    assert multi_positive_loss(aligned, text, owner, scale) < multi_positive_loss(shuffled, text, owner, scale)
    bias = torch.tensor(-1.0)
    assert sigmoid_loss(aligned, text, owner, scale, bias) < sigmoid_loss(shuffled, text, owner, scale, bias)
    assert clip_loss(aligned, aligned, scale) < clip_loss(aligned, shuffled, scale)


def test_text_tokenizer_roundtrip(tmp_path):
    tok = TextTokenizer.build(["homo sapiens promoter", "mus musculus cds"])
    tokens, lengths = tok.batch(["homo sapiens unseenword", "cds"])
    assert tokens.shape == (2, 3) and lengths.tolist() == [3, 1] and tokens[0, 2] == 1
    tok.save(tmp_path / "v.json")
    assert TextTokenizer.load(tmp_path / "v.json").vocab == tok.vocab
    assert "human" in caption("Homo sapiens", "CDS")


def test_splits_keep_chromosomes_and_duplicates_together():
    records = synthetic_records(20, 60)
    assign_splits(records, "chromosome")
    by_unit = {}
    for r in records:
        by_unit.setdefault((r.species, r.chrom), set()).add(r.split)
    assert all(len(s) == 1 for s in by_unit.values())
    assert {r.split for r in records} == {"train", "validation", "test"}
    a, b = Record("a", "ACGTTTGA"), Record("b", reverse_complement("ACGTTTGA"))
    assign_splits([a, b], "random")
    assert a.split == b.split
    assert len(cap_per_cell(records, 5)) <= 5 * len({r.cell for r in records})


def test_dataset_crops_and_batch_captions():
    records = synthetic_records(4, 100)
    tok = TextTokenizer.build([caption(r.species, r.biotype) for r in records])
    table = CaptionTable([caption(r.species, r.biotype) for r in records], tok)
    ds = SequenceDataset(records, table, lambda r: caption(r.species, r.biotype), crop_lengths=[40], random_rc=True)
    batch = collate([ds[i] for i in range(8)])
    assert batch["codes"].shape == (8, 40)
    tokens, _, owner = batch_captions(batch, table)
    assert tokens.shape[0] == len(set(batch["owner"].tolist())) and owner.max() < tokens.shape[0]


def test_metrics():
    names = ["A|x", "A|y", "B|x"]
    truth = ["A|x", "A|y", "B|x", "B|x"]
    scores = np.array([[0.9, 0.1, 0], [0.2, 0.7, 0.1], [0.6, 0, 0.4], [0, 0, 1]])
    m = hierarchical_metrics(truth, scores, names)
    assert m["cell_accuracy"] == 0.75 and m["species_accuracy"] == 0.75
    assert m["biotype_macro_f1_by_species"]["B"] == 1.0  # biotype decided within the true species
    assert label_bias(truth, truth) == 0.0 and label_bias(truth, ["B|x"] * 4) == 0.5
    lo, hi = bootstrap_f1(truth, m["predictions"]["cell"], 50)
    assert 0 <= lo <= hi <= 1
    proto = np.eye(3, 4)
    dna = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0.9, 0.1]])
    r = retrieval_metrics(dna, truth, proto, names)
    assert r["text_to_dna_map"] == pytest.approx(1.0) and r["dna_to_text_top1"] == 1.0


def test_baselines():
    idx = canonical_index(2)
    assert idx.max() + 1 == 10  # 16 dinucleotides -> 10 strand classes
    p = kmer_profiles(["ACGTACGT", "AAAA"], 2)
    assert np.allclose(p.sum(1), 1)
    names, votes = knn_predict(np.eye(3), ["a", "b", "b"], np.array([[1, 0, 0.1]]), k=2)
    assert names[votes.argmax()] == "a"


def test_pipeline_end_to_end(tmp_path):
    from clop.train import TrainConfig
    from clop.workflow import prepare_records, train_model

    records = synthetic_records(30, 120)
    parts = prepare_records(records, "chromosome")
    summary = train_model(parts, ModelConfig(width=16, depth=1, embed_dim=16), TrainConfig(epochs=2, batch_size=32), tmp_path)
    assert {"zero_shot", "knn", "ensemble", "retrieval"} <= set(summary["test"])
    pipe = Pipeline.load(tmp_path / "model.pt")
    assert len(pipe.classify([records[0].sequence], top=2)[0]["top"]) == 2
    calls = pipe.classify([records[0].sequence], top=3)
    assert abs(sum(x["score"] for x in calls[0]["species"]) - 1) < 1e-4  # three species, all listed
    pipe.pool_labels("biotype", ["rRNA"])
    assert set(pipe.label_names) == {"CDS", "promoter", "tRNA", "intergenic", "rRNA"}
    assert np.allclose(np.linalg.norm(pipe.prototypes, axis=1), 1, atol=1e-5)
    hits = pipe.search("synthetica alba trna", pipe.embed([r.sequence for r in records[:10]]), top=3)
    assert len(hits) == 3
    history = json.loads((tmp_path / "history.json").read_text())
    assert len(history) == 2 and "val_cell_f1" in history[0]


def test_resume(tmp_path):
    from clop.data import CaptionTable, SequenceDataset
    from clop.train import TrainConfig, Trainer

    records = synthetic_records(5, 60)
    tok = TextTokenizer.build([caption(r.species, r.biotype) for r in records])
    table = CaptionTable([caption(r.species, r.biotype) for r in records], tok)
    ds = SequenceDataset(records, table, lambda r: caption(r.species, r.biotype))
    cfg = ModelConfig(width=8, depth=1, embed_dim=8, text_vocab_size=len(tok))
    first = Trainer(CLOP(cfg), table, TrainConfig(epochs=1, batch_size=16, patience=10))
    first.fit(ds)
    # simulate an interrupted 3-epoch run: write a state after epoch 1 and resume
    trainer = Trainer(CLOP(cfg), table, TrainConfig(epochs=3, batch_size=16, patience=10))
    trainer.cfg.epochs = 1
    trainer.fit(ds, tmp_path)
    assert (tmp_path / "model.pt").exists()
