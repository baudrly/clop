import json
import shutil
import subprocess

import numpy as np
import pytest

from clop.demo import synthetic_records
from clop.models import ModelConfig
from clop.pipeline import Pipeline
from clop.train import TrainConfig
from clop.workflow import prepare_records, train_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    out = tmp_path_factory.mktemp("run")
    parts = prepare_records(synthetic_records(20, 150), "chromosome")
    train_model(parts, ModelConfig(width=16, depth=2, embed_dim=16), TrainConfig(epochs=1, batch_size=32), out)
    return out


def test_onnx_matches_pytorch(trained, tmp_path):
    pytest.importorskip("onnxruntime")
    from clop.export import export_onnx

    paths = export_onnx(Pipeline.load(trained / "model.pt"), tmp_path)  # raises on mismatch
    assert all(p.exists() for p in paths) and (tmp_path / "onnx_meta.json").exists()


def test_report(trained):
    from clop.report import write_report

    page = write_report(trained).read_text()
    assert "Atlas3D" in page and "<canvas" in page and (trained / "figures" / "confusion_cells.png").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_web_reader_matches_pytorch(trained, tmp_path):
    from clop.export import export_web

    pipe = Pipeline.load(trained / "model.pt")
    records = synthetic_records(5, 150)
    export_web(pipe, tmp_path, records, 50)
    long = "".join(r.sequence for r in records[:9])  # longer than the window: tiled in both implementations
    assert len(long) > pipe.model.cfg.window
    sequences = [records[0].sequence, records[7].sequence[:97] + "NNNN" + records[7].sequence[97:], "acgt" * 33, long]
    texts = ["synthetica alba trna", "words never seen"]
    expected = {
        "seqs": sequences,
        "texts": texts,
        "dna": pipe.embed(sequences).tolist(),
        "text": pipe.embed_texts(texts).tolist(),
    }
    (tmp_path / "expected.json").write_text(json.dumps(expected))
    script = """
const fs = require("fs"); global.atob = s => Buffer.from(s, "base64").toString("binary");
const {CLOPModel} = require(process.argv[2]);
require(process.argv[3]);  // sets globalThis.CLOP_MODEL, as a <script> tag does
const m = new CLOPModel(CLOP_MODEL), e = JSON.parse(fs.readFileSync(process.argv[4]));
let worst = 0;
e.seqs.forEach((s, i) => m.embedDNA(s).forEach((v, j) => (worst = Math.max(worst, Math.abs(v - e.dna[i][j])))));
e.texts.forEach((t, i) => m.embedText(t).forEach((v, j) => (worst = Math.max(worst, Math.abs(v - e.text[i][j])))));
console.log(worst);
"""
    (tmp_path / "check.js").write_text(script)
    result = subprocess.run(
        [
            "node",
            str(tmp_path / "check.js"),
            str(tmp_path / "clop.js"),
            str(tmp_path / "clop-model.js"),
            str(tmp_path / "expected.json"),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert float(result.stdout) < 1e-4
    text = (tmp_path / "clop-atlas.js").read_text()
    atlas = json.loads(text[text.index("=") + 1 : text.rindex(";")].replace("<\\/", "</"))
    assert len(atlas["positions"]) == 3 * len(atlas["species"]) == 3 * 50
    assert np.isfinite(atlas["positions"]).all() and all(1 <= len(v) <= 3 for v in atlas["examples"].values())
    page = (tmp_path / "clop-reader.html").read_text()  # single file: nothing left to load
    assert "<script src=" not in page and 'href="style.css"' not in page and "globalThis.CLOP_ATLAS=" in page
    assert page.count("</script>") == 5  # one per inlined file: no inlined text closes its script early


def test_legacy_flags():
    import warnings

    from clop.cli import translate_legacy

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        argv = translate_legacy(
            ["--input_file", "x.fa", "--output_dir", "out", "--batch_size", "16", "--export_onnx", "--dna_tokenizer_type", "kmer"]
        )
    assert argv[:7] == ["train", "--input", "x.fa", "--output", "out", "--batch-size", "16"]
    assert "--dna-encoder" in argv and "kmer" not in argv
    assert translate_legacy(["--run_dummy_example"])[0] == "demo"
