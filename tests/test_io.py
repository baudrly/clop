import gzip

import pytest

from clop.io import (
    Reference,
    centred_window,
    deduplicate,
    intergenic_windows,
    load_records,
    parse_attributes,
    parse_bed,
    parse_fasta,
    parse_genbank,
    parse_gff3,
    parse_header,
    promoter_windows,
    read_fasta,
    reverse_complement,
    write_fasta,
)

REFERENCE = ">chrTest1|species=TestSpecies|biotype=RefContig\nATGCGTATGCGTNNNATGCGTATGCGT\n>chrTest2\nCCCCCCCCCCCCCCCCCCCCCCCCC\n"


@pytest.fixture
def reference(tmp_path):
    path = tmp_path / "ref.fa"
    path.write_text(REFERENCE)
    return path


def test_reverse_complement_iupac_and_case():
    assert reverse_complement("ACGTRYN") == "NRYACGT"
    assert reverse_complement("acgt") == "acgt"


@pytest.mark.parametrize(
    "header,expected",
    [
        ("s1|species=H|biotype=G", {"id": "s1", "species": "H", "biotype": "G"}),
        (">s1|species=Homo sapiens|biotype=lncRNA|description=a | b", {"species": "Homo sapiens", "biotype": "lncRNA"}),
        ("Homo sapiens|snRNA|U1 RNA", {"species": "Homo sapiens", "biotype": "snRNA", "description": "U1 RNA"}),
        ("s2 free text here", {"id": "s2", "description": "free text here"}),
        ("species=Mus musculus|biotype=CDS", {"id": "Mus musculus", "species": "Mus musculus"}),
    ],
)
def test_parse_header(header, expected):
    meta = parse_header(header)
    for key, value in expected.items():
        assert meta[key] == value


def test_parse_fasta_keyed_and_plain(tmp_path):
    path = tmp_path / "x.fa.gz"
    with gzip.open(path, "wt") as handle:
        handle.write(
            ">s1|species=H|biotype=G|chrom=chr1|split=test\nAAAAA\nccc\n>s2 description\nGGGGXT\n>s3|biotype=unknown\nA\n"
        )
    records = parse_fasta(path)
    assert [r.id for r in records] == ["s1", "s2", "s3"]
    assert records[0].sequence == "AAAAACCC" and records[0].species == "H" and records[0].biotype == "G"
    assert records[0].chrom == "chr1" and records[0].split == "test"
    assert records[1].description == "description" and records[1].species == "unknown"
    assert len(parse_fasta(path, keep_unknown=False)) == 1


def test_write_fasta_roundtrip(tmp_path):
    records = parse_fasta(_fasta(tmp_path, ">a|species=Homo sapiens|biotype=CDS|chrom=chr2|start=10|end=20|strand=-\nACGT\n"))
    out = tmp_path / "rt.fa"
    write_fasta(records, out)
    again = parse_fasta(out)
    assert again[0] == records[0]


def _fasta(tmp_path, text):
    path = tmp_path / "in.fa"
    path.write_text(text)
    return path


def test_parse_bed(tmp_path, reference):
    bed = tmp_path / "x.bed"
    bed.write_text(
        "chrTest1\t0\t6\tfeat1\t100\t+\t.\tgene_id=g1;biotype=coding_bed\n"
        "chrTest1\t10\t15\tfeat2\t0\t-\nchrTestNonExistent\t0\t5\tfeat3\n"
    )
    records = parse_bed(bed, reference)
    assert len(records) == 2
    assert records[0].sequence == "ATGCGT" and records[0].biotype == "coding_bed" and records[0].species == "TestSpecies"
    assert records[1].sequence == reverse_complement("GTNNN") and records[1].biotype == "RefContig"


def test_parse_gff3_and_gtf(tmp_path, reference):
    gff = tmp_path / "x.gff3"
    gff.write_text(
        "##gff-version 3\nchrTest1\tT\tgene\t1\t6\t.\t+\t.\tID=gene1;Name=G1;biotype=protein_coding\n"
        "chrTest1\tT\tmRNA\t10\t15\t.\t-\t.\tID=rna1;Parent=gene1;description=minus%20strand\n"
        "chrMissing\tT\tgene\t1\t5\t.\t+\t.\tID=gene2\n"
    )
    records = parse_gff3(gff, reference)
    assert [r.id for r in records] == ["gene1", "rna1"]
    assert records[0].biotype == "protein_coding" and records[0].sequence == "ATGCGT"
    assert (
        records[1].biotype == "mRNA"
        and records[1].sequence == reverse_complement("CGTNNN")
        and records[1].description == "minus strand"
    )
    assert len(parse_gff3(gff, reference, ["gene"])) == 1
    gtf = tmp_path / "x.gtf"
    gtf.write_text('chrTest1\tT\texon\t1\t6\t.\t+\t.\tgene_id "g1"; gene_biotype "snRNA";\n')
    assert parse_gff3(gtf, reference)[0].biotype == "snRNA"


def test_parse_attributes_urldecodes():
    assert parse_attributes("Note=a%3Bb;ID=x") == {"Note": "a;b", "ID": "x"}


GENBANK = """LOCUS       TEST1                     60 bp    DNA     linear   BCT 01-JAN-2000
DEFINITION  Test record.
ACCESSION   TEST1
  ORGANISM  Testus bacterium
FEATURES             Location/Qualifiers
     source          1..60
     CDS             1..12
                     /locus_tag="T_001"
                     /product="hypothetical
                     protein"
     tRNA            complement(join(20..25,30..>35))
                     /locus_tag="T_t1"
ORIGIN
        1 atgaaacccg ggtttaaacc cgggtttaaa cccgggttta aacccgggtt taaacccggg
//
"""


def test_parse_genbank(tmp_path):
    path = tmp_path / "x.gb"
    path.write_text(GENBANK)
    records, reference = parse_genbank(path, None)
    kinds = {r.biotype: r for r in records}
    assert set(kinds) == {"source", "CDS", "tRNA"}
    assert kinds["CDS"].sequence == "ATGAAACCCGGG" and kinds["CDS"].description == "hypothetical protein"
    assert kinds["tRNA"].strand == "-" and (kinds["tRNA"].start, kinds["tRNA"].end) == (19, 35)
    assert kinds["CDS"].species == "Testus bacterium" and reference.length("TEST1") == 60
    assert len(load_records(path, feature_types=["CDS"])) == 1


def test_windows(tmp_path, reference):
    ref = Reference.from_fasta(reference)
    gff = tmp_path / "x.gff3"
    gff.write_text("chrTest1\tT\tgene\t5\t8\t.\t-\t.\tID=g\n")
    feature = parse_gff3(gff, ref)[0]
    window = centred_window(feature, ref, 10)
    assert len(window.sequence) == 10 and window.strand == "-"
    promoter = promoter_windows([feature], ref, 6, upstream=2)[0]
    assert promoter.biotype == "promoter" and len(promoter.sequence) == 6
    assert ref.fetch("chrTest1", -3, 2) == "NNNAT"


def test_intergenic_and_dedup(tmp_path):
    ref = Reference({"c": "ACGT" * 500})
    from clop.io import Record

    features = [Record("f", "", chrom="c", start=900, end=1100)]
    windows = intergenic_windows(features, ref, 100, 20, margin=200, species="S")
    assert windows and all(w.end <= 700 or w.start >= 1300 for w in windows)
    dup = [Record("a", "ACGTTT"), Record("b", "AAACGT"), Record("c", "ACGTTT")]
    assert [r.id for r in deduplicate(dup)] == ["a"]


def test_read_fasta_streams_multiline(tmp_path):
    path = _fasta(tmp_path, ">x\nAC\nGT\n\n>y\nNN\n")
    assert list(read_fasta(path)) == [("x", "ACGT"), ("y", "NN")]
