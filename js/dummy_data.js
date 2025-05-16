// js/dummy_data.js

const dummyFastaExamples = [
    {
        name: "Short Sequences (Mixed)",
        data: `>seq1 Human gene promoter region|species=Homo sapiens|biotype=promoter
{A*10}{C*5}GTAG{T*8}NNN{C*5}
>seq2 Mouse enhancer element|species=Mus musculus|biotype=enhancer
{G*15}ATAT{C*10}GTAG{A*7}
>seq3 Yeast coding sequence|species=Saccharomyces cerevisiae|biotype=protein_coding
ATG{C*3}G{A*3}T{G*3}C{T*3}A{G*3}TAA
>seq4 Fly non-coding RNA|species=Drosophila melanogaster|biotype=lncRNA
{T*5}G{C*8}A{T*5}G{A*8}C{N*5}`
    },
    {
        name: "Repetitive Elements",
        data: `>repeat_A_human|species=Homo sapiens|biotype=repeat
{A*50}
>repeat_GC_mouse|species=Mus musculus|biotype=repeat
{GC*30}`
    },
    {
        name: "Minimal Example",
        data: `>min1|species=Unknown|biotype=Unknown
ACGT
>min2|species=Unknown|biotype=Unknown
TGCA`
    }
];

function getDummyFastaExamples() {
    return dummyFastaExamples;
}

const dummyEmbeddingsPath = 'examples/dummy_embeddings.json';
const dummyDnaVocabPath = 'examples/dummy_dna_vocab.json';
const dummyTextVocabPath = 'examples/dummy_text_vocab.json';

// Example JSON contents are now separate files in `examples/` folder.