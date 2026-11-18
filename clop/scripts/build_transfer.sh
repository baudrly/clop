#!/usr/bin/env bash
# Out-of-atlas transfer set: three genomes absent from training, windowed exactly like the atlas (500 bp).
#   kiwifruit (Actinidia chinensis) chr1 GFF3 - an unseen plant;  Bacteroides fragilis GenBank - an unseen bacterium;
#   the Arabidopsis thaliana chloroplast - an organelle genome of a training species, of cyanobacterial origin.
# Inputs come from public test-data repositories (nf-core/test-datasets, biopython). Usage: build_transfer.sh SRC OUT
set -euo pipefail
src=${1:?source directory}; out=${2:?output directory}
mkdir -p "$out"
kiwi="$src/actinidia_chinensis"; bfrag="$src/bacteroides_fragilis"
clop prepare -i "$kiwi/genome.gff3.gz" --reference "$kiwi/genome.fasta.gz" --species "Actinidia chinensis" \
  --feature-types CDS,intron,mRNA --window-types CDS,intron --promoters mRNA --promoter-offset 100 --intergenic 600 \
  --split none --cap 600 -o "$out/kiwifruit.fa"
clop prepare -i "$bfrag/genome.gbff.gz" --species "Bacteroides fragilis" --feature-types CDS,tRNA,rRNA \
  --promoters CDS --promoter-offset 50 --intergenic 600 --intergenic-margin 50 --split none --cap 600 -o "$out/bacteroides.fa"
clop prepare -i "$src/NC_000932.gb" --species "Arabidopsis thaliana chloroplast" --feature-types CDS,tRNA,rRNA \
  --promoters CDS --promoter-offset 50 --intergenic 200 --intergenic-margin 50 --split none --cap 600 -o "$out/chloroplast.fa"
