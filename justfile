set positional-arguments
set shell := ["bash", "-cue"]

mod nix './tools/just/nix.just'

[private]
default:
  just --unsorted --list

# Enter a development shell.
develop:
  just nix::develop default

# Format and lint python code
format *args:
  ruff format {{args}} src/ tests/ scripts/
  ruff check --fix src/ tests/ scripts/

# Run the test suite
test *args:
  pytest {{args}}

# Train the synthetic demo and build its report
demo:
  clop demo -o clop_demo

# Build the web reader from the shipped model
web:
  clop export -m models/clop-atlas6.pt -o _site --web --atlas models/atlas-sample.fa.gz && echo "open _site/index.html (or share _site/clop-reader.html)"

alias dev := develop
alias fmt := format
