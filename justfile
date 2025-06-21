set positional-arguments
set shell := ["bash", "-cue"]

mod nix './tools/just/nix.just'

[private]
default:
  just --unsorted --list

# Enter a development shell.
develop:
  just nix::develop default

# Format python code
format *args:
  ruff format {{args}} src/

alias dev := develop
alias fmt := format 
