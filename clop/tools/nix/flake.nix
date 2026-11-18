{
  description = "CUDA-enabled python environment using uv.";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    nixpkgsStable.url = "github:nixos/nixpkgs/nixos-25.05";
    flake-utils.url = "github:numtide/flake-utils";
    devenv.url = "github:cachix/devenv";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    devenv,
    ...
  } @ inputs:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
        };

        # python environment defined separately
        pythonModules = import ./devenv.nix {
          inherit pkgs;
          inherit (pkgs) lib;
        };
      in {
        devShells = {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = pythonModules;
          };
        };
      }
    );
}
