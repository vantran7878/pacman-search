{
  description = "Example Python flake";

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
  in {
    devShell.${system} = pkgs.mkShell {
      buildInputs = [
        (pkgs.python312.withPackages (python-pkgs: [
          python-pkgs.mypy
          python-pkgs.pyglet
          python-pkgs.flake8
          python-pkgs.pylint
        ]))

        pkgs.pyright
      ];
    };
  };
}
