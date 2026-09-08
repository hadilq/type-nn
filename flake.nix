{
  description = "Vortex configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
    nixpkgs-unstable.url = "github:NixOS/nixpkgs/190a798e08f0214c6573599972fb1b593b832496";
  };

  outputs =
    {
      self,
      nixpkgs,
      nixpkgs-unstable,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
      };
      pkgs-unstable = import nixpkgs-unstable {
        inherit system;
      };

      devShell = pkgs.mkShell {
        packages = (with pkgs; [
          pkg-config
          pcre2
          gcc
          gnumake 
          gdb 
          clang
          llvmPackages_20.openmp
          (pkgs.python3.withPackages (python-pkgs: with python-pkgs; [
            pip
            ninja
            setuptools
            torch
          ]))
        ]);
      };

      package = pkgs.stdenv.mkDerivation {
        name = "vortex";
        src = ./.;
        nativeBuildInputs = [ pkgs.gcc pkgs.gnumake ];
        buildPhase = "make";
        installPhase = "mkdir -p $out/bin && cp vortex $out/bin/";
      };

    in
    {
      packages.${system}.default = package;

      # Development shell
      devShells.${system}.default = devShell;

      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
