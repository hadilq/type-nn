{
  description = "type-nn: product-of-affine (AND-OR) nets from Type Mechanics";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.11";
  };

  outputs =
    {
      self,
      nixpkgs,
      ...
    }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs { inherit system; };

      py = pkgs.python3.withPackages (
        ps: with ps; [
          torch
        ]
      );

      # Pinned public datasets. Hashes are SRI of the exact bytes fetched
      # from these URLs (see README). Re-run `nix flake lock` is not needed
      # when only these hashes are already fixed.
      iris = pkgs.fetchurl {
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data";
        hash = "sha256-b2CLcacxchYxm00ntNm8hOar1zTtp4crcaRYVp4mVsA=";
      };
      wine = pkgs.fetchurl {
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data";
        hash = "sha256-a+axID89Ud8LVTpw5XuKcjzUBWg5WCBPltI9fNauplk=";
      };
      wdbc = pkgs.fetchurl {
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data";
        hash = "sha256-1gavQR8+W+ijF6WotlK0Jarw/zjKaD1TJ///lMNpX0o=";
      };
      diabetes = pkgs.fetchurl {
        url = "https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt";
        hash = "sha256-RzP+vuaXhiwiE5zayHR4owDODRAVk96wftbA8zKKmc0=";
      };
      ionosphere = pkgs.fetchurl {
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data";
        hash = "sha256-RtUhhrhOIL5SkYrbk+j7mSazR5X/dQTCQ1CuBhagS70=";
      };

      datasets = pkgs.runCommand "type-nn-datasets" { } ''
        mkdir -p $out/share/type-nn
        cp ${iris} $out/share/type-nn/iris.data
        cp ${wine} $out/share/type-nn/wine.data
        cp ${wdbc} $out/share/type-nn/wdbc.data
        cp ${diabetes} $out/share/type-nn/diabetes.tab.txt
        cp ${ionosphere} $out/share/type-nn/ionosphere.data
      '';

      devShell = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          gdb
          clang
          pkg-config
          curl
          py
        ];
        TYPE_NN_DATA = "${datasets}/share/type-nn";
        shellHook = ''
          echo "TYPE_NN_DATA=$TYPE_NN_DATA"
          echo "datasets: iris wine wdbc diabetes ionosphere"
          cp $TYPE_NN_DATA/*.data data/
          cp $TYPE_NN_DATA/diabetes.tab.txt data/
        '';
      };

      package = pkgs.stdenv.mkDerivation {
        name = "type-nn";
        src = ./.;
        nativeBuildInputs = [
          pkgs.gcc
          pkgs.gnumake
        ];
        buildInputs = [ datasets ];
        buildPhase = "make type-nn test_type_nn bench_type_nn";
        installPhase = ''
          mkdir -p $out/bin $out/share/type-nn
          cp type-nn test_type_nn bench_type_nn $out/bin/
          cp -R ${datasets}/share/type-nn/. $out/share/type-nn/
        '';
      };
    in
    {
      packages.${system}.default = package;
      devShells.${system}.default = devShell;
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
