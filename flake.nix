{
  description = "type-nn: a trainable stack of partition functions (Or / And / depth scaling)";

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

      # Rocq 9 ships its standard library as a separate package. Without it
      # `From Stdlib Require ...` (and the deprecated `From Coq`) fails with
      # "Cannot find a physical path bound to logical path List with prefix
      # Stdlib". Taking both from one coqPackages set keeps versions matched.
      rocq = pkgs.coqPackages.coq;
      rocqStdlib = pkgs.coqPackages.stdlib;
      rocqPath = "${rocqStdlib}/lib/coq/${rocq.coq-version}/user-contrib";

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
          python3
          rocq
        ];
        # the setup hook finds user-contrib of buildInputs; the explicit
        # paths are a fallback for shells that do not run it
        buildInputs = [ rocqStdlib ];
        COQPATH = rocqPath;
        ROCQPATH = rocqPath;
        TYPE_NN_DATA = "${datasets}/share/type-nn";
        shellHook = ''
          echo "TYPE_NN_DATA=$TYPE_NN_DATA"
          echo "datasets: iris wine wdbc diabetes ionosphere"
          mkdir -p data
          cp -f --no-preserve=mode $TYPE_NN_DATA/*.data $TYPE_NN_DATA/diabetes.tab.txt data/
          echo "make test    -> unit tests, gradient checks"
          echo "./bench.sh   -> type-nn, type-nn-overfit, c-mlp board (BOARD.txt)"
          echo "make coq     -> check the proofs in ./coqLang"
        '';
      };

      # `nix build .#proofs` type-checks the Coq development. Hermetic.
      proofs = pkgs.stdenv.mkDerivation {
        name = "type-nn-proofs";
        src = ./coqLang;
        nativeBuildInputs = [ rocq pkgs.gnumake ];
        buildInputs = [ rocqStdlib ];
        COQPATH = rocqPath;
        ROCQPATH = rocqPath;
        buildPhase = "make";
        installPhase = ''
          mkdir -p $out
          cp -R TypeNN $out/
          echo "TypeNN proofs check out" > $out/RESULT
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
        buildPhase = "make test_type_nn test_type_nn_overfit test_cmlp bench && ./test_type_nn && ./test_type_nn_overfit && ./test_cmlp";
        installPhase = ''
          mkdir -p $out/bin $out/share/type-nn
          cp test_type_nn test_type_nn_overfit test_cmlp bench bench.sh $out/bin/
          cp -R ${datasets}/share/type-nn/. $out/share/type-nn/
        '';
      };
    in
    {
      packages.${system} = {
        default = package;
        inherit proofs;
      };
      devShells.${system}.default = devShell;
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
