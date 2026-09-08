{
  description = "Vortex dynamic AND-OR network";

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

      devShell = pkgs.mkShell {
        packages = with pkgs; [
          gcc
          gnumake
          gdb
          clang
          pkg-config
        ];
      };

      package = pkgs.stdenv.mkDerivation {
        name = "vortex";
        src = ./.;
        nativeBuildInputs = [
          pkgs.gcc
          pkgs.gnumake
        ];
        buildPhase = "make vortex test_vortex";
        installPhase = ''
          mkdir -p $out/bin
          cp vortex test_vortex $out/bin/
        '';
      };
    in
    {
      packages.${system}.default = package;
      devShells.${system}.default = devShell;
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
