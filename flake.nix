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
      spambase = pkgs.fetchurl {
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data";
        hash = "sha256-se+T3nH5dxTT19T1j8n3GNp7vIrIoVDv8neGFqgJexI=";
      };
      mnistImages = pkgs.fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz";
        hash = "sha256-RA/Kv3PMVG+iFHXoHqNwJlYF9WviEKQCTSyo8gNSNgk=";
      };
      mnistLabels = pkgs.fetchurl {
        url = "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz";
        hash = "sha256-NVJTSgpVi77WrtMrMMSVzKI9Vn7FLKyL4aBzDoAQJVw=";
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
        cp ${spambase} $out/share/type-nn/spambase.data
        # MNIST subset: 2000 samples, 14x14 downsampled, 10-class one-hot
        ${pkgs.gzip}/bin/gzip -dc ${mnistImages} > images.idx
        ${pkgs.gzip}/bin/gzip -dc ${mnistLabels} > labels.idx
        ${pkgs.python3}/bin/python3 - "$out" <<'PY'
import struct, sys
out = sys.argv[1]
def read_images(path):
    with open(path,'rb') as f:
        magic, n, r, c = struct.unpack('>IIII', f.read(16))
        return n, r, c, f.read()
def read_labels(path):
    with open(path,'rb') as f:
        magic, n = struct.unpack('>II', f.read(8))
        return n, f.read()
n, r, c, imgs = read_images('images.idx')
nl, labs = read_labels('labels.idx')
out_n, out_r = 2000, 14
step = r // out_r
with open(out + '/share/type-nn/mnist_subset.bin','wb') as f:
    f.write(struct.pack('<III', out_n, out_r*out_r, 10))
    for i in range(out_n):
        base = i * r * c
        for y in range(out_r):
            for x in range(out_r):
                s = 0
                for dy in range(step):
                    for dx in range(step):
                        s += imgs[base + (y*step+dy)*c + (x*step+dx)]
                f.write(struct.pack('<f', s / (step*step) / 255.0))
        onehot = [0.0]*10
        onehot[labs[i]] = 1.0
        for v in onehot:
            f.write(struct.pack('<f', v))
PY
        # Video frames: synthetic moving-square video, 4x8x8 windows
        ${pkgs.ffmpeg}/bin/ffmpeg -y -f lavfi -i "color=c=black:s=32x32:d=2:r=10" \
          -f lavfi -i "color=c=white:s=6x6:d=2:r=10" \
          -filter_complex "[0][1]overlay=x='10+8*sin(2*PI*t)':y='10+6*cos(2*PI*t)'" \
          -frames:v 20 moving.mp4 >/dev/null 2>&1 || true
        mkdir -p frames
        ${pkgs.ffmpeg}/bin/ffmpeg -y -i moving.mp4 -vf scale=8:8 frames/f_%03d.png >/dev/null 2>&1 || true
        ${pkgs.python3}/bin/python3 - "$out" <<'PY'
import struct, glob, random, sys, math
out = sys.argv[1]
frames = []
try:
    from PIL import Image
    for p in sorted(glob.glob('frames/f_*.png')):
        im = Image.open(p).convert('L')
        frames.append([v/255.0 for v in im.getdata()])
except Exception:
    pass
if not frames:
    for t in range(20):
        fr = []
        for y in range(8):
            for x in range(8):
                cx, cy = 4+2*math.sin(t*0.3), 4+1.5*math.cos(t*0.3)
                fr.append(1.0 if abs(x-cx)<1.5 and abs(y-cy)<1.5 else 0.0)
        frames.append(fr)
random.seed(42)
samples = []
for rep in range(40):
    for i in range(max(1, len(frames)-4)):
        x = []
        for k in range(4):
            for v in frames[(i+k) % len(frames)]:
                x.append(max(0.0, min(1.0, v + random.uniform(-0.03, 0.03))))
        y = 1.0 if i >= (len(frames)-4)//2 else 0.0
        samples.append((x, y))
with open(out + '/share/type-nn/video_frames.bin','wb') as f:
    nin = len(samples[0][0])
    f.write(struct.pack('<III', len(samples), nin, 1))
    for x, y in samples:
        for v in x: f.write(struct.pack('<f', v))
        f.write(struct.pack('<f', y))
PY
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
          echo "datasets: iris wine wdbc diabetes ionosphere spambase mnist mnist_large video video_large"
          mkdir -p data
          cp -f --no-preserve=mode $TYPE_NN_DATA/*.data $TYPE_NN_DATA/diabetes.tab.txt data/
          echo "make test    -> unit tests, gradient checks"
          echo "./bench.sh   -> graph-type-nn, type-nn, type-nn-overfit, c-mlp board (BOARD.md)"
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
        buildPhase = "make test_type_nn test_type_nn_overfit test_cmlp test_type_nn_graph bench && ./test_type_nn && ./test_type_nn_overfit && ./test_cmlp && ./test_type_nn_graph";
        installPhase = ''
          mkdir -p $out/bin $out/share/type-nn
          cp test_type_nn test_type_nn_overfit test_cmlp test_type_nn_graph bench bench.sh $out/bin/
          cp -R ${datasets}/share/type-nn/. $out/share/type-nn/
        '';
      };
    in
    {
      packages.${system} = {
        default = package;
        inherit proofs;
      };
      apps.${system} = {
        bench = {
          type = "app";
          program = "${package}/bin/bench.sh";
        };
        default = {
          type = "app";
          program = "${package}/bin/bench.sh";
        };
      };
      checks.${system} = {
        tests = pkgs.stdenv.mkDerivation {
          name = "type-nn-tests";
          src = ./.;
          nativeBuildInputs = [ pkgs.gcc pkgs.gnumake ];
          buildPhase = "make test";
          installPhase = "mkdir -p $out && echo ok > $out/result";
        };
        tests-asan = pkgs.stdenv.mkDerivation {
          name = "type-nn-tests-asan";
          src = ./.;
          nativeBuildInputs = [ pkgs.clang pkgs.gnumake ];
          buildPhase = "make test-asan";
          installPhase = "mkdir -p $out && echo ok > $out/result";
        };
      };
      devShells.${system}.default = devShell;
      formatter.${system} = nixpkgs.legacyPackages.${system}.nixfmt-tree;
    };
}
