CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wformat -I.
LDFLAGS ?= -lm

TNN  = type_nn.c type_nn_scale.c type_nn_graph.c
TNNO = type_nn_overfit.c type_nn_overfit_scale.c
HDR  = common.h type_nn.h type_nn_scale.h type_nn_overfit.h type_nn_overfit_scale.h c_mlp.h type_nn_graph.h

.PHONY: all test test-asan bench run-bench data coq clean

all: test_type_nn test_type_nn_overfit test_cmlp test_type_nn_graph bench

test_type_nn: test_type_nn.c $(TNN) $(HDR)
	$(CC) $(CFLAGS) -o $@ test_type_nn.c $(TNN) $(LDFLAGS)

test_type_nn_overfit: test_type_nn_overfit.c $(TNNO) $(HDR)
	$(CC) $(CFLAGS) -o $@ test_type_nn_overfit.c $(TNNO) $(LDFLAGS)

test_cmlp: test_cmlp.c c_mlp.c $(HDR)
	$(CC) $(CFLAGS) -o $@ test_cmlp.c c_mlp.c $(LDFLAGS)

test_type_nn_graph: test_type_nn_graph.c $(TNN) $(HDR)
	$(CC) $(CFLAGS) -o $@ test_type_nn_graph.c $(TNN) $(LDFLAGS)

bench: bench.c $(TNN) $(TNNO) c_mlp.c dataset.c dataset.h bench_time.h $(HDR)
	$(CC) $(CFLAGS) -o $@ bench.c $(TNN) $(TNNO) c_mlp.c dataset.c $(LDFLAGS)

run-bench: bench
	./bench all | tee bench.jsonl

test: test_type_nn test_type_nn_overfit test_cmlp test_type_nn_graph
	./test_type_nn
	./test_type_nn_overfit
	./test_cmlp
	./test_type_nn_graph

test-asan: test_type_nn.c test_type_nn_overfit.c test_cmlp.c test_type_nn_graph.c $(TNN) $(TNNO) c_mlp.c $(HDR)
	$(CC) -std=c11 -g -O1 -Wall -Wextra -I. -fsanitize=address,undefined -D_FORTIFY_SOURCE=0 \
	    -o test_type_nn_asan test_type_nn.c $(TNN) $(LDFLAGS)
	$(CC) -std=c11 -g -O1 -Wall -Wextra -I. -fsanitize=address,undefined -D_FORTIFY_SOURCE=0 \
	    -o test_type_nn_overfit_asan test_type_nn_overfit.c $(TNNO) $(LDFLAGS)
	$(CC) -std=c11 -g -O1 -Wall -Wextra -I. -fsanitize=address,undefined -D_FORTIFY_SOURCE=0 \
	    -o test_cmlp_asan test_cmlp.c c_mlp.c $(LDFLAGS)
	$(CC) -std=c11 -g -O1 -Wall -Wextra -I. -fsanitize=address,undefined -D_FORTIFY_SOURCE=0 \
	    -o test_type_nn_graph_asan test_type_nn_graph.c $(TNN) $(LDFLAGS)
	./test_type_nn_asan
	./test_type_nn_overfit_asan
	./test_cmlp_asan
	./test_type_nn_graph_asan

bench-run: bench
	./bench.sh

data:
	mkdir -p data
	curl -fsSL -o data/iris.data \
	  https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
	curl -fsSL -o data/wine.data \
	  https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data
	curl -fsSL -o data/wdbc.data \
	  https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data
	curl -fsSL -o data/diabetes.tab.txt \
	  https://www4.stat.ncsu.edu/~boos/var.select/diabetes.tab.txt
	curl -fsSL -o data/ionosphere.data \
	  https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
	curl -fsSL -o data/spambase.data \
	  https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
	@echo 'mnist_*.bin and video_*.bin are built by flake.nix: run nix develop (sets TYPE_NN_DATA).'

coq:
	@if command -v coqc >/dev/null 2>&1; then $(MAKE) -C coqLang; \
	else echo "coqc not on PATH (nix develop provides it)"; exit 1; fi

clean:
	rm -f test_type_nn test_type_nn_overfit test_cmlp test_type_nn_graph bench *_asan
	$(MAKE) -C coqLang clean
