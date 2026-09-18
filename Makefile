CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wformat -I.
LDFLAGS ?= -lm

.PHONY: all test test-asan demo bench data clean lean lean-clean

LN = type_nn_ln.c type_nn_ln.h type_nn_layer.c type_nn_layer.h type_nn_grow.c type_nn_grow.h

# Board models that stay: type-nn. c-mlp lives in bench_alts.
MODELS = type_nn_model.c

all: type-nn test_type_nn

type-nn: type_nn.c run.c type_nn.h $(LN) $(MODELS)
	$(CC) $(CFLAGS) -o $@ type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c \
	    $(MODELS) run.c $(LDFLAGS)

test_type_nn: type_nn.c test_type_nn.c type_nn.h $(LN) $(MODELS)
	$(CC) $(CFLAGS) -o $@ type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c \
	    $(MODELS) test_type_nn.c $(LDFLAGS)

bench_type_nn: type_nn.c bench_type_nn.c dataset.c type_nn.h dataset.h $(LN) $(MODELS)
	$(CC) $(CFLAGS) -o $@ type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c \
	    bench_type_nn.c dataset.c $(MODELS) $(LDFLAGS)

bench_alts: bench_alts.c dataset.c dataset.h type_nn_alt.h type_nn_alt.c type_nn_cmlp.c type_nn_cmlp.h
	$(CC) $(CFLAGS) -o $@ bench_alts.c dataset.c type_nn_alt.c type_nn_cmlp.c $(LDFLAGS)

test_alts: test_alts.c type_nn_alt.h type_nn_alt.c type_nn_cmlp.c type_nn_cmlp.h
	$(CC) $(CFLAGS) -o $@ test_alts.c type_nn_alt.c type_nn_cmlp.c $(LDFLAGS)

test: test_type_nn test_alts
	./test_type_nn
	./test_alts

test-asan: type_nn.c test_type_nn.c type_nn.h $(LN) $(MODELS)
	$(CC) -std=c11 -g -O0 -Wall -Wextra -I. -fsanitize=address,undefined \
	    -o test_type_nn_asan type_nn.c type_nn_ln.c type_nn_layer.c type_nn_grow.c \
	    $(MODELS) test_type_nn.c $(LDFLAGS)
	./test_type_nn_asan

demo: type-nn
	./type-nn

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

bench: bench_type_nn bench_alts
	chmod +x bench.sh
	./bench.sh

lean:
	cd lean && lake build

lean-clean:
	rm -rf lean/.lake lean/lake-manifest.json

clean: lean-clean
	rm -f type-nn test_type_nn test_type_nn_asan bench_type_nn bench_alts test_alts
