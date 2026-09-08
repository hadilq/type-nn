CC      ?= gcc
CFLAGS  ?= -std=c11 -O2 -Wall -Wextra -Wformat -I.
LDFLAGS ?= -lm

.PHONY: all test test-asan demo bench clean

all: vortex test_vortex

vortex: vortex.c run.c vortex.h
	$(CC) $(CFLAGS) -o $@ vortex.c run.c $(LDFLAGS)

test_vortex: vortex.c test_vortex.c vortex.h
	$(CC) $(CFLAGS) -o $@ vortex.c test_vortex.c $(LDFLAGS)

bench_vortex: vortex.c bench_vortex.c vortex.h
	$(CC) $(CFLAGS) -o $@ vortex.c bench_vortex.c $(LDFLAGS)

test: test_vortex
	./test_vortex

test-asan: vortex.c test_vortex.c vortex.h
	$(CC) -std=c11 -g -O0 -Wall -Wextra -I. -fsanitize=address,undefined \
	    -o test_vortex_asan vortex.c test_vortex.c $(LDFLAGS)
	./test_vortex_asan

demo: vortex
	./vortex

bench: bench_vortex
	chmod +x bench.sh
	./bench.sh

clean:
	rm -f vortex test_vortex test_vortex_asan bench_vortex
