/*
 * bench_type_nn.c – wall-clock + RSS measurements for the C AND-OR net.
 *
 * Prints one JSON object per task so bench.sh can merge with the PyTorch side.
 */
#define _POSIX_C_SOURCE 200809L
#include "type_nn.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

static double wall_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static long rss_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmRSS:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

static long hwm_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    if (!f) return -1;
    char line[256];
    long kb = -1;
    while (fgets(line, sizeof(line), f)) {
        if (strncmp(line, "VmHWM:", 6) == 0) {
            sscanf(line + 6, "%ld", &kb);
            break;
        }
    }
    fclose(f);
    return kb;
}

static double **mat_alloc(size_t rows, size_t cols, int seed, int kind)
{
    srand((unsigned)seed);
    double **m = (double **)malloc(rows * sizeof(double *));
    for (size_t i = 0; i < rows; i++) {
        m[i] = (double *)malloc(cols * sizeof(double));
        for (size_t j = 0; j < cols; j++) {
            if (kind == 0) {          /* U[-1,1] */
                m[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
            } else if (kind == 1) {   /* XOR-style bits */
                m[i][j] = (double)(rand() & 1);
            } else {
                m[i][j] = 0.0;
            }
        }
    }
    return m;
}

static void mat_free(double **m, size_t rows)
{
    for (size_t i = 0; i < rows; i++) free(m[i]);
    free(m);
}

static double mse(Network *net, double **X, double **Y, size_t n, size_t in, size_t out)
{
    double *pred = (double *)calloc(out, sizeof(double));
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        network_predict(net, X[i], in, pred, out);
        for (size_t k = 0; k < out; k++) {
            double d = pred[k] - Y[i][k];
            acc += d * d;
        }
    }
    free(pred);
    return acc / (double)n;
}

static void emit(const char *task, double train_s, double infer_s,
                 size_t infer_n, long rss, long hwm,
                 size_t params, size_t nbytes, double final_mse, size_t depth,
                 size_t n_samples, double acc)
{
    double us_per = infer_n ? (infer_s * 1e6 / (double)infer_n) : 0.0;
    printf(
        "{\"impl\":\"type-nn\",\"task\":\"%s\",\"train_s\":%.6f,"
        "\"infer_s\":%.6f,\"infer_n\":%zu,\"us_per_infer\":%.3f,"
        "\"rss_kb\":%ld,\"hwm_kb\":%ld,\"params\":%zu,\"nbytes\":%zu,"
        "\"mse\":%.8f,\"depth\":%zu,\"n\":%zu,\"acc\":%.6f}\n",
        task, train_s, infer_s, infer_n, us_per,
        rss, hwm, params, nbytes, final_mse, depth, n_samples, acc);
}

static void bench_xor(void)
{
    srand(34972);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, 0.08);
    network_init_weights(net);

    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    double t0 = wall_s();
    network_train(net, X, Y, 4, 400);
    double train_s = wall_s() - t0;

    const size_t reps = 20000;
    double pred[1];
    double t1 = wall_s();
    for (size_t r = 0; r < reps; r++)
        network_predict(net, Xd[r & 3], 2, pred, 1);
    double infer_s = wall_s() - t1;

    emit("xor", train_s, infer_s, reps, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, 4, 2, 1), network_depth(net), 4, -1.0);
    network_free(net);
}

static void bench_quadratic(void)
{
    /* target = x0*x1 + 0.25*x0  — a genuine rank-2 polynomial */
    const size_t N = 64;
    double **X = mat_alloc(N, 2, 7, 0);
    double **Y = mat_alloc(N, 1, 0, 2);
    for (size_t i = 0; i < N; i++)
        Y[i][0] = X[i][0] * X[i][1] + 0.25 * X[i][0];

    srand(7);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, 0.04);
    network_init_weights(net);

    double t0 = wall_s();
    network_train(net, X, Y, N, 200);
    double train_s = wall_s() - t0;

    const size_t reps = 5000;
    double pred[1];
    double t1 = wall_s();
    for (size_t r = 0; r < reps; r++)
        network_predict(net, X[r % N], 2, pred, 1);
    double infer_s = wall_s() - t1;

    emit("quadratic", train_s, infer_s, reps, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, N, 2, 1), network_depth(net), N, -1.0);
    network_free(net);
    mat_free(X, N);
    mat_free(Y, N);
}

static void bench_mlp_scale(void)
{
    /* 32 → 16 → 8 synthetic regression, 256 samples. */
    const size_t N = 256, IN = 32, HID = 16, OUT = 8;
    double **X = mat_alloc(N, IN, 11, 0);
    double **Y = mat_alloc(N, OUT, 13, 0);
    /* squash targets a bit so clips do not dominate */
    for (size_t i = 0; i < N; i++)
        for (size_t k = 0; k < OUT; k++)
            Y[i][k] *= 0.3;

    srand(11);
    Network *net = network_create(IN, HID);
    network_add_layer(net, HID, OUT);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, 0.01);
    network_init_weights(net);

    double t0 = wall_s();
    network_train(net, X, Y, N, 30);
    double train_s = wall_s() - t0;

    const size_t reps = 1000;
    double *pred = (double *)calloc(OUT, sizeof(double));
    double t1 = wall_s();
    for (size_t r = 0; r < reps; r++)
        network_predict(net, X[r % N], IN, pred, OUT);
    double infer_s = wall_s() - t1;
    free(pred);

    emit("mlp32x16x8", train_s, infer_s, reps, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, N, IN, OUT), network_depth(net), N, -1.0);
    network_free(net);
    mat_free(X, N);
    mat_free(Y, N);
}


static double acc_class(Network *net, Dataset *ds)
{
    double *pred = (double *)calloc(ds->out, sizeof(double));
    size_t correct = 0;
    for (size_t i = 0; i < ds->n; i++) {
        network_predict(net, ds->X[i], ds->in, pred, ds->out);
        if (ds->out == 1) {
            int got = pred[0] >= 0.5;
            int want = ds->Y[i][0] >= 0.5;
            if (got == want) correct++;
        } else {
            size_t gp = 0, wp = 0;
            for (size_t k = 1; k < ds->out; k++) {
                if (pred[k] > pred[gp]) gp = k;
                if (ds->Y[i][k] > ds->Y[i][wp]) wp = k;
            }
            if (gp == wp) correct++;
        }
    }
    free(pred);
    return (double)correct / (double)ds->n;
}

static int bench_real(const char *task, const char *file,
                      int (*loader)(const char *, Dataset *),
                      size_t epochs, double lr, size_t infer_reps)
{
    char *path = dataset_find(file);
    if (!path) {
        fprintf(stderr, "skip %s: %s not found (set TYPE_NN_DATA or make data)\n",
                task, file);
        return 0;
    }
    Dataset ds;
    if (loader(path, &ds) != 0 || ds.n == 0) {
        fprintf(stderr, "skip %s: failed to parse %s\n", task, path);
        free(path);
        return 0;
    }
    dataset_standardize_inputs(&ds);
    if (!ds.classification) dataset_minmax_outputs(&ds);

    srand(34972);
    Network *net = network_create(ds.in, ds.out);
    network_set_dynamic(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, lr);
    network_init_weights(net);

    double t0 = wall_s();
    network_train(net, ds.X, ds.Y, ds.n, epochs);
    double train_s = wall_s() - t0;

    double *pred = (double *)calloc(ds.out, sizeof(double));
    double t1 = wall_s();
    for (size_t r = 0; r < infer_reps; r++)
        network_predict(net, ds.X[r % ds.n], ds.in, pred, ds.out);
    double infer_s = wall_s() - t1;
    free(pred);

    double acc = ds.classification ? acc_class(net, &ds) : -1.0;
    emit(task, train_s, infer_s, infer_reps, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, ds.X, ds.Y, ds.n, ds.in, ds.out),
         network_depth(net), ds.n, acc);
    network_free(net);
    dataset_free(&ds);
    free(path);
    return 1;
}

int main(int argc, char **argv)
{
    const char *task = (argc > 1) ? argv[1] : "all";
    if (!strcmp(task, "xor") || !strcmp(task, "all")) bench_xor();
    if (!strcmp(task, "quadratic") || !strcmp(task, "all")) bench_quadratic();
    if (!strcmp(task, "mlp32x16x8") || !strcmp(task, "all")) bench_mlp_scale();
    if (!strcmp(task, "iris") || !strcmp(task, "all") || !strcmp(task, "real"))
        bench_real("iris", "iris.data", dataset_load_iris, 250, 0.05, 2000);
    if (!strcmp(task, "wine") || !strcmp(task, "all") || !strcmp(task, "real"))
        bench_real("wine", "wine.data", dataset_load_wine, 200, 0.03, 2000);
    if (!strcmp(task, "wdbc") || !strcmp(task, "all") || !strcmp(task, "real"))
        bench_real("wdbc", "wdbc.data", dataset_load_wdbc, 80, 0.02, 1000);
    if (!strcmp(task, "diabetes") || !strcmp(task, "all") || !strcmp(task, "real"))
        bench_real("diabetes", "diabetes.tab.txt", dataset_load_diabetes, 150, 0.02, 2000);
    return 0;
}
