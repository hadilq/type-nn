#define _POSIX_C_SOURCE 200809L
#include "type_nn_alt.h"
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

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

static void emit(const char *impl, const char *task,
                 double train_s, double infer_s, size_t infer_n,
                 size_t params, size_t nbytes, double mse,
                 size_t n, double acc)
{
    double us = infer_n ? infer_s * 1e6 / (double)infer_n : 0.0;
    printf("{\"impl\":\"%s\",\"task\":\"%s\",\"train_s\":%.6f,"
           "\"infer_s\":%.6f,\"infer_n\":%zu,\"us_per_infer\":%.3f,"
           "\"rss_kb\":%ld,\"hwm_kb\":%ld,\"params\":%zu,\"nbytes\":%zu,"
           "\"mse\":%.8f,\"depth\":1,\"n\":%zu,\"acc\":%.6f}\n",
           impl, task, train_s, infer_s, infer_n, us,
           rss_kb(), rss_kb(), params, nbytes, mse, n, acc);
}

static double mse_of(AltNet *a, double **X, double **Y, size_t n)
{
    double *pred = (double *)calloc(a->out, sizeof(double));
    double acc = 0.0;
    for (size_t i = 0; i < n; i++) {
        a->forward(a->ctx, X[i], pred);
        for (size_t k = 0; k < a->out; k++) {
            double d = pred[k] - Y[i][k];
            acc += d * d;
        }
    }
    free(pred);
    return acc / (double)n;
}

static double acc_of(AltNet *a, Dataset *ds)
{
    if (!ds->classification) return -1.0;
    double *pred = (double *)calloc(a->out, sizeof(double));
    size_t ok = 0;
    for (size_t i = 0; i < ds->n; i++) {
        a->forward(a->ctx, ds->X[i], pred);
        if (ds->out == 1) {
            if ((pred[0] >= 0.5) == (ds->Y[i][0] >= 0.5)) ok++;
        } else {
            size_t gp = 0, wp = 0;
            for (size_t k = 1; k < ds->out; k++) {
                if (pred[k] > pred[gp]) gp = k;
                if (ds->Y[i][k] > ds->Y[i][wp]) wp = k;
            }
            if (gp == wp) ok++;
        }
    }
    free(pred);
    return (double)ok / (double)ds->n;
}

static void run_xy(const char *task, AltNet *a, double **X, double **Y,
                   size_t n, size_t epochs, double lr, size_t reps, double acc)
{
    srand(34972);
    a->init(a->ctx);
    double t0 = wall_s();
    type_nn_alt_train(a, X, Y, n, epochs, lr);
    double train_s = wall_s() - t0;
    double *pred = (double *)calloc(a->out, sizeof(double));
    double t1 = wall_s();
    for (size_t r = 0; r < reps; r++)
        a->forward(a->ctx, X[r % n], pred);
    double infer_s = wall_s() - t1;
    free(pred);
    emit(a->impl, task, train_s, infer_s, reps,
         a->param_count(a->ctx), a->nbytes(a->ctx),
         mse_of(a, X, Y, n), n, acc);
}

static void bench_xor(AltNet (*open)(size_t, size_t))
{
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    AltNet a = open(2, 1);
    a.set_dynamic(a.ctx, 0);
    run_xy("xor", &a, X, Y, 4, 400, 0.08, 20000, -1.0);
    a.free(a.ctx);
}

static void bench_quadratic(AltNet (*open)(size_t, size_t))
{
    const size_t N = 64;
    srand(7);
    double **X = (double **)malloc(N * sizeof(double *));
    double **Y = (double **)malloc(N * sizeof(double *));
    for (size_t i = 0; i < N; i++) {
        X[i] = (double *)malloc(2 * sizeof(double));
        Y[i] = (double *)malloc(sizeof(double));
        X[i][0] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        X[i][1] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        Y[i][0] = X[i][0] * X[i][1] + 0.25 * X[i][0];
    }
    AltNet a = open(2, 1);
    a.set_dynamic(a.ctx, 0);
    run_xy("quadratic", &a, X, Y, N, 200, 0.04, 5000, -1.0);
    a.free(a.ctx);
    for (size_t i = 0; i < N; i++) { free(X[i]); free(Y[i]); }
    free(X); free(Y);
}

static void bench_mlp(AltNet (*open)(size_t, size_t))
{
    const size_t N = 256, IN = 32, OUT = 8;
    srand(11);
    double **X = (double **)malloc(N * sizeof(double *));
    double **Y = (double **)malloc(N * sizeof(double *));
    for (size_t i = 0; i < N; i++) {
        X[i] = (double *)malloc(IN * sizeof(double));
        Y[i] = (double *)malloc(OUT * sizeof(double));
        for (size_t j = 0; j < IN; j++)
            X[i][j] = (double)rand() / RAND_MAX * 2.0 - 1.0;
        for (size_t j = 0; j < OUT; j++)
            Y[i][j] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * 0.3;
    }
    AltNet a = open(IN, OUT);
    a.set_dynamic(a.ctx, 0);
    run_xy("mlp32x16x8", &a, X, Y, N, 30, 0.01, 1000, -1.0);
    a.free(a.ctx);
    for (size_t i = 0; i < N; i++) { free(X[i]); free(Y[i]); }
    free(X); free(Y);
}

static void bench_real(AltNet (*open)(size_t, size_t),
                       const char *task, const char *file,
                       int (*loader)(const char *, Dataset *),
                       size_t epochs, double lr, size_t reps)
{
    char *path = dataset_find(file);
    if (!path) {
        fprintf(stderr, "skip %s/%s: data missing\n", task, file);
        return;
    }
    Dataset ds;
    if (loader(path, &ds) != 0) {
        free(path);
        return;
    }
    dataset_standardize_inputs(&ds);
    if (!ds.classification) dataset_minmax_outputs(&ds);
    AltNet a = open(ds.in, ds.out);
    a.set_dynamic(a.ctx, 0);
    srand(34972);
    a.init(a.ctx);
    double t0 = wall_s();
    type_nn_alt_train(&a, ds.X, ds.Y, ds.n, epochs, lr);
    double train_s = wall_s() - t0;
    double *pred = (double *)calloc(a.out, sizeof(double));
    double t1 = wall_s();
    for (size_t r = 0; r < reps; r++)
        a.forward(a.ctx, ds.X[r % ds.n], pred);
    double infer_s = wall_s() - t1;
    free(pred);
    emit(a.impl, task, train_s, infer_s, reps,
         a.param_count(a.ctx), a.nbytes(a.ctx),
         mse_of(&a, ds.X, ds.Y, ds.n), ds.n, acc_of(&a, &ds));
    a.free(a.ctx);
    dataset_free(&ds);
    free(path);
}

int main(int argc, char **argv)
{
    const char *task = (argc > 1) ? argv[1] : "all";
    const char *only = (argc > 2) ? argv[2] : NULL;
    for (size_t i = 0; i < type_nn_alt_count(); i++) {
        if (only && strcmp(only, type_nn_alt_name(i)) != 0) continue;
        AltNet (*open)(size_t, size_t) = type_nn_alt_opener(i);
        if (!strcmp(task, "xor") || !strcmp(task, "all")) bench_xor(open);
        if (!strcmp(task, "quadratic") || !strcmp(task, "all")) bench_quadratic(open);
        if (!strcmp(task, "mlp32x16x8") || !strcmp(task, "all")) bench_mlp(open);
        if (!strcmp(task, "iris") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "iris", "iris.data", dataset_load_iris, 250, 0.05, 2000);
        if (!strcmp(task, "wine") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "wine", "wine.data", dataset_load_wine, 200, 0.03, 2000);
        if (!strcmp(task, "wdbc") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "wdbc", "wdbc.data", dataset_load_wdbc, 80, 0.02, 1000);
        if (!strcmp(task, "diabetes") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "diabetes", "diabetes.tab.txt",
                       dataset_load_diabetes, 150, 0.02, 2000);
    }
    return 0;
}
