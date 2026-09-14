#define _POSIX_C_SOURCE 200809L
#include "type_nn_alt.h"
#include "dataset.h"
#include "bench_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double wall_s(void)
{
    return bench_wall_s();
}

static double time_alt_infer(AltNet *a, double **X, size_t n,
                             size_t min_reps, size_t *infer_n)
{
    double *pred = (double *)calloc(a->out ? a->out : 1, sizeof(double));
    volatile double sink = 0.0;
    size_t done = 0;
    size_t batch = min_reps ? min_reps : 1000;
    double t0 = wall_s();
    do {
        for (size_t r = 0; r < batch; r++) {
            a->forward(a->ctx, X[r % n], pred);
            sink += pred[0];
        }
        done += batch;
    } while (wall_s() - t0 < BENCH_INFER_MIN_S);
    double dt = wall_s() - t0;
    if (sink < -1e300) done++;
    free(pred);
    if (infer_n) *infer_n = done;
    return dt;
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
                 size_t n, double acc, size_t depth,
                 double dyn_scale, int dyn_depth, long dyn_params,
                 long layer_add, long layer_drop,
                 long or_add, long or_drop, long and_add, long and_drop,
                 double hold_mse, double hold_acc)
{
    double us = infer_n ? infer_s * 1e6 / (double)infer_n : 0.0;
    printf("{\"impl\":\"%s\",\"task\":\"%s\",\"train_s\":%.6f,"
           "\"infer_s\":%.6f,\"infer_n\":%zu,\"us_per_infer\":%.4f,"
           "\"rss_kb\":%ld,\"hwm_kb\":%ld,\"params\":%zu,\"nbytes\":%zu,"
           "\"mse\":%.8f,\"depth\":%zu,\"n\":%zu,\"acc\":%.6f,"
           "\"dyn_scale\":%.4f,\"dyn_depth\":%d,\"dyn_params\":%ld,"
           "\"layer_add\":%ld,\"layer_drop\":%ld,"
           "\"or_add\":%ld,\"or_drop\":%ld,\"and_add\":%ld,\"and_drop\":%ld,"
           "\"hold_mse\":%.8f,\"hold_acc\":%.6f}\n",
           impl, task, train_s, infer_s, infer_n, us,
           rss_kb(), rss_kb(), params, nbytes, mse, depth, n, acc,
           dyn_scale, dyn_depth, dyn_params, layer_add, layer_drop,
           or_add, or_drop, and_add, and_drop,
           hold_mse, hold_acc);
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
    /* Match torch F.mse_loss: mean over samples *and* output channels.
       Dividing only by n made 3-class iris look 3× worse than 1-class WDBC. */
    return acc / (double)(n * (a->out ? a->out : 1));
}

static double mse_idx(AltNet *a, Dataset *ds, const size_t *idx, size_t n)
{
    if (!n) return 0.0;
    double *pred = (double *)calloc(a->out, sizeof(double));
    double acc = 0.0;
    for (size_t t = 0; t < n; t++) {
        size_t i = idx[t];
        a->forward(a->ctx, ds->X[i], pred);
        for (size_t k = 0; k < a->out; k++) {
            double d = pred[k] - ds->Y[i][k];
            acc += d * d;
        }
    }
    free(pred);
    return acc / (double)(n * (a->out ? a->out : 1));
}

static double acc_idx(AltNet *a, Dataset *ds, const size_t *idx, size_t n)
{
    if (!ds->classification || !n) return -1.0;
    double *pred = (double *)calloc(a->out, sizeof(double));
    size_t ok = 0;
    for (size_t t = 0; t < n; t++) {
        size_t i = idx[t];
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
    return (double)ok / (double)n;
}

static void run_xy(const char *task, AltNet *a, double **X, double **Y,
                   size_t n, size_t epochs, double lr, size_t reps, double acc)
{
    srand(34972);
    a->init(a->ctx);
    TnnSnap before, after;
    type_nn_alt_snap(a, &before);
    size_t p0 = a->param_count ? a->param_count(a->ctx) : 0;
    double t0 = wall_s();
    type_nn_alt_train(a, X, Y, n, epochs, lr);
    double train_s = wall_s() - t0;
    type_nn_alt_snap(a, &after);
    double dyn_scale = 0.0;
    int dyn_depth = 0;
    type_nn_alt_dyn_score(&before, &after, &dyn_scale, &dyn_depth);
    size_t p1 = a->param_count ? a->param_count(a->ctx) : p0;
    long dyn_params = (long)p1 - (long)p0;
    size_t infer_n = 0;
    double infer_s = time_alt_infer(a, X, n, reps, &infer_n);
    size_t depth = a->depth ? a->depth(a->ctx) : 1;
    long ladd = a->n_add ? (long)a->n_add(a->ctx) : 0;
    long ldrop = a->n_drop ? (long)a->n_drop(a->ctx) : 0;
    long oa = a->or_add ? (long)a->or_add(a->ctx) : 0;
    long od = a->or_drop ? (long)a->or_drop(a->ctx) : 0;
    long aa = a->and_add ? (long)a->and_add(a->ctx) : 0;
    long ad = a->and_drop ? (long)a->and_drop(a->ctx) : 0;
    double m = mse_of(a, X, Y, n);
    emit(a->impl, task, train_s, infer_s, infer_n,
         a->param_count(a->ctx), a->nbytes(a->ctx),
         m, n, acc, depth, dyn_scale, dyn_depth, dyn_params,
         ladd, ldrop, oa, od, aa, ad, m, acc);
}

static void bench_xor(AltNet (*open)(size_t, size_t))
{
    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    AltNet a = open(2, 1);
    if (a.set_dynamic)
        a.set_dynamic(a.ctx, strncmp(a.impl, "type-nn-", 8) == 0);
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
    if (a.set_dynamic)
        a.set_dynamic(a.ctx, strncmp(a.impl, "type-nn-", 8) == 0);
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
    if (a.set_dynamic)
        a.set_dynamic(a.ctx, strncmp(a.impl, "type-nn-", 8) == 0);
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
    if (a.set_dynamic)
        a.set_dynamic(a.ctx, strncmp(a.impl, "type-nn-", 8) == 0);
    /* Split first — never after init — so every impl is judged on the
       same 70/30 cut. Weight init may then use libc rand(). */
    size_t *perm = (size_t *)malloc(ds.n * sizeof(size_t));
    dataset_perm(ds.n, DATASET_SPLIT_SEED, perm);
    size_t ntr = dataset_ntrain(ds.n);
    srand(34972);
    a.init(a.ctx);
    size_t nte = ds.n - ntr;
    double **Xtr = (double **)malloc(ntr * sizeof(double *));
    double **Ytr = (double **)malloc(ntr * sizeof(double *));
    for (size_t i = 0; i < ntr; i++) {
        Xtr[i] = ds.X[perm[i]];
        Ytr[i] = ds.Y[perm[i]];
    }
    TnnSnap before, after;
    type_nn_alt_snap(&a, &before);
    size_t p0 = a.param_count ? a.param_count(a.ctx) : 0;
    double t0 = wall_s();
    type_nn_alt_train(&a, Xtr, Ytr, ntr, epochs, lr);
    double train_s = wall_s() - t0;
    type_nn_alt_snap(&a, &after);
    double dyn_scale = 0.0;
    int dyn_depth = 0;
    type_nn_alt_dyn_score(&before, &after, &dyn_scale, &dyn_depth);
    size_t p1 = a.param_count ? a.param_count(a.ctx) : p0;
    long dyn_params = (long)p1 - (long)p0;
    size_t infer_n = 0;
    double infer_s = time_alt_infer(&a, ds.X, ds.n, reps, &infer_n);
    {
        long ladd = a.n_add ? (long)a.n_add(a.ctx) : 0;
        long ldrop = a.n_drop ? (long)a.n_drop(a.ctx) : 0;
        long oa = a.or_add ? (long)a.or_add(a.ctx) : 0;
        long od = a.or_drop ? (long)a.or_drop(a.ctx) : 0;
        long aa = a.and_add ? (long)a.and_add(a.ctx) : 0;
        long ad = a.and_drop ? (long)a.and_drop(a.ctx) : 0;
        double tr_mse = mse_idx(&a, &ds, perm, ntr);
        double te_mse = nte ? mse_idx(&a, &ds, perm + ntr, nte) : tr_mse;
        double tr_acc = acc_idx(&a, &ds, perm, ntr);
        double te_acc = nte ? acc_idx(&a, &ds, perm + ntr, nte) : tr_acc;
        emit(a.impl, task, train_s, infer_s, infer_n,
             a.param_count(a.ctx), a.nbytes(a.ctx),
             tr_mse, ds.n, tr_acc,
             a.depth ? a.depth(a.ctx) : 1, dyn_scale, dyn_depth, dyn_params,
             ladd, ldrop, oa, od, aa, ad, te_mse, te_acc);
    }
    free(Xtr); free(Ytr); free(perm);
    a.free(a.ctx);
    dataset_free(&ds);
    free(path);
}

static int env_epochs(int def)
{
    const char *e = getenv("TYPE_NN_EPOCHS");
    if (!e || !*e) return def;
    int v = atoi(e);
    return v > 0 ? v : def;
}

int main(int argc, char **argv)
{
    const char *task = (argc > 1) ? argv[1] : "all";
    const char *only = (argc > 2) ? argv[2] : NULL;
    for (size_t i = 0; i < type_nn_alt_count(); i++) {
        AltNet (*open)(size_t, size_t);
        const char *name;
        name = type_nn_alt_name(i);
        open = type_nn_alt_opener(i);
        if (!name || !open) continue;
        if (only && strcmp(only, name) != 0) continue;
        if (!strcmp(task, "xor") || !strcmp(task, "all")) bench_xor(open);
        if (!strcmp(task, "quadratic") || !strcmp(task, "all")) bench_quadratic(open);
        if (!strcmp(task, "mlp32x16x8") || !strcmp(task, "all")) bench_mlp(open);
        if (!strcmp(task, "iris") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "iris", "iris.data", dataset_load_iris, (size_t)env_epochs(250), 0.05, 2000);
        if (!strcmp(task, "wine") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "wine", "wine.data", dataset_load_wine, (size_t)env_epochs(200), 0.03, 2000);
        if (!strcmp(task, "wdbc") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "wdbc", "wdbc.data", dataset_load_wdbc, (size_t)env_epochs(80), 0.02, 1000);
        if (!strcmp(task, "diabetes") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "diabetes", "diabetes.tab.txt",
                       dataset_load_diabetes, (size_t)env_epochs(150), 0.02, 2000);
        if (!strcmp(task, "ionosphere") || !strcmp(task, "all") || !strcmp(task, "real"))
            bench_real(open, "ionosphere", "ionosphere.data",
                       dataset_load_ionosphere, (size_t)env_epochs(120), 0.02, 1000);
    }
    return 0;
}
