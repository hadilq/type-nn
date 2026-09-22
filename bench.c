/*
 * bench.c — one harness for every board model.
 *
 * Both models go through the same code path: same split, same
 * train-only standardisation, same per-epoch shuffle, same mean-MSE
 * gradient (y − t)/m, same epochs and lr, same metrics, same timer.
 * The only difference is the vtable.
 *
 * Each (task, model) is trained from TNN_SEEDS initialisation seeds
 * (default 5) on the one fixed 70/30 split, and the board reports the
 * mean and standard deviation across seeds. A single seed is not a
 * measurement on data sets this small.
 *
 *   ./bench [task|all] [type-nn|type-nn-overfit|c-mlp|all]
 */
#define _POSIX_C_SOURCE 200809L
#include "type_nn.h"
#include "type_nn_overfit.h"
#include "c_mlp.h"
#include "common.h"
#include "dataset.h"
#include "bench_time.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── model vtable ──────────────────────────────────────────────────── */

typedef struct {
    const char *name;
    void         *(*create)(size_t in, size_t out, unsigned seed);
    void          (*begin)(void *m, size_t n, size_t epochs, double lr);
    void          (*training)(void *m, int on);
    const double *(*forward)(void *m, const double *x);
    void          (*backward)(void *m, const double *dy);
    void          (*epoch_end)(void *m);
    void          (*end)(void *m);
    size_t        (*params)(const void *m);
    void          (*counters)(const void *m, double *c); /* 8 slots */
    void          (*destroy)(void *m);
} Model;

static void *tn_create(size_t in, size_t out, unsigned s) { return tnn_create(in, out, s); }
static void  tn_begin(void *m, size_t n, size_t e, double lr) { tnn_begin(m, n, e, lr); }
static void  tn_training(void *m, int on) { ((TypeNN *)m)->training = on; }
static const double *tn_forward(void *m, const double *x) { return tnn_forward(m, x); }
static void  tn_backward(void *m, const double *dy) { tnn_backward(m, dy); }
static void  tn_epoch(void *m) { tnn_epoch_end(m); }
static void  tn_end(void *m) { tnn_end(m); }
static size_t tn_params(const void *m) { return tnn_params(m); }
static void  tn_counters(const void *mm, double *c)
{
    const TypeNN *m = mm;
    c[0] = (double)m->init_depth; c[1] = (double)m->depth;
    c[2] = m->or_add;  c[3] = m->or_drop;
    c[4] = m->and_add; c[5] = m->and_drop;
    c[6] = m->layer_add; c[7] = m->layer_drop;
}
static void  tn_destroy(void *m) { tnn_free(m); }

static void *to_create(size_t in, size_t out, unsigned s) { return tnno_create(in, out, s); }
static void  to_begin(void *m, size_t n, size_t e, double lr) { tnno_begin(m, n, e, lr); }
static void  to_training(void *m, int on) { ((TypeNNOverfit *)m)->training = on; }
static const double *to_forward(void *m, const double *x) { return tnno_forward(m, x); }
static void  to_backward(void *m, const double *dy) { tnno_backward(m, dy); }
static void  to_epoch(void *m) { tnno_epoch_end(m); }
static void  to_end(void *m) { tnno_end(m); }
static size_t to_params(const void *m) { return tnno_params(m); }
static void  to_counters(const void *mm, double *c)
{
    const TypeNNOverfit *m = mm;
    c[0] = (double)m->init_depth; c[1] = (double)m->depth;
    c[2] = m->or_add;  c[3] = m->or_drop;
    c[4] = m->and_add; c[5] = m->and_drop;
    c[6] = m->layer_add; c[7] = m->layer_drop;
}
static void  to_destroy(void *m) { tnno_free(m); }

static void *cm_create(size_t in, size_t out, unsigned s) { return cmlp_create(in, out, s); }
static void  cm_begin(void *m, size_t n, size_t e, double lr) { cmlp_begin(m, n, e, lr); }
static void  cm_training(void *m, int on) { (void)m; (void)on; }
static const double *cm_forward(void *m, const double *x) { return cmlp_forward(m, x); }
static void  cm_backward(void *m, const double *dy) { cmlp_backward(m, dy); }
static void  cm_noop(void *m) { (void)m; }
static size_t cm_params(const void *m) { return cmlp_params(m); }
static void  cm_counters(const void *m, double *c)
{
    (void)m;
    memset(c, 0, 8 * sizeof(double));
    c[0] = c[1] = 2.0;
}
static void  cm_destroy(void *m) { cmlp_free(m); }

static const Model MODELS[] = {
    { "type-nn", tn_create, tn_begin, tn_training, tn_forward, tn_backward,
      tn_epoch, tn_end, tn_params, tn_counters, tn_destroy },
    { "type-nn-overfit", to_create, to_begin, to_training, to_forward, to_backward,
      to_epoch, to_end, to_params, to_counters, to_destroy },
    { "c-mlp", cm_create, cm_begin, cm_training, cm_forward, cm_backward,
      cm_noop, cm_noop, cm_params, cm_counters, cm_destroy },
};
#define N_MODELS (sizeof(MODELS) / sizeof(MODELS[0]))

/* ── shared training loop ──────────────────────────────────────────── */

static void train(const Model *md, void *m, double **X, double **Y,
                  size_t n, size_t out, size_t epochs, double lr)
{
    size_t *ord = (size_t *)malloc(n * sizeof(size_t));
    double *dy = (double *)malloc(out * sizeof(double));
    md->begin(m, n, epochs, lr);
    for (size_t ep = 0; ep < epochs; ep++) {
        for (size_t i = 0; i < n; i++) ord[i] = i;
        unsigned st = DATASET_SPLIT_SEED ^ (unsigned)((ep + 1u) * 0x9E3779B9u);
        for (size_t i = n; i > 1; i--) {
            size_t j = (size_t)(tnn_xorshift32(&st) % (unsigned)i);
            size_t t = ord[i - 1]; ord[i - 1] = ord[j]; ord[j] = t;
        }
        md->training(m, 1);
        for (size_t s = 0; s < n; s++) {
            size_t si = ord[s];
            const double *y = md->forward(m, X[si]);
            for (size_t k = 0; k < out; k++)
                dy[k] = (y[k] - Y[si][k]) / (double)out;   /* mean-MSE */
            md->backward(m, dy);
        }
        md->training(m, 0);
        md->epoch_end(m);
    }
    md->end(m);
    free(ord);
    free(dy);
}

static double eval_mse(const Model *md, void *m, double **X, double **Y,
                       size_t n, size_t out)
{
    double s = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double *y = md->forward(m, X[i]);
        for (size_t k = 0; k < out; k++) {
            double d = y[k] - Y[i][k];
            s += d * d;
        }
    }
    return s / (double)(n * out);
}

static double eval_acc(const Model *md, void *m, double **X, double **Y,
                       size_t n, size_t out)
{
    size_t ok = 0;
    for (size_t i = 0; i < n; i++) {
        const double *y = md->forward(m, X[i]);
        if (out == 1) {
            ok += (y[0] >= 0.5) == (Y[i][0] >= 0.5);
        } else {
            size_t gp = 0, wp = 0;
            for (size_t k = 1; k < out; k++) {
                if (y[k] > y[gp]) gp = k;
                if (Y[i][k] > Y[i][wp]) wp = k;
            }
            ok += gp == wp;
        }
    }
    return (double)ok / (double)n;
}

static double time_infer(const Model *md, void *m, double **X, size_t n)
{
    volatile double sink = 0.0;
    size_t done = 0;
    double t0 = bench_wall_s(), dt;
    do {
        for (size_t r = 0; r < 1000; r++) sink += md->forward(m, X[r % n])[0];
        done += 1000;
        dt = bench_wall_s() - t0;
    } while (dt < BENCH_INFER_MIN_S);
    (void)sink;
    return dt * 1e6 / (double)done;
}

/* ── statistics over seeds ─────────────────────────────────────────── */

typedef struct { double s, ss; size_t n; } Acc;
static void acc_add(Acc *a, double v) { a->s += v; a->ss += v * v; a->n++; }
static double acc_mean(const Acc *a) { return a->n ? a->s / (double)a->n : NAN; }
static double acc_sd(const Acc *a)
{
    if (a->n < 2) return 0.0;
    double m = acc_mean(a);
    double v = (a->ss - (double)a->n * m * m) / (double)(a->n - 1);
    return v > 0.0 ? sqrt(v) : 0.0;
}

static size_t n_seeds(void)
{
    const char *e = getenv("TNN_SEEDS");
    long v = e ? atol(e) : 5;
    return v < 1 ? 1 : (size_t)v;
}

static void jnum(const char *key, double v, int last)
{
    if (isfinite(v)) printf("\"%s\":%.8g%s", key, v, last ? "" : ",");
    else printf("\"%s\":null%s", key, last ? "" : ",");
}

/* One (task, model) cell: train from every seed, print one JSON line. */
static void run_cell(const Model *md, const char *task,
                     double **Xtr, double **Ytr, size_t ntr,
                     double **Xte, double **Yte, size_t nte,
                     size_t in, size_t out, int classify,
                     size_t epochs, double lr)
{
    Acc hm = {0}, ha = {0}, tm = {0}, ta = {0}, pa = {0}, ts = {0};
    Acc cnt[8];
    memset(cnt, 0, sizeof(cnt));
    double us = NAN;
    size_t S = n_seeds();
    for (size_t s = 0; s < S; s++) {
        unsigned seed = DATASET_SPLIT_SEED + 7919u * (unsigned)(s + 1);
        void *m = md->create(in, out, seed);
        double t0 = bench_wall_s();
        train(md, m, Xtr, Ytr, ntr, out, epochs, lr);
        acc_add(&ts, bench_wall_s() - t0);
        acc_add(&tm, eval_mse(md, m, Xtr, Ytr, ntr, out));
        if (classify) acc_add(&ta, eval_acc(md, m, Xtr, Ytr, ntr, out));
        if (nte) {
            acc_add(&hm, eval_mse(md, m, Xte, Yte, nte, out));
            if (classify) acc_add(&ha, eval_acc(md, m, Xte, Yte, nte, out));
        }
        acc_add(&pa, (double)md->params(m));
        double c[8];
        md->counters(m, c);
        for (int i = 0; i < 8; i++) acc_add(&cnt[i], c[i]);
        if (getenv("TNN_VERBOSE"))
            fprintf(stderr, "  %s %s seed %zu: hold_mse %.5f train_mse %.2e params %zu"
                    " layers %g\n", task, md->name, s,
                    nte ? eval_mse(md, m, Xte, Yte, nte, out) : NAN,
                    eval_mse(md, m, Xtr, Ytr, ntr, out), md->params(m), c[1]);
        if (s + 1 == S) us = time_infer(md, m, Xtr, ntr);
        md->destroy(m);
    }
    printf("{\"impl\":\"%s\",\"task\":\"%s\",\"seeds\":%zu,\"epochs\":%zu,\"lr\":%g,",
           md->name, task, S, epochs, lr);
    jnum("hold_mse", acc_mean(&hm), 0); jnum("hold_mse_sd", acc_sd(&hm), 0);
    jnum("hold_acc", acc_mean(&ha), 0); jnum("hold_acc_sd", acc_sd(&ha), 0);
    jnum("mse", acc_mean(&tm), 0);      jnum("acc", acc_mean(&ta), 0);
    jnum("params", acc_mean(&pa), 0);   jnum("params_sd", acc_sd(&pa), 0);
    jnum("train_s", acc_mean(&ts), 0);  jnum("us_per_infer", us, 0);
    static const char *k[8] = { "init_layers", "layers", "or_add", "or_drop",
                                "and_add", "and_drop", "layer_add", "layer_drop" };
    for (int i = 0; i < 8; i++) jnum(k[i], acc_mean(&cnt[i]), i == 7);
    printf("}\n");
    fflush(stdout);
}

/* ── tasks ─────────────────────────────────────────────────────────── */

static int want(const char *sel, const char *name)
{
    return !strcmp(sel, "all") || !strcmp(sel, name);
}

/* XOR has no hold-out (4 points; leaving one out asks the model to
   predict the opposite of what the other three imply). Fit only. */
static void bench_xor(const char *msel)
{
    static double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    for (size_t i = 0; i < N_MODELS; i++)
        if (want(msel, MODELS[i].name))
            run_cell(&MODELS[i], "xor", X, Y, 4, NULL, NULL, 0, 2, 1, 1, 2000, 0.08);
}

static void bench_real(const char *msel, const char *task, const char *file,
                       int (*loader)(const char *, Dataset *),
                       size_t epochs, double lr)
{
    char *path = dataset_find(file);
    Dataset ds;
    if (!path || loader(path, &ds) != 0 || ds.n == 0) {
        fprintf(stderr, "skip %s: %s not found (set TYPE_NN_DATA or make data)\n",
                task, file);
        free(path);
        return;
    }
    size_t *perm = (size_t *)malloc(ds.n * sizeof(size_t));
    dataset_perm(ds.n, DATASET_SPLIT_SEED, perm);
    size_t ntr = dataset_ntrain(ds.n), nte = ds.n - ntr;
    dataset_standardize_train(&ds, perm, ntr);
    if (!ds.classification) dataset_minmax_train(&ds, perm, ntr);

    double **Xtr = malloc(ntr * sizeof(double *)), **Ytr = malloc(ntr * sizeof(double *));
    double **Xte = malloc((nte ? nte : 1) * sizeof(double *));
    double **Yte = malloc((nte ? nte : 1) * sizeof(double *));
    for (size_t i = 0; i < ntr; i++) { Xtr[i] = ds.X[perm[i]]; Ytr[i] = ds.Y[perm[i]]; }
    for (size_t i = 0; i < nte; i++) { Xte[i] = ds.X[perm[ntr + i]]; Yte[i] = ds.Y[perm[ntr + i]]; }

    for (size_t i = 0; i < N_MODELS; i++)
        if (want(msel, MODELS[i].name))
            run_cell(&MODELS[i], task, Xtr, Ytr, ntr, Xte, Yte, nte,
                     ds.in, ds.out, ds.classification, epochs, lr);

    free(Xtr); free(Ytr); free(Xte); free(Yte); free(perm);
    dataset_free(&ds);
    free(path);
}

int main(int argc, char **argv)
{
    const char *task = argc > 1 ? argv[1] : "all";
    const char *msel = argc > 2 ? argv[2] : "all";
    if (want(task, "xor")) bench_xor(msel);
    if (want(task, "iris"))
        bench_real(msel, "iris", "iris.data", dataset_load_iris, 250, 0.05);
    if (want(task, "wine"))
        bench_real(msel, "wine", "wine.data", dataset_load_wine, 200, 0.03);
    if (want(task, "wdbc"))
        bench_real(msel, "wdbc", "wdbc.data", dataset_load_wdbc, 80, 0.02);
    if (want(task, "diabetes"))
        bench_real(msel, "diabetes", "diabetes.tab.txt", dataset_load_diabetes, 150, 0.02);
    if (want(task, "ionosphere"))
        bench_real(msel, "ionosphere", "ionosphere.data", dataset_load_ionosphere, 120, 0.02);
    return 0;
}
