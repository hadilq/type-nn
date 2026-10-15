/*
 * bench.c — one harness for every board model.
 *
 * Every model goes through the same code path: same split, same
 * train-only standardisation, same per-epoch shuffle, same mean-MSE
 * gradient (y − t)/m, same epochs and lr, same metrics, same timer.
 * The only difference is the vtable.
 *
 * Each (task, model) cell is trained on TNN_SPLITS 70/30 splits
 * (default 1: the fixed split, seed 34972) from TNN_SEEDS initialisation
 * seeds each (default 5). The JSON line reports the mean and sd over
 * every (split, seed) run, and the per-split means, so bench.sh can pair
 * the models split by split.
 *
 *   ./bench [task|all] [model|all]
 *   models: type-nn type-nn-overfit c-mlp graph-type-nn c-mlp-matched
 *
 * c-mlp-matched is the widest c-mlp whose parameter count does not exceed
 * graph-type-nn's mean on the same task; it is run right after the graph
 * (it needs the graph's count, never its hold-out).
 */
#define _POSIX_C_SOURCE 200809L
#include "type_nn.h"
#include "type_nn_graph.h"
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

#define N_COUNTERS 12

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
    /* init_layers, layers, or+, or-, and+, and-, layer+, layer-,
       init_edges, edges, edge+, edge- */
    void          (*counters)(const void *m, double *c);
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
    /* a chain of D layers has D edges, x -> L0 included */
    c[8] = (double)m->init_depth; c[9] = (double)m->depth;
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
    c[8] = (double)m->init_depth; c[9] = (double)m->depth;
}
static void  to_destroy(void *m) { tnno_free(m); }

static size_t matched_hidden = 0;   /* set after the graph cell of a task */

static void *cm_create(size_t in, size_t out, unsigned s) { return cmlp_create(in, out, s); }
static void *cmm_create(size_t in, size_t out, unsigned s)
{
    return cmlp_create_hidden(in, out, matched_hidden ? matched_hidden : 1, s);
}
static void  cm_begin(void *m, size_t n, size_t e, double lr) { cmlp_begin(m, n, e, lr); }
static void  cm_training(void *m, int on) { (void)m; (void)on; }
static const double *cm_forward(void *m, const double *x) { return cmlp_forward(m, x); }
static void  cm_backward(void *m, const double *dy) { cmlp_backward(m, dy); }
static void  cm_noop(void *m) { (void)m; }
static size_t cm_params(const void *m) { return cmlp_params(m); }
static void  cm_counters(const void *m, double *c)
{
    (void)m;
    memset(c, 0, N_COUNTERS * sizeof(double));
    c[0] = c[1] = 2.0;
    c[8] = c[9] = 2.0;
}
static void  cm_destroy(void *m) { cmlp_free(m); }

static void *gg_create(size_t in, size_t out, unsigned s) { return tnng_create(in, out, s); }
static void  gg_begin(void *m, size_t n, size_t e, double lr) { tnng_begin(m, n, e, lr); }
static void  gg_training(void *m, int on) { ((TypeNNGraph *)m)->training = on; }
static const double *gg_forward(void *m, const double *x) { return tnng_forward(m, x); }
static void  gg_backward(void *m, const double *dy) { tnng_backward(m, dy); }
static void  gg_epoch(void *m) { tnng_epoch_end(m); }
static void  gg_end(void *m) { tnng_end(m); }
static size_t gg_params(const void *m) { return tnng_params(m); }
static void  gg_counters(const void *m, double *c)
{
    const TypeNNGraph *g = m;
    c[0] = (double)g->birth_vertices; c[1] = (double)g->n_nodes;
    c[2] = g->or_add;   c[3] = g->or_drop;
    c[4] = g->and_add;  c[5] = g->and_drop;
    c[6] = g->node_add; c[7] = g->node_drop;
    c[8] = (double)g->birth_edges; c[9] = (double)g->n_edges;
    c[10] = g->edge_add; c[11] = g->edge_drop;
}
static void  gg_destroy(void *m) { tnng_free(m); }

static const Model MODELS[] = {
    { "type-nn", tn_create, tn_begin, tn_training, tn_forward, tn_backward,
      tn_epoch, tn_end, tn_params, tn_counters, tn_destroy },
    { "type-nn-overfit", to_create, to_begin, to_training, to_forward, to_backward,
      to_epoch, to_end, to_params, to_counters, to_destroy },
    { "c-mlp", cm_create, cm_begin, cm_training, cm_forward, cm_backward,
      cm_noop, cm_noop, cm_params, cm_counters, cm_destroy },
    { "graph-type-nn", gg_create, gg_begin, gg_training, gg_forward, gg_backward,
      gg_epoch, gg_end, gg_params, gg_counters, gg_destroy },
    { "c-mlp-matched", cmm_create, cm_begin, cm_training, cm_forward, cm_backward,
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

/* ── statistics ────────────────────────────────────────────────────── */

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

static size_t env_count(const char *name, long dflt, long max)
{
    const char *e = getenv(name);
    long v = e ? atol(e) : dflt;
    if (v < 1) v = 1;
    if (v > max) v = max;
    return (size_t)v;
}
static size_t n_seeds(void)  { return env_count("TNN_SEEDS", 5, 1000); }
#define MAX_SPLITS 64
static size_t n_splits(void) { return env_count("TNN_SPLITS", 1, MAX_SPLITS); }

/* split 0 is the historical fixed split */
static unsigned split_seed(size_t s) { return DATASET_SPLIT_SEED + 104729u * (unsigned)s; }

static void jnum(const char *key, double v, int last)
{
    if (isfinite(v)) printf("\"%s\":%.8g%s", key, v, last ? "" : ",");
    else printf("\"%s\":null%s", key, last ? "" : ",");
}

typedef struct {
    Acc hm, ha, tm, ta, pa, ts;
    Acc cnt[N_COUNTERS];
    double split_hold[MAX_SPLITS], split_params[MAX_SPLITS];
    size_t n_split;
    double us;
} Cell;

/* One split of one cell: train from every seed. */
static void run_split(const Model *md, Cell *cell, const char *task,
                      double **Xtr, double **Ytr, size_t ntr,
                      double **Xte, double **Yte, size_t nte,
                      size_t in, size_t out, int classify,
                      size_t epochs, double lr, int last_split)
{
    Acc sh = {0}, sp = {0};
    size_t S = n_seeds();
    for (size_t s = 0; s < S; s++) {
        unsigned seed = DATASET_SPLIT_SEED + 7919u * (unsigned)(s + 1);
        void *m = md->create(in, out, seed);
        double t0 = bench_wall_s();
        train(md, m, Xtr, Ytr, ntr, out, epochs, lr);
        acc_add(&cell->ts, bench_wall_s() - t0);
        acc_add(&cell->tm, eval_mse(md, m, Xtr, Ytr, ntr, out));
        if (classify) acc_add(&cell->ta, eval_acc(md, m, Xtr, Ytr, ntr, out));
        double hold = NAN;
        if (nte) {
            hold = eval_mse(md, m, Xte, Yte, nte, out);
            acc_add(&cell->hm, hold);
            acc_add(&sh, hold);
            if (classify) acc_add(&cell->ha, eval_acc(md, m, Xte, Yte, nte, out));
        }
        acc_add(&cell->pa, (double)md->params(m));
        acc_add(&sp, (double)md->params(m));
        double c[N_COUNTERS];
        memset(c, 0, sizeof(c));
        md->counters(m, c);
        for (int i = 0; i < N_COUNTERS; i++) acc_add(&cell->cnt[i], c[i]);
        if (getenv("TNN_VERBOSE"))
            fprintf(stderr, "  %s %s split %zu seed %zu: hold_mse %.5f train_mse %.2e params %zu"
                    " layers %g edges %g\n", task, md->name, cell->n_split, s, hold,
                    eval_mse(md, m, Xtr, Ytr, ntr, out), md->params(m), c[1], c[9]);
        if (last_split && s + 1 == S) cell->us = time_infer(md, m, Xtr, ntr);
        md->destroy(m);
    }
    cell->split_hold[cell->n_split] = acc_mean(&sh);
    cell->split_params[cell->n_split] = acc_mean(&sp);
    cell->n_split++;
}

static void print_cell(const Model *md, const Cell *c, const char *task,
                       size_t epochs, double lr)
{
    printf("{\"impl\":\"%s\",\"task\":\"%s\",\"seeds\":%zu,\"splits\":%zu,\"epochs\":%zu,\"lr\":%g,",
           md->name, task, n_seeds(), c->n_split, epochs, lr);
    jnum("hold_mse", acc_mean(&c->hm), 0); jnum("hold_mse_sd", acc_sd(&c->hm), 0);
    jnum("hold_acc", acc_mean(&c->ha), 0); jnum("hold_acc_sd", acc_sd(&c->ha), 0);
    jnum("mse", acc_mean(&c->tm), 0);      jnum("acc", acc_mean(&c->ta), 0);
    jnum("params", acc_mean(&c->pa), 0);   jnum("params_sd", acc_sd(&c->pa), 0);
    jnum("train_s", acc_mean(&c->ts), 0);  jnum("us_per_infer", c->us, 0);
    static const char *k[N_COUNTERS] = {
        "init_layers", "layers", "or_add", "or_drop", "and_add", "and_drop",
        "layer_add", "layer_drop", "init_edges", "edges", "edge_add", "edge_drop" };
    for (int i = 0; i < N_COUNTERS; i++) jnum(k[i], acc_mean(&c->cnt[i]), 0);
    printf("\"hold_mse_splits\":[");
    for (size_t s = 0; s < c->n_split; s++)
        printf(isfinite(c->split_hold[s]) ? "%s%.8g" : "%snull", s ? "," : "", c->split_hold[s]);
    printf("],\"params_splits\":[");
    for (size_t s = 0; s < c->n_split; s++) printf("%s%.8g", s ? "," : "", c->split_params[s]);
    printf("]}\n");
    fflush(stdout);
}

/* ── tasks ─────────────────────────────────────────────────────────── */

static int want(const char *sel, const char *name)
{
    return !strcmp(sel, "all") || !strcmp(sel, name);
}

static void set_matched(const Cell *graph_cell, size_t in, size_t out)
{
    double P = acc_mean(&graph_cell->pa);
    /* params(H) = H (in + 1) + out (H + 1) <= P */
    double H = floor((P - (double)out) / (double)(in + 1 + out));
    matched_hidden = H < 1.0 ? 1 : (size_t)H;
}

/* XOR has no hold-out (4 points; leaving one out asks the model to
   predict the opposite of what the other three imply). Fit only. */
static void bench_xor(const char *msel)
{
    static double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    static double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }
    for (size_t i = 0; i < N_MODELS; i++) {
        if (!want(msel, MODELS[i].name)) continue;
        if (!strcmp(MODELS[i].name, "c-mlp-matched") && !matched_hidden) continue;
        Cell c;
        memset(&c, 0, sizeof c);
        run_split(&MODELS[i], &c, "xor", X, Y, 4, NULL, NULL, 0, 2, 1, 1, 2000, 0.08, 1);
        print_cell(&MODELS[i], &c, "xor", 2000, 0.08);
        if (!strcmp(MODELS[i].name, "graph-type-nn")) set_matched(&c, 2, 1);
    }
    matched_hidden = 0;
}

typedef struct {
    Dataset ds;
    double **Xtr, **Ytr, **Xte, **Yte;
    size_t ntr, nte;
} Split;

static int load_split(const char *file, int (*loader)(const char *, Dataset *),
                      unsigned sseed, Split *sp)
{
    memset(sp, 0, sizeof *sp);
    if (loader) {
        char *path = dataset_find(file);
        int bad = !path || loader(path, &sp->ds) != 0 || sp->ds.n == 0;
        free(path);
        if (bad) return -1;
    } else if (dataset_make_synth(&sp->ds, 4000, 42) != 0) {
        return -1;
    }
    Dataset *ds = &sp->ds;
    size_t *perm = (size_t *)malloc(ds->n * sizeof(size_t));
    dataset_perm(ds->n, sseed, perm);
    sp->ntr = dataset_ntrain(ds->n);
    sp->nte = ds->n - sp->ntr;
    dataset_standardize_train(ds, perm, sp->ntr);
    if (!ds->classification) dataset_minmax_train(ds, perm, sp->ntr);
    sp->Xtr = malloc(sp->ntr * sizeof(double *)); sp->Ytr = malloc(sp->ntr * sizeof(double *));
    sp->Xte = malloc((sp->nte ? sp->nte : 1) * sizeof(double *));
    sp->Yte = malloc((sp->nte ? sp->nte : 1) * sizeof(double *));
    for (size_t i = 0; i < sp->ntr; i++) { sp->Xtr[i] = ds->X[perm[i]]; sp->Ytr[i] = ds->Y[perm[i]]; }
    for (size_t i = 0; i < sp->nte; i++) { sp->Xte[i] = ds->X[perm[sp->ntr + i]]; sp->Yte[i] = ds->Y[perm[sp->ntr + i]]; }
    free(perm);
    return 0;
}

static void free_split(Split *sp)
{
    free(sp->Xtr); free(sp->Ytr); free(sp->Xte); free(sp->Yte);
    dataset_free(&sp->ds);
}

static void bench_task(const char *msel, const char *task, const char *file,
                       int (*loader)(const char *, Dataset *),
                       size_t epochs, double lr)
{
    Split probe;
    if (load_split(file, loader, split_seed(0), &probe) != 0) {
        fprintf(stderr, "skip %s: %s not found (set TYPE_NN_DATA or make data)\n", task, file);
        return;
    }
    size_t in = probe.ds.in, out = probe.ds.out;
    int classify = probe.ds.classification;
    free_split(&probe);
    size_t P = n_splits();
    for (size_t i = 0; i < N_MODELS; i++) {
        if (!want(msel, MODELS[i].name)) continue;
        if (!strcmp(MODELS[i].name, "c-mlp-matched") && !matched_hidden) {
            fprintf(stderr, "skip %s c-mlp-matched: needs graph-type-nn in the same run\n", task);
            continue;
        }
        Cell c;
        memset(&c, 0, sizeof c);
        for (size_t s = 0; s < P; s++) {
            Split sp;
            if (load_split(file, loader, split_seed(s), &sp) != 0) break;
            run_split(&MODELS[i], &c, task, sp.Xtr, sp.Ytr, sp.ntr, sp.Xte, sp.Yte, sp.nte,
                      in, out, classify, epochs, lr, s + 1 == P);
            free_split(&sp);
        }
        print_cell(&MODELS[i], &c, task, epochs, lr);
        if (!strcmp(MODELS[i].name, "graph-type-nn")) set_matched(&c, in, out);
    }
    matched_hidden = 0;
}

int main(int argc, char **argv)
{
    const char *task = argc > 1 ? argv[1] : "all";
    const char *msel = argc > 2 ? argv[2] : "all";
    if (!strcmp(msel, "c-mlp-matched")) {
        fprintf(stderr, "c-mlp-matched needs graph-type-nn in the same run; use 'all'\n");
        return 1;
    }
    if (want(task, "xor")) bench_xor(msel);
    if (want(task, "iris"))
        bench_task(msel, "iris", "iris.data", dataset_load_iris, 250, 0.05);
    if (want(task, "wine"))
        bench_task(msel, "wine", "wine.data", dataset_load_wine, 200, 0.03);
    if (want(task, "wdbc"))
        bench_task(msel, "wdbc", "wdbc.data", dataset_load_wdbc, 80, 0.02);
    if (want(task, "diabetes"))
        bench_task(msel, "diabetes", "diabetes.tab.txt", dataset_load_diabetes, 150, 0.02);
    if (want(task, "ionosphere"))
        bench_task(msel, "ionosphere", "ionosphere.data", dataset_load_ionosphere, 120, 0.02);
    if (want(task, "spambase"))
        bench_task(msel, "spambase", "spambase.data", dataset_load_spambase, 40, 0.01);
    if (want(task, "synth"))       /* generated: 4000 samples, sparse interactions */
        bench_task(msel, "synth", NULL, NULL, 60, 0.02);
    if (want(task, "mnist"))
        bench_task(msel, "mnist", "mnist_subset.bin", dataset_load_bin, 15, 0.01);
    if (want(task, "mnist_large"))
        bench_task(msel, "mnist_large", "mnist_large.bin", dataset_load_bin, 8, 0.008);
    if (want(task, "video"))
        bench_task(msel, "video", "video_frames.bin", dataset_load_bin, 20, 0.01);
    if (want(task, "video_large"))
        bench_task(msel, "video_large", "video_large.bin", dataset_load_bin, 10, 0.008);
    return 0;
}
