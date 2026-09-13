/*
 * bench_type_nn.c – wall-clock + RSS measurements for the C AND-OR net.
 *
 * Prints one JSON object per task so bench.sh can merge with the PyTorch side.
 */
#define _POSIX_C_SOURCE 200809L
#include "type_nn.h"
#include "dataset.h"
#include "bench_time.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

static double wall_s(void)
{
    return bench_wall_s();
}

/* Time network_predict. Sink + min wall time so us/infer cannot print 0. */
static double time_net_infer(Network *net, double **X, size_t n, size_t in,
                             size_t out, size_t min_reps, size_t *infer_n)
{
    double *pred = (double *)calloc(out ? out : 1, sizeof(double));
    volatile double sink = 0.0;
    size_t done = 0;
    size_t batch = min_reps ? min_reps : 1000;
    double t0 = wall_s();
    do {
        for (size_t r = 0; r < batch; r++) {
            network_predict(net, X[r % n], in, pred, out);
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
    return acc / (double)(n * (out ? out : 1));
}


static void net_snap(const Network *net, size_t *depth,
                     size_t *n_and, size_t *n_or)
{
    size_t d = 0;
    for (Layer *L = net->head; L && d < 8; L = L->next, d++) {
        size_t a = 0, o = 0;
        for (AndNode *an = L->and_row; an; an = an->right) {
            a++;
            for (OrNode *orn = an->or_row; orn; orn = orn->right) o++;
        }
        n_and[d] = a;
        n_or[d] = o;
    }
    *depth = d;
}

static double net_dyn_scale(size_t db, const size_t *ab, const size_t *ob,
                            size_t da, const size_t *aa, const size_t *oa)
{
    size_t n = db > da ? db : da;
    double m = 0.0;
    for (size_t i = 0; i < n && i < 8; i++) {
        long d_and = (long)(i < da ? aa[i] : 0) - (long)(i < db ? ab[i] : 0);
        long d_or  = (long)(i < da ? oa[i] : 0) - (long)(i < db ? ob[i] : 0);
        if (d_and < 0) d_and = -d_and;
        if (d_or < 0) d_or = -d_or;
        m += (double)d_or + (double)d_and + (double)d_or * (double)d_and;
    }
    return m;
}

static const char *g_impl = "type-nn";
static const char *g_mode = NULL;

static void apply_mode(Network *net, size_t n_samples, size_t epochs)
{
    if (!g_mode) return;
    network_set_andpol(net, g_mode);
    if (n_samples && epochs)
        network_set_orcool_span(net, (unsigned)(n_samples * epochs));
}

static void emit(const char *task, double train_s, double infer_s,
                 size_t infer_n, long rss, long hwm,
                 size_t params, size_t nbytes, double final_mse, size_t depth,
                 size_t n_samples, double acc, double dyn_scale, int dyn_depth, long dyn_params,
                 long layer_add, long layer_drop,
                 long or_add, long or_drop, long and_add, long and_drop,
                 double hold_mse, double hold_acc)
{
    double us_per = infer_n ? (infer_s * 1e6 / (double)infer_n) : 0.0;
    printf(
        "{\"impl\":\"%s\",\"task\":\"%s\",\"train_s\":%.6f,"
        "\"infer_s\":%.6f,\"infer_n\":%zu,\"us_per_infer\":%.4f,"
        "\"rss_kb\":%ld,\"hwm_kb\":%ld,\"params\":%zu,\"nbytes\":%zu,"
        "\"mse\":%.8f,\"depth\":%zu,\"n\":%zu,\"acc\":%.6f,"
        "\"dyn_scale\":%.4f,\"dyn_depth\":%d,\"dyn_params\":%ld,"
        "\"layer_add\":%ld,\"layer_drop\":%ld,"
        "\"or_add\":%ld,\"or_drop\":%ld,\"and_add\":%ld,\"and_drop\":%ld,"
        "\"hold_mse\":%.8f,\"hold_acc\":%.6f}\n",
        g_impl, task, train_s, infer_s, infer_n, us_per,
        rss, hwm, params, nbytes, final_mse, depth, n_samples, acc,
        dyn_scale, dyn_depth, dyn_params,
        layer_add, layer_drop, or_add, or_drop, and_add, and_drop,
        hold_mse, hold_acc);
}

static void bench_xor(void)
{
    srand(34972);
    Network *net = network_create(2, 1);
    network_set_dynamic(net, 1);
    network_set_layer_probe(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, 0.08);
    apply_mode(net, 4, 4000);
    network_init_weights(net);

    double Xd[4][2] = {{0,0},{0,1},{1,0},{1,1}};
    double Yd[4][1] = {{0},{1},{1},{0}};
    double *X[4], *Y[4];
    for (int i = 0; i < 4; i++) { X[i] = Xd[i]; Y[i] = Yd[i]; }

    size_t db = 0, da = 0, ab[8] = {0}, ob[8] = {0}, aa[8] = {0}, oa[8] = {0};
    net_snap(net, &db, ab, ob);
    size_t p0 = network_param_count(net);
    double t0 = wall_s();
    network_train(net, X, Y, 4, 400);
    double train_s = wall_s() - t0;
    net_snap(net, &da, aa, oa);
    double dyn_scale = net_dyn_scale(db, ab, ob, da, aa, oa);
    int dyn_depth = (int)da - (int)db;
    long dyn_params = (long)network_param_count(net) - (long)p0;

    size_t infer_n = 0;
    double infer_s = time_net_infer(net, X, 4, 2, 1, 20000, &infer_n);

    emit("xor", train_s, infer_s, infer_n, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, 4, 2, 1), network_depth(net), 4, -1.0, dyn_scale, dyn_depth, dyn_params,
         (long)net->layer_add, (long)net->layer_drop,
         (long)net->or_add, (long)net->or_drop,
         (long)net->and_add, (long)net->and_drop,
         mse(net, X, Y, 4, 2, 1), -1.0);
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

    size_t db = 0, da = 0, ab[8] = {0}, ob[8] = {0}, aa[8] = {0}, oa[8] = {0};
    net_snap(net, &db, ab, ob);
    size_t p0 = network_param_count(net);
    double t0 = wall_s();
    network_train(net, X, Y, N, 200);
    double train_s = wall_s() - t0;
    net_snap(net, &da, aa, oa);
    double dyn_scale = net_dyn_scale(db, ab, ob, da, aa, oa);
    int dyn_depth = (int)da - (int)db;
    long dyn_params = (long)network_param_count(net) - (long)p0;

    size_t infer_n = 0;
    double infer_s = time_net_infer(net, X, N, 2, 1, 5000, &infer_n);

    emit("quadratic", train_s, infer_s, infer_n, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, N, 2, 1), network_depth(net), N, -1.0, dyn_scale, dyn_depth, dyn_params,
         (long)net->layer_add, (long)net->layer_drop,
         (long)net->or_add, (long)net->or_drop,
         (long)net->and_add, (long)net->and_drop,
         mse(net, X, Y, N, 2, 1), -1.0);
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

    size_t db = 0, da = 0, ab[8] = {0}, ob[8] = {0}, aa[8] = {0}, oa[8] = {0};
    net_snap(net, &db, ab, ob);
    size_t p0 = network_param_count(net);
    double t0 = wall_s();
    network_train(net, X, Y, N, 30);
    double train_s = wall_s() - t0;
    net_snap(net, &da, aa, oa);
    double dyn_scale = net_dyn_scale(db, ab, ob, da, aa, oa);
    int dyn_depth = (int)da - (int)db;
    long dyn_params = (long)network_param_count(net) - (long)p0;

    size_t infer_n = 0;
    double infer_s = time_net_infer(net, X, N, IN, OUT, 1000, &infer_n);

    emit("mlp32x16x8", train_s, infer_s, infer_n, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         mse(net, X, Y, N, IN, OUT), network_depth(net), N, -1.0, dyn_scale, dyn_depth, dyn_params,
         (long)net->layer_add, (long)net->layer_drop,
         (long)net->or_add, (long)net->or_drop,
         (long)net->and_add, (long)net->and_drop,
         mse(net, X, Y, N, IN, OUT), -1.0);
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

    size_t *perm = (size_t *)malloc(ds.n * sizeof(size_t));
    dataset_perm(ds.n, DATASET_SPLIT_SEED, perm);
    size_t ntr = dataset_ntrain(ds.n);
    size_t nte = ds.n - ntr;
    double **Xtr = (double **)malloc(ntr * sizeof(double *));
    double **Ytr = (double **)malloc(ntr * sizeof(double *));
    for (size_t i = 0; i < ntr; i++) {
        Xtr[i] = ds.X[perm[i]];
        Ytr[i] = ds.Y[perm[i]];
    }

    srand(34972);
    Network *net = network_create(ds.in, ds.out);
    network_set_dynamic(net, 1);
    network_set_layer_probe(net, 0);
    network_set_verbose(net, 0);
    network_set_learning_rate(net, lr);
    apply_mode(net, ntr, epochs);
    network_init_weights(net);

    size_t db = 0, da = 0, ab[8] = {0}, ob[8] = {0}, aa[8] = {0}, oa[8] = {0};
    net_snap(net, &db, ab, ob);
    size_t p0 = network_param_count(net);
    double t0 = wall_s();
    network_train(net, Xtr, Ytr, ntr, epochs);
    double train_s = wall_s() - t0;
    net_snap(net, &da, aa, oa);
    double dyn_scale = net_dyn_scale(db, ab, ob, da, aa, oa);
    int dyn_depth = (int)da - (int)db;
    long dyn_params = (long)network_param_count(net) - (long)p0;

    size_t infer_n = 0;
    double infer_s = time_net_infer(net, ds.X, ds.n, ds.in, ds.out,
                                   infer_reps, &infer_n);

    double tr_mse = mse(net, Xtr, Ytr, ntr, ds.in, ds.out);
    double te_mse = 0.0, te_acc = -1.0, tr_acc = -1.0;
    if (nte) {
        double **Xte = (double **)malloc(nte * sizeof(double *));
        double **Yte = (double **)malloc(nte * sizeof(double *));
        for (size_t i = 0; i < nte; i++) {
            Xte[i] = ds.X[perm[ntr + i]];
            Yte[i] = ds.Y[perm[ntr + i]];
        }
        te_mse = mse(net, Xte, Yte, nte, ds.in, ds.out);
        if (ds.classification) {
            Dataset tr = ds, te = ds;
            tr.X = Xtr; tr.Y = Ytr; tr.n = ntr;
            te.X = Xte; te.Y = Yte; te.n = nte;
            tr_acc = acc_class(net, &tr);
            te_acc = acc_class(net, &te);
        }
        free(Xte); free(Yte);
    } else {
        te_mse = tr_mse;
        if (ds.classification) tr_acc = te_acc = acc_class(net, &ds);
    }
    emit(task, train_s, infer_s, infer_n, rss_kb(), hwm_kb(),
         network_param_count(net), network_nbytes(net),
         tr_mse, network_depth(net), ds.n, tr_acc, dyn_scale, dyn_depth, dyn_params,
         (long)net->layer_add, (long)net->layer_drop,
         (long)net->or_add, (long)net->or_drop,
         (long)net->and_add, (long)net->and_drop,
         te_mse, te_acc);
    free(Xtr); free(Ytr); free(perm);
    network_free(net);
    dataset_free(&ds);
    free(path);
    return 1;
}

int main(int argc, char **argv)
{
    const char *task = (argc > 1) ? argv[1] : "all";
    if (argc > 2 && argv[2][0]) {
        g_mode = argv[2];
        static char impl[64];
        snprintf(impl, sizeof(impl), "type-nn-%s", g_mode);
        g_impl = impl;
    }
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
    if (!strcmp(task, "ionosphere") || !strcmp(task, "all") || !strcmp(task, "real"))
        bench_real("ionosphere", "ionosphere.data", dataset_load_ionosphere, 120, 0.02, 1000);
    return 0;
}
