#define _POSIX_C_SOURCE 200809L
#include "dataset.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>

static int file_readable(const char *p)
{
    FILE *f = fopen(p, "r");
    if (!f) return 0;
    fclose(f);
    return 1;
}

char *dataset_find(const char *filename)
{
    const char *dirs[6];
    int n = 0;
    const char *env = getenv("TYPE_NN_DATA");
    if (env && env[0]) dirs[n++] = env;
    dirs[n++] = "data";
    dirs[n++] = "./data";
    dirs[n++] = "/tmp/type-nn-data";
    dirs[n++] = "/usr/share/type-nn";

    for (int i = 0; i < n; i++) {
        char buf[512];
        snprintf(buf, sizeof(buf), "%s/%s", dirs[i], filename);
        if (file_readable(buf)) return strdup(buf);
        /* also accept the env dir *being* the file when filename matches */
        if (file_readable(dirs[i]) && strstr(dirs[i], filename))
            return strdup(dirs[i]);
    }
    return NULL;
}

static double **mat_new(size_t rows, size_t cols)
{
    double **m = (double **)calloc(rows, sizeof(double *));
    for (size_t i = 0; i < rows; i++)
        m[i] = (double *)calloc(cols, sizeof(double));
    return m;
}

/* Loaders allocate `cap` rows before they know n; free the unused tail. */
static void trim_rows(Dataset *ds, size_t cap, size_t n)
{
    for (size_t i = n; i < cap; i++) { free(ds->X[i]); free(ds->Y[i]); }
}

void dataset_free(Dataset *ds)
{
    if (!ds) return;
    if (ds->X) {
        for (size_t i = 0; i < ds->n; i++) free(ds->X[i]);
        free(ds->X);
    }
    if (ds->Y) {
        for (size_t i = 0; i < ds->n; i++) free(ds->Y[i]);
        free(ds->Y);
    }
    memset(ds, 0, sizeof(*ds));
}

static void one_hot(double *row, size_t k, size_t out)
{
    for (size_t i = 0; i < out; i++) row[i] = (i == k) ? 1.0 : 0.0;
}

/* Standardize X with the mean / std of the training rows only, then
   apply the same affine map to every row. The hold-out rows never
   contribute a statistic. */
void dataset_standardize_train(Dataset *ds, const size_t *rows, size_t ntr)
{
    if (!ds || !ntr || !ds->in) return;
    for (size_t j = 0; j < ds->in; j++) {
        double mean = 0.0, var = 0.0;
        for (size_t i = 0; i < ntr; i++) mean += ds->X[rows[i]][j];
        mean /= (double)ntr;
        for (size_t i = 0; i < ntr; i++) {
            double d = ds->X[rows[i]][j] - mean;
            var += d * d;
        }
        double sd = sqrt(var / (double)(ntr > 1 ? ntr - 1 : 1));
        if (sd < 1e-12) sd = 1.0;
        for (size_t i = 0; i < ds->n; i++)
            ds->X[i][j] = (ds->X[i][j] - mean) / sd;
    }
}

/* Regression targets: min-max from the training rows only. */
void dataset_minmax_train(Dataset *ds, const size_t *rows, size_t ntr)
{
    if (!ds || !ntr || !ds->out) return;
    for (size_t j = 0; j < ds->out; j++) {
        double lo = ds->Y[rows[0]][j], hi = lo;
        for (size_t i = 1; i < ntr; i++) {
            double v = ds->Y[rows[i]][j];
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        double span = hi - lo;
        if (span < 1e-12) span = 1.0;
        for (size_t i = 0; i < ds->n; i++)
            ds->Y[i][j] = (ds->Y[i][j] - lo) / span;
    }
}

/* ---------- Iris: 4 features, class name ---------- */

int dataset_load_iris(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "iris");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[256];
    size_t cap = 160, n = 0;
    ds->in = 4;
    ds->out = 3;
    ds->classification = 1;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        double a, b, c, d;
        char lab[64];
        if (sscanf(line, "%lf,%lf,%lf,%lf,%63s", &a, &b, &c, &d, lab) != 5)
            continue;
        if (n >= cap) break;
        ds->X[n][0] = a; ds->X[n][1] = b; ds->X[n][2] = c; ds->X[n][3] = d;
        size_t k = 0;
        if (strstr(lab, "versicolor")) k = 1;
        else if (strstr(lab, "virginica")) k = 2;
        else k = 0; /* setosa */
        one_hot(ds->Y[n], k, 3);
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}

/* ---------- Wine: class, then 13 features ---------- */

int dataset_load_wine(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "wine");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    size_t cap = 200, n = 0;
    ds->in = 13;
    ds->out = 3;
    ds->classification = 1;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        int cls = 0;
        double v[13];
        if (sscanf(line,
                   "%d,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
                   &cls, &v[0], &v[1], &v[2], &v[3], &v[4], &v[5], &v[6],
                   &v[7], &v[8], &v[9], &v[10], &v[11], &v[12]) != 14)
            continue;
        if (n >= cap) break;
        for (int j = 0; j < 13; j++) ds->X[n][j] = v[j];
        int k = cls - 1;
        if (k < 0) k = 0;
        if (k > 2) k = 2;
        one_hot(ds->Y[n], (size_t)k, 3);
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}

/* ---------- WDBC: id, M/B, 30 features ---------- */

int dataset_load_wdbc(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "wdbc");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[2048];
    size_t cap = 600, n = 0;
    ds->in = 30;
    ds->out = 1;
    ds->classification = 1;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        char *p = line;
        /* skip id */
        while (*p && *p != ',') p++;
        if (*p != ',') continue;
        p++;
        char lab = *p;
        while (*p && *p != ',') p++;
        if (*p != ',') continue;
        p++;
        if (n >= cap) break;
        int ok = 1;
        for (int j = 0; j < 30; j++) {
            char *end = NULL;
            ds->X[n][j] = strtod(p, &end);
            if (end == p) { ok = 0; break; }
            p = end;
            if (*p == ',') p++;
        }
        if (!ok) continue;
        ds->Y[n][0] = (lab == 'M' || lab == 'm') ? 1.0 : 0.0;
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}

/* ---------- Diabetes: TSV with header, 10 features + Y ---------- */

int dataset_load_diabetes(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "diabetes");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[512];
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; } /* header */
    size_t cap = 500, n = 0;
    ds->in = 10;
    ds->out = 1;
    ds->classification = 0;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        double v[11];
        if (sscanf(line, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf",
                   &v[0], &v[1], &v[2], &v[3], &v[4], &v[5],
                   &v[6], &v[7], &v[8], &v[9], &v[10]) != 11)
            continue;
        if (n >= cap) break;
        for (int j = 0; j < 10; j++) ds->X[n][j] = v[j];
        ds->Y[n][0] = v[10];
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}


/* ---------- Ionosphere: 34 radar features + g/b ---------- */

int dataset_load_ionosphere(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "ionosphere");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    size_t cap = 400, n = 0;
    ds->in = 34;
    ds->out = 1;
    ds->classification = 1;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        char *p = line;
        if (n >= cap) break;
        int ok = 1;
        for (int j = 0; j < 34; j++) {
            char *end = NULL;
            ds->X[n][j] = strtod(p, &end);
            if (end == p) { ok = 0; break; }
            p = end;
            if (*p == ',') p++;
        }
        if (!ok) continue;
        while (*p == ' ' || *p == ',') p++;
        ds->Y[n][0] = (p[0] == 'g' || p[0] == 'G') ? 1.0 : 0.0;
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}

/* ---------- Spambase: 57 features, binary label (larger) ---------- */

int dataset_load_spambase(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "spambase");
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    size_t cap = 5000, n = 0;
    ds->in = 57;
    ds->out = 1;
    ds->classification = 1;
    ds->X = mat_new(cap, ds->in);
    ds->Y = mat_new(cap, ds->out);
    while (fgets(line, sizeof(line), f)) {
        if (line[0] == '\n' || line[0] == '\0') continue;
        if (n >= cap) break;
        char *p = line;
        int ok = 1;
        for (int j = 0; j < 57; j++) {
            char *end = NULL;
            ds->X[n][j] = strtod(p, &end);
            if (end == p) { ok = 0; break; }
            p = end;
            if (*p == ',') p++;
        }
        if (!ok) continue;
        while (*p == ' ' || *p == ',') p++;
        ds->Y[n][0] = (*p == '1') ? 1.0 : 0.0;
        n++;
    }
    fclose(f);
    trim_rows(ds, cap, n);
    ds->n = n;
    return n > 0 ? 0 : -1;
}

/* ---------- Generic float32 binary: <n in out> then n rows of (in + out) floats ---------- */

int dataset_load_bin(const char *path, Dataset *ds)
{
    memset(ds, 0, sizeof(*ds));
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    unsigned hdr[3];
    if (fread(hdr, sizeof(unsigned), 3, f) != 3) { fclose(f); return -1; }
    size_t n = hdr[0], nin = hdr[1], nout = hdr[2];
    if (n == 0 || nin == 0 || nout == 0 || n > 100000 || nin > 4096) {
        fclose(f); return -1;
    }
    snprintf(ds->name, sizeof(ds->name), "bin");
    ds->n = n; ds->in = nin; ds->out = nout;
    ds->classification = 1;
    ds->X = mat_new(n, nin);
    ds->Y = mat_new(n, nout);
    for (size_t i = 0; i < n; i++) {
        float *row = malloc((nin + nout) * sizeof(float));
        if (!row || fread(row, sizeof(float), nin + nout, f) != nin + nout) {
            free(row); fclose(f); return -1;
        }
        for (size_t j = 0; j < nin; j++) ds->X[i][j] = (double)row[j];
        for (size_t j = 0; j < nout; j++) ds->Y[i][j] = (double)row[nin + j];
        free(row);
    }
    fclose(f);
    return 0;
}

/* ---------- Synthetic larger set: sparse interactions ---------- */

int dataset_make_synth(Dataset *ds, size_t n, unsigned seed)
{
    memset(ds, 0, sizeof(*ds));
    snprintf(ds->name, sizeof(ds->name), "synth");
    if (n < 100) n = 100;
    ds->n = n;
    ds->in = 16;
    ds->out = 1;
    ds->classification = 1;
    ds->X = mat_new(n, ds->in);
    ds->Y = mat_new(n, ds->out);
    unsigned rng = seed ? seed : 1;
    for (size_t i = 0; i < n; i++) {
        for (int j = 0; j < 16; j++) {
            unsigned r = dataset_xorshift32(&rng);
            ds->X[i][j] = ((r >> 8) / 16777216.0) * 2.0 - 1.0;
        }
        /* sparse interaction: label depends on x0*x1 + x2*x3 + noise */
        double s = ds->X[i][0] * ds->X[i][1] + ds->X[i][2] * ds->X[i][3];
        unsigned r = dataset_xorshift32(&rng);
        double noise = ((r >> 8) / 16777216.0) * 0.4 - 0.2;
        ds->Y[i][0] = (s + noise > 0.0) ? 1.0 : 0.0;
    }
    return 0;
}

/* Portable xorshift32 for the split. */
unsigned dataset_xorshift32(unsigned *state)
{
    unsigned x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

void dataset_perm(size_t n, unsigned seed, size_t *perm)
{
    for (size_t i = 0; i < n; i++) perm[i] = i;
    unsigned s = seed ? seed : DATASET_SPLIT_SEED;
    if (s == 0) s = 1;
    for (size_t i = n; i > 1; i--) {
        unsigned r = dataset_xorshift32(&s);
        size_t j = (size_t)(r % (unsigned)i);
        size_t tmp = perm[i - 1];
        perm[i - 1] = perm[j];
        perm[j] = tmp;
    }
}

size_t dataset_ntrain(size_t n)
{
    size_t ntr = n * DATASET_TRAIN_NUM / DATASET_TRAIN_DEN;
    if (ntr < 1) ntr = n;
    if (ntr > n) ntr = n;
    return ntr;
}
