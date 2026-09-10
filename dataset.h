#ifndef TYPE_NN_DATASET_H
#define TYPE_NN_DATASET_H

#include <stddef.h>

typedef struct {
    char     name[32];
    size_t   n;
    size_t   in;
    size_t   out;
    int      classification; /* 1 = one-hot Y, 0 = regression */
    double **X;
    double **Y;
} Dataset;

/* Search TYPE_NN_DATA, $1, ./data, /tmp/type-nn-data for `filename`. */
char *dataset_find(const char *filename);

int  dataset_load_iris(const char *path, Dataset *ds);
int  dataset_load_wine(const char *path, Dataset *ds);
int  dataset_load_wdbc(const char *path, Dataset *ds);
int  dataset_load_diabetes(const char *path, Dataset *ds);
int  dataset_load_ionosphere(const char *path, Dataset *ds);

void dataset_standardize_inputs(Dataset *ds);
void dataset_minmax_outputs(Dataset *ds); /* map Y to ~[0,1] for regression */
void dataset_free(Dataset *ds);

/* Shared 70/30 hold-out. Same algorithm in bench_torch.py.
   Seed 34972, xorshift32 Fisher–Yates. Independent of libc rand()
   and of how many rand() calls weight-init consumed. */
#define DATASET_SPLIT_SEED 34972u
#define DATASET_TRAIN_NUM  7
#define DATASET_TRAIN_DEN  10

unsigned dataset_xorshift32(unsigned *state);
void     dataset_perm(size_t n, unsigned seed, size_t *perm);
size_t   dataset_ntrain(size_t n);

#endif
