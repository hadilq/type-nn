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

/* Statistics from the training rows only; applied to every row. */
void dataset_standardize_train(Dataset *ds, const size_t *rows, size_t ntr);
void dataset_minmax_train(Dataset *ds, const size_t *rows, size_t ntr);
void dataset_free(Dataset *ds);

/* Shared 70/30 hold-out: xorshift32 Fisher–Yates, seed 34972.
   Independent of libc rand() and of model initialisation. */
#define DATASET_SPLIT_SEED 34972u
#define DATASET_TRAIN_NUM  7
#define DATASET_TRAIN_DEN  10

unsigned dataset_xorshift32(unsigned *state);
void     dataset_perm(size_t n, unsigned seed, size_t *perm);
size_t   dataset_ntrain(size_t n);

#endif
