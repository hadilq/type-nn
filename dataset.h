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

#endif
