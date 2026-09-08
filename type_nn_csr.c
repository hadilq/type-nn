#include "type_nn_dense.h"

typedef struct {
    DenseLayer *d;
    int *row_ptr, *col;
    double *val;
    int nnz;
} CsrLayer;

static void csr_pack(CsrLayer *C)
{
    DenseLayer *L = C->d;
    size_t n_or = L->out * L->k;
    C->row_ptr = (int *)realloc(C->row_ptr, (n_or + 1) * sizeof(int));
    C->col = (int *)realloc(C->col, n_or * L->in * sizeof(int));
    C->val = (double *)realloc(C->val, n_or * L->in * sizeof(double));
    int z = 0;
    for (size_t r = 0; r < n_or; r++) {
        C->row_ptr[r] = z;
        for (size_t j = 0; j < L->in; j++) {
            C->col[z] = (int)j;
            C->val[z] = L->W[r * L->in + j];
            z++;
        }
    }
    C->row_ptr[n_or] = z;
    C->nnz = z;
}
static void csr_unpack(CsrLayer *C)
{
    DenseLayer *L = C->d;
    size_t n_or = L->out * L->k;
    for (size_t r = 0; r < n_or; r++)
        for (int p = C->row_ptr[r]; p < C->row_ptr[r + 1]; p++)
            L->W[r * L->in + C->col[p]] = C->val[p];
}

static void *csr_new(size_t in, size_t out, size_t k)
{
    CsrLayer *C = (CsrLayer *)calloc(1, sizeof(CsrLayer));
    C->d = dense_new(in, out, k, 0);
    csr_pack(C);
    return C;
}
static void csr_init(void *p) { CsrLayer *C=p; dense_init(C->d); csr_pack(C); }
static void csr_fwd(void *p, const double *x, double *y)
{
    CsrLayer *C = (CsrLayer *)p;
    DenseLayer *L = C->d;
    size_t n_or = L->out * L->k;
    for (size_t r = 0; r < n_or; r++) {
        double acc = L->b[r];
        for (int t = C->row_ptr[r]; t < C->row_ptr[r + 1]; t++)
            acc += C->val[t] * x[C->col[t]];
        L->or_val[r] = tnn_clamp(acc, -TNN_ORCLIP, TNN_ORCLIP);
    }
    for (size_t i = 0; i < L->out; i++) {
        double prod = 1.0;
        for (size_t t = 0; t < L->k; t++) prod *= L->or_val[i * L->k + t];
        y[i] = tnn_clamp(prod, -TNN_ANDCLIP, TNN_ANDCLIP);
    }
}
static void csr_bwd(void *p, const double *x, const double *dy, double *dx, double lr)
{
    CsrLayer *C = (CsrLayer *)p;
    csr_unpack(C);
    dense_bwd(C->d, x, dy, dx, lr);
    csr_pack(C);
}
static void csr_rin(void *p, size_t in)  { CsrLayer *C=p; dense_resize_in(C->d,in); csr_pack(C); }
static void csr_rout(void *p, size_t o)  { CsrLayer *C=p; dense_resize_out(C->d,o); csr_pack(C); }
static void csr_sk(void *p, size_t k)    { CsrLayer *C=p; dense_set_k(C->d,k); csr_pack(C); }
static void csr_id(void *p)              { CsrLayer *C=p; dense_identity(C->d); csr_pack(C); }
static size_t csr_in(void *p)  { return dense_in(((CsrLayer *)p)->d); }
static size_t csr_out(void *p) { return dense_out(((CsrLayer *)p)->d); }
static size_t csr_k(void *p)   { return dense_k(((CsrLayer *)p)->d); }
static size_t csr_params(void *p) { return dense_params(((CsrLayer *)p)->d); }
static size_t csr_nbytes(void *p)
{
    CsrLayer *C = (CsrLayer *)p;
    DenseLayer *L = C->d;
    return sizeof(CsrLayer) + sizeof(DenseLayer)
        + sizeof(int) * (L->out * L->k + 1)
        + (sizeof(int) + sizeof(double)) * (size_t)C->nnz
        + sizeof(double) * 2 * L->out * L->k;
}
static void csr_free(void *p)
{
    CsrLayer *C = (CsrLayer *)p;
    dense_free(C->d); free(C->row_ptr); free(C->col); free(C->val); free(C);
}

static const TLayerOps OPS = {
    csr_new, csr_init, csr_fwd, csr_bwd,
    csr_rin, csr_rout, csr_sk, csr_id,
    csr_in, csr_out, csr_k, csr_params, csr_nbytes, csr_free
};

AltNet type_nn_csr_open(size_t in, size_t out)
{
    AltNet h;
    tstack_bind(&h, tstack_open("type-nn-csr", &OPS, in, out));
    return h;
}
