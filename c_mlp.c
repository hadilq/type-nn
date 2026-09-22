#include "c_mlp.h"
#include "common.h"

#include <stdlib.h>
#include <string.h>

struct CMlp {
    size_t in, out, hid;
    double *W1, *b1, *W2, *b2;             /* hid×in, hid, out×hid, out */
    double *mW1, *vW1, *mb1, *vb1;
    double *mW2, *vW2, *mb2, *vb2;
    double *x, *h, *u, *y;                 /* forward caches */
    double *gh;                            /* backward scratch */
    double lr;
    long   t;
};

static double *zalloc(size_t n) { return (double *)calloc(n ? n : 1, sizeof(double)); }

CMlp *cmlp_create(size_t in, size_t out, unsigned seed)
{
    CMlp *m = (CMlp *)calloc(1, sizeof(CMlp));
    m->in = in;
    m->out = out;
    m->hid = (in <= 4) ? 8 : 16;
    if (m->hid < out) m->hid = out;
    size_t H = m->hid;
    m->W1 = zalloc(H * in);  m->b1 = zalloc(H);
    m->W2 = zalloc(out * H); m->b2 = zalloc(out);
    m->mW1 = zalloc(H * in); m->vW1 = zalloc(H * in);
    m->mb1 = zalloc(H);      m->vb1 = zalloc(H);
    m->mW2 = zalloc(out * H); m->vW2 = zalloc(out * H);
    m->mb2 = zalloc(out);    m->vb2 = zalloc(out);
    m->x = zalloc(in); m->h = zalloc(H); m->u = zalloc(out); m->y = zalloc(out);
    m->gh = zalloc(H);

    unsigned s = seed;
    double k1 = 1.0 / sqrt((double)(in ? in : 1));
    double k2 = 1.0 / sqrt((double)H);
    for (size_t i = 0; i < H * in; i++) m->W1[i] = k1 * tnn_uniform(&s);
    for (size_t i = 0; i < H; i++)      m->b1[i] = k1 * tnn_uniform(&s);
    for (size_t i = 0; i < out * H; i++) m->W2[i] = k2 * tnn_uniform(&s);
    for (size_t i = 0; i < out; i++)    m->b2[i] = k2 * tnn_uniform(&s);
    return m;
}

void cmlp_free(CMlp *m)
{
    if (!m) return;
    free(m->W1); free(m->b1); free(m->W2); free(m->b2);
    free(m->mW1); free(m->vW1); free(m->mb1); free(m->vb1);
    free(m->mW2); free(m->vW2); free(m->mb2); free(m->vb2);
    free(m->x); free(m->h); free(m->u); free(m->y); free(m->gh);
    free(m);
}

void cmlp_begin(CMlp *m, size_t n_train, size_t epochs, double lr)
{
    (void)n_train; (void)epochs;
    m->lr = lr * TNN_LR_SCALE;
    m->t = 0;
}

const double *cmlp_forward(CMlp *m, const double *x)
{
    size_t in = m->in, H = m->hid;
    memcpy(m->x, x, in * sizeof(double));
    for (size_t k = 0; k < H; k++) {
        const double *w = m->W1 + k * in;
        double a = m->b1[k];
        for (size_t j = 0; j < in; j++) a += w[j] * x[j];
        m->h[k] = a > 0.0 ? a : 0.0;
    }
    for (size_t o = 0; o < m->out; o++) {
        const double *w = m->W2 + o * H;
        double a = m->b2[o];
        for (size_t k = 0; k < H; k++) a += w[k] * m->h[k];
        m->u[o] = a;
        m->y[o] = tnn_F(a);
    }
    return m->y;
}

/* dy = ∂L/∂y. Gradients use the pre-update weights (textbook back-prop). */
void cmlp_backward(CMlp *m, const double *dy)
{
    size_t in = m->in, H = m->hid;
    m->t++;
    TnnAdamBias bc = tnn_adam_bias(m->t);
    double lr = m->lr;

    memset(m->gh, 0, H * sizeof(double));
    for (size_t o = 0; o < m->out; o++) {
        double gu = dy[o] * tnn_dF(m->u[o]);
        double *w = m->W2 + o * H;
        for (size_t k = 0; k < H; k++) {
            m->gh[k] += gu * w[k];
            w[k] -= tnn_adam(&m->mW2[o * H + k], &m->vW2[o * H + k],
                             gu * m->h[k], lr, bc);
        }
        m->b2[o] -= tnn_adam(&m->mb2[o], &m->vb2[o], gu, lr, bc);
    }
    for (size_t k = 0; k < H; k++) {
        double g = (m->h[k] > 0.0) ? m->gh[k] : 0.0;
        double *w = m->W1 + k * in;
        for (size_t j = 0; j < in; j++)
            w[j] -= tnn_adam(&m->mW1[k * in + j], &m->vW1[k * in + j],
                             g * m->x[j], lr, bc);
        m->b1[k] -= tnn_adam(&m->mb1[k], &m->vb1[k], g, lr, bc);
    }
}

size_t cmlp_params(const CMlp *m)
{
    return m->hid * (m->in + 1) + m->out * (m->hid + 1);
}

size_t cmlp_hidden(const CMlp *m) { return m->hid; }
