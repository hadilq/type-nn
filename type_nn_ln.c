#include "type_nn_ln.h"

#include <math.h>
#include <string.h>
#include <float.h>

static Network *g_ln = NULL;

void tnn_ln_bind(Network *net)
{
    g_ln = net;
}

int tnn_ln_on(void)
{
    return g_ln && (g_ln->andpol & TNN_AP_LOG);
}

int tnn_ln_bit(unsigned bit)
{
    return tnn_ln_on() && g_ln && (g_ln->lnpol & bit);
}

/*
 * Every named ln-* recipe is AP_LOG. LN_LOGZ is on unless the name is
 * ln-naive: that is the linear-space control so the board can measure
 * what the log-space product itself is worth.
 *
 * Isolated bits sit on top of LOGZ so a win can be attributed.
 */
int tnn_ln_apply(Network *net, const char *name)
{
    if (!net || !name || !name[0]) return 0;
    unsigned ap = TNN_AP_LOG;
    unsigned lp = LN_LOGZ;

    if (!strcmp(name, "ln") || !strcmp(name, "log"))
        lp = LN_LOGZ;
    else if (!strcmp(name, "ln-naive"))
        lp = LN_NAIVE;
    else if (!strcmp(name, "ln-logz"))
        lp = LN_LOGZ;
    else if (!strcmp(name, "ln-stuck"))
        ap |= TNN_AP_STUCK;
    else if (!strcmp(name, "ln-budget"))
        ap |= TNN_AP_BUDGET;
    else if (!strcmp(name, "ln-tas")) {
        ap |= TNN_AP_TSGD | TNN_AP_ANDTAU;
        lp |= LN_ATAU;
    } else if (!strcmp(name, "ln-next")) {
        ap |= TNN_AP_TENS | TNN_AP_ANDTAU | TNN_AP_BUDGET;
        lp |= LN_ATAU;
    } else if (!strcmp(name, "ln-apos"))
        lp |= LN_APOS;
    else if (!strcmp(name, "ln-orclip"))
        lp |= LN_ORCLIP;
    else if (!strcmp(name, "ln-asmall"))
        lp |= LN_ASMALL;
    else if (!strcmp(name, "ln-alr"))
        lp |= LN_ALR;
    else if (!strcmp(name, "ln-y01"))
        lp |= LN_Y01;
    else if (!strcmp(name, "ln-atau"))
        lp |= LN_ATAU;
    else if (!strcmp(name, "ln-logz-tau"))
        lp |= LN_ATAU;
    else if (!strcmp(name, "ln-stable"))
        lp |= LN_ORCLIP;
    else if (!strcmp(name, "ln-clock")) {
        /* log-space + Or clip + the tanh clock that actually won */
        ap |= TNN_AP_TSGD | TNN_AP_ANDTAU;
        lp |= LN_ORCLIP | LN_ATAU;
    } else if (!strcmp(name, "ln-best")) {
        ap |= TNN_AP_BUDGET | TNN_AP_TENS | TNN_AP_TSGD | TNN_AP_ANDTAU;
        lp = LN_LOGZ | LN_APOS | LN_ORCLIP | LN_ASMALL | LN_ALR | LN_ATAU | LN_Y01;
    } else if (!strcmp(name, "ln-v2")) {
        /* two numeric winners: a≥0 and slow a */
        lp |= LN_APOS | LN_ALR;
    } else if (!strcmp(name, "ln-cap")) {
        lp |= LN_APOS | LN_AMAX;
    } else if (!strcmp(name, "ln-v2c")) {
        lp |= LN_APOS | LN_ALR | LN_AMAX;
    } else if (!strcmp(name, "ln-v2t")) {
        ap |= TNN_AP_TSGD | TNN_AP_ANDTAU;
        lp |= LN_APOS | LN_ALR | LN_ATAU;
    } else if (!strcmp(name, "ln-v2w")) {
        lp |= LN_APOS | LN_ALR | LN_WIDE;
    } else if (!strcmp(name, "ln-aw")) {
        lp |= LN_APOS | LN_WIDE;
    } else if (!strcmp(name, "ln-v2a")) {
        lp |= LN_APOS | LN_ALR | LN_ATAU;
    } else if (!strcmp(name, "ln-v3")) {
        /* numeric winners + tas clock + wide Or + a cap. No y01/asmall/orclip. */
        ap |= TNN_AP_TSGD | TNN_AP_ANDTAU;
        lp |= LN_APOS | LN_ALR | LN_AMAX | LN_ATAU | LN_WIDE;
    } else if (!strcmp(name, "ln-v3b")) {
        ap |= TNN_AP_TSGD | TNN_AP_ANDTAU | TNN_AP_BUDGET;
        lp |= LN_APOS | LN_ALR | LN_AMAX | LN_ATAU | LN_WIDE;
    } else if (!strcmp(name, "ln-taud") || !strcmp(name, "ln-sqrt")) {
        lp |= LN_APOS | LN_TAUD;
    } else if (!strcmp(name, "ln-v2w-taud") || !strcmp(name, "ln-v2w+taud")) {
        lp |= LN_APOS | LN_ALR | LN_WIDE | LN_TAUD;
    } else if (!strcmp(name, "taud")) {
        if (net->lnpol)
            lp = net->lnpol | LN_TAUD | LN_APOS;
        else
            lp |= LN_APOS | LN_TAUD;
        if (net->andpol & TNN_AP_LOG)
            ap = net->andpol;
    } else if (!strcmp(name, "ln-body") || !strcmp(name, "ln-tanh")) {
        lp |= LN_APOS | LN_ALR | LN_TANH_TAIL;
    } else if (!strcmp(name, "ln-adam")) {
        /* log-space + assembly a>0 + Adam on W, b, a, τ */
        lp |= LN_APOS | LN_ALR | LN_ADAM;
    } else if (!strcmp(name, "ln-v2w-adam") || !strcmp(name, "ln-v2adam")) {
        lp |= LN_APOS | LN_ALR | LN_WIDE | LN_ADAM;
    } else if (!strcmp(name, "adam") || !strcmp(name, "Adam")) {
        /* plus-form token: keep whatever ln bits are already on */
        if (net->lnpol)
            lp = net->lnpol | LN_ADAM | LN_APOS;
        else
            lp |= LN_APOS | LN_ADAM;
        if (net->andpol & TNN_AP_LOG)
            ap = net->andpol;
    } else
        return 0;

    /* Assembly index is part of the type generating function: a_{i,r} > 0
       on every ln recipe, including the naive product control. */
    lp |= LN_APOS;
    net->andpol = ap;
    net->lnpol = lp;
    net->orcool = 0;
    return 1;
}

double tnn_ln_signed_pow(double base, double p)
{
    if (isnan(base) || isnan(p)) return 0.0;
    if (base == 0.0) return (p > 0.0) ? 0.0 : 1.0;
    double mag = exp(p * log(fabs(base)));
    if (!isfinite(mag)) mag = (p > 0.0) ? DBL_MAX : 0.0;
    return copysign(mag, base);
}

int tnn_ln_and_is_dummy(const AndNode *a)
{
    if (!a) return 1;
    const OrNode *o = a->or_row;
    while (o) {
        if (fabs(o->bias.value - 1.0) > 0.2) return 0;
        const WeightNode *w = o->weight;
        while (w) {
            if (fabs(w->value) > 0.2) return 0;
            w = w->right;
        }
        o = o->right;
    }
    return 1;
}

double tnn_ln_and_term(const AndNode *a)
{
    if (!a) return 1.0;
    if (!tnn_ln_on()) return a->value;
    return tnn_ln_signed_pow(a->value, a->expn);
}

/*
 * A = σ exp(clip(Σ log|Õ|)).  Sets a->value and a->and_gate = ∂λ̃/∂λ.
 * Without LN_LOGZ this just returns the already-computed linear product
 * and gate = 1.
 */
double tnn_ln_and_eval(AndNode *a)
{
    if (!a) return 1.0;
    a->and_gate = 1.0;
    if (!tnn_ln_bit(LN_LOGZ))
        return a->value;

    int sign = 1;
    int zero = 0;
    double ell = 0.0;
    const OrNode *o = a->or_row;
    while (o) {
        double v = o->value;
        if (v == 0.0 || !isfinite(v)) {
            zero = 1;
            break;
        }
        if (v < 0.0) sign = -sign;
        double mag = fabs(v);
        if (mag < LN_LOG_EPS) mag = LN_LOG_EPS;
        ell += log(mag);
        o = o->right;
    }
    if (zero) {
        a->value = 0.0;
        a->and_gate = 0.0;
        return 0.0;
    }
    double ell_s = ell;
    a->and_gate = 1.0;
    if (ell > LN_ELL_CAP) { ell_s = LN_ELL_CAP; a->and_gate = 0.0; }
    if (ell < -LN_ELL_CAP) { ell_s = -LN_ELL_CAP; a->and_gate = 0.0; }
    double mag = exp(ell_s);
    if (!isfinite(mag)) mag = DBL_MAX;
    a->value = (sign < 0) ? -mag : mag;
    if (!isfinite(a->value)) a->value = 0.0;
    return a->value;
}

void tnn_ln_and_accums(AndNode *a)
{
    if (!a) return;
    /*
     * ∂A/∂Õ_t = A · γ / Õ_t   when LN_LOGZ
     *         = Π_{s≠t} Õ_s    otherwise (caller uses compute_accums)
     */
    if (!tnn_ln_bit(LN_LOGZ)) return;
    double A = a->value;
    double g = a->and_gate;
    OrNode *o = a->or_row;
    while (o) {
        if (isfinite(A) && isfinite(o->value) && fabs(o->value) > LN_LOG_EPS)
            o->accum = A * g / o->value;
        else
            o->accum = 0.0;
        if (!isfinite(o->accum)) o->accum = 0.0;
        o = o->right;
    }
}

TnnLnZ tnn_ln_eval(const AndNode *row, size_t index)
{
    TnnLnZ o;
    o.z = 1.0;
    o.ell = 0.0;
    o.ell_s = 0.0;
    o.gate = 1.0;

    int logz = tnn_ln_bit(LN_LOGZ);
    int sign = 1;
    int zero = 0;
    const AndNode *a = row;
    while (a) {
        if (a->right_index == index) {
            double A = a->value;
            double p = tnn_ln_on() ? a->expn : 1.0;
            if (A == 0.0 || !isfinite(A)) {
                if (p > 0.0) { zero = 1; break; }
            } else if (logz) {
                if (A < 0.0) sign = -sign;
                double mag = fabs(A);
                if (mag < LN_LOG_EPS) mag = LN_LOG_EPS;
                o.ell += p * log(mag);
            } else {
                o.z *= tnn_ln_signed_pow(A, p);
            }
        }
        a = a->right;
    }
    if (zero) {
        o.z = 0.0;
        o.ell = -LN_ELL_CAP;
        o.ell_s = -LN_ELL_CAP;
        o.gate = 0.0;
        return o;
    }
    if (logz) {
        o.ell_s = o.ell;
        o.gate = 1.0;
        if (o.ell > LN_ELL_CAP) { o.ell_s = LN_ELL_CAP; o.gate = 0.0; }
        if (o.ell < -LN_ELL_CAP) { o.ell_s = -LN_ELL_CAP; o.gate = 0.0; }
        double mag = exp(o.ell_s);
        if (!isfinite(mag)) mag = DBL_MAX;
        o.z = (sign < 0) ? -mag : mag;
    }
    if (!isfinite(o.z)) o.z = 0.0;
    return o;
}

double tnn_ln_tau(const Layer *l, size_t index, double tau_sqrt_d)
{
    if (tau_sqrt_d < 1e-12) tau_sqrt_d = 1.0;
    if (!tnn_ln_bit(LN_ATAU) || !l) return tau_sqrt_d;
    double s = 0.0;
    size_t n = 0;
    const AndNode *a = l->and_row;
    while (a) {
        if (a->right_index == index && !tnn_ln_and_is_dummy(a)) {
            n++;
            s += fabs(a->expn);
        }
        a = a->right;
    }
    if (s > 1.0) return tau_sqrt_d * s;
    if (n > 1) return tau_sqrt_d * (double)n;
    return tau_sqrt_d;
}

double tnn_ln_readout(double z, double tau)
{
    if (tau < 1e-12) tau = 1.0;
    if (isnan(z)) return 0.0;
    if (!isfinite(z))
        return copysign(log(DBL_MAX), z);
    double u = z / tau;
    if (u == 0.0) return 0.0;
    double y = copysign(log1p(fabs(u)), u);
    if (tnn_ln_bit(LN_Y01))
        y /= log(2.0);
    return y;
}

double tnn_ln_dydz(double z, double tau)
{
    if (tau < LN_TAU_MIN) tau = LN_TAU_MIN;
    if (!isfinite(z)) return 0.0;
    double d = 1.0 / (tau + fabs(z));
    if (tnn_ln_bit(LN_Y01))
        d /= log(2.0);
    return d;
}

double tnn_ln_dydtau(double z, double tau)
{
    /* y = s ln(1+|z|/τ),  ∂y/∂τ = −s |z| / (τ (τ+|z|)) */
    if (tau < LN_TAU_MIN) tau = LN_TAU_MIN;
    if (!isfinite(z) || z == 0.0) return 0.0;
    double s = (z > 0.0) ? 1.0 : -1.0;
    double az = fabs(z);
    double d = -s * az / (tau * (tau + az));
    if (tnn_ln_bit(LN_Y01))
        d /= log(2.0);
    return d;
}

void tnn_ln_step_tau(Layer *l, size_t index, double g_tau, double lr)
{
    if (!l || !l->tau || index >= l->tau_n) return;
    if (!isfinite(g_tau)) g_tau = 0.0;
    l->tau_g[index] = g_tau;
    double eta = lr;
    if (tnn_ln_bit(LN_ALR)) eta *= 0.1;
    if (tnn_ln_bit(LN_ADAM) && g_ln) {
        eta *= 0.1;
        unsigned long t = g_ln->adam_t ? g_ln->adam_t : 1;
        l->tau_m[index] = LN_ADAM_B1 * l->tau_m[index] + (1.0 - LN_ADAM_B1) * g_tau;
        l->tau_v[index] = LN_ADAM_B2 * l->tau_v[index] + (1.0 - LN_ADAM_B2) * g_tau * g_tau;
        double b1p = g_ln->adam_b1p > 0.0 && g_ln->adam_b1p < 1.0
            ? g_ln->adam_b1p : (1.0 - pow(LN_ADAM_B1, (double)t));
        double b2p = g_ln->adam_b2p > 0.0 && g_ln->adam_b2p < 1.0
            ? g_ln->adam_b2p : (1.0 - pow(LN_ADAM_B2, (double)t));
        double mh = l->tau_m[index] / (1.0 - b1p);
        double vh = l->tau_v[index] / (1.0 - b2p);
        if (vh < 0.0) vh = 0.0;
        l->tau[index] -= eta * mh / (sqrt(vh) + LN_ADAM_EPS);
    } else {
        l->tau[index] -= eta * g_tau;
    }
    if (!isfinite(l->tau[index]) || l->tau[index] < LN_TAU_MIN)
        l->tau[index] = LN_TAU_MIN;
}

double tnn_ln_lr_a(double lr)
{
    if (tnn_ln_bit(LN_ALR)) return lr * 0.1;
    return lr;
}

double tnn_ln_project_a(double a)
{
    if (!isfinite(a) || a < LN_A_MIN)
        a = LN_A_MIN;
    if (tnn_ln_bit(LN_AMAX) && a > LN_A_CAP)
        a = LN_A_CAP;
    return a;
}

void tnn_ln_init_expn(Network *net)
{
    if (!net || !tnn_ln_on()) return;
    for (Layer *l = net->head; l; l = l->next) {
        double a0 = 1.0;
        if (tnn_ln_bit(LN_ASMALL)) {
            size_t d = l->in_size ? l->in_size : 1;
            a0 = 1.0 / sqrt((double)d);
        }
        a0 = tnn_ln_project_a(a0);
        for (AndNode *a = l->and_row; a; a = a->right) {
            a->expn = a0;
            a->expn_m = 0.0;
            a->expn_v = 0.0;
        }
    }
}

double tnn_ln_or_clip(double or_value, int is_dummy)
{
    if (!tnn_ln_bit(LN_ORCLIP) || is_dummy) return or_value;
    if (or_value > LN_OR_CAP) return LN_OR_CAP;
    if (or_value < -LN_OR_CAP) return -LN_OR_CAP;
    return or_value;
}

int tnn_ln_or_clip_gate(double or_value, int is_dummy)
{
    if (!tnn_ln_bit(LN_ORCLIP) || is_dummy) return 1;
    if (or_value >= LN_OR_CAP || or_value <= -LN_OR_CAP) return 0;
    return 1;
}

void tnn_ln_step_expn(AndNode *a, double dL_dz, double z, double gate, double lr)
{
    if (!a || !tnn_ln_on()) return;
    /* tas clock freezes dummy-And SGD, not live assembly indices. */
    if (g_ln && (g_ln->andpol & TNN_AP_TSGD)
        && g_ln->orcool_step % 32u != 0
        && tnn_ln_and_is_dummy(a))
        return;
    double A = a->value;
    double g_a = 0.0;
    if (isfinite(z) && isfinite(A) && fabs(A) > LN_LOG_EPS)
        g_a = dL_dz * z * gate * log(fabs(A));
    if (!isfinite(g_a)) g_a = 0.0;
    a->expn_grad = g_a;
    double eta = tnn_ln_lr_a(lr);
    double na;
    if (tnn_ln_bit(LN_ADAM) && g_ln) {
        unsigned long t = g_ln->adam_t ? g_ln->adam_t : 1;
        a->expn_m = LN_ADAM_B1 * a->expn_m + (1.0 - LN_ADAM_B1) * g_a;
        a->expn_v = LN_ADAM_B2 * a->expn_v + (1.0 - LN_ADAM_B2) * g_a * g_a;
        double b1p = g_ln->adam_b1p > 0.0 ? g_ln->adam_b1p : (1.0 - pow(LN_ADAM_B1, (double)t));
        double b2p = g_ln->adam_b2p > 0.0 ? g_ln->adam_b2p : (1.0 - pow(LN_ADAM_B2, (double)t));
        if (b1p >= 1.0) b1p = 1.0 - 1e-12;
        if (b2p >= 1.0) b2p = 1.0 - 1e-12;
        double mh = a->expn_m / (1.0 - b1p);
        double vh = a->expn_v / (1.0 - b2p);
        if (vh < 0.0) vh = 0.0;
        na = a->expn - eta * mh / (sqrt(vh) + LN_ADAM_EPS);
    } else {
        na = a->expn - eta * g_a;
    }
    double q = a->quantization > 0.0 ? a->quantization : 1e-7;
    if (fabs(na - 1.0) < q) na = 1.0;
    a->expn = tnn_ln_project_a(na);
}

int tnn_ln_zero_expn(const AndNode *a)
{
    /* Live clauses keep a ≥ LN_A_MIN. Hitting the floor means the
       clause assembled to nothing and may be dropped; a new dummy
       is born at a = 1. */
    if (!a) return 0;
    return a->expn <= LN_A_MIN * 1.0000001;
}
