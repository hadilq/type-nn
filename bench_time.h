#ifndef TYPE_NN_BENCH_TIME_H
#define TYPE_NN_BENCH_TIME_H

/*
 * Wall-clock helper for the benches.
 *
 * A bare
 *     t0 = now(); for (r) predict(...); dt = now()-t0;
 * can print us/infer = 0 for two reasons:
 *
 *   1. The compiler sees `pred` is never read and deletes the loop
 *      (same-TU function pointers in bench_alts + -O2).
 *   2. A few thousand calls finish inside one clock tick on a coarse
 *      CLOCK_MONOTONIC, so dt rounds to 0.000 when printed with %.3f.
 *
 * Fix: accumulate a sink the optimizer cannot drop, and keep calling
 * until at least BENCH_INFER_MIN_S of wall time has elapsed.
 */

#include <time.h>
#include <stddef.h>

#define BENCH_INFER_MIN_S 0.05

static inline double bench_wall_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* `work(ctx)` must perform one forward. Returns elapsed seconds.
   *n_out is the number of calls actually issued. */
static inline double bench_time_infer(void (*work)(void *), void *ctx,
                                      size_t min_reps, double *n_out)
{
    volatile double sink = 0.0;
    size_t n = 0;
    double t0 = bench_wall_s();
    do {
        size_t batch = min_reps ? min_reps : 1000;
        for (size_t i = 0; i < batch; i++) {
            work(ctx);
            sink += 1.0;
        }
        n += batch;
    } while (bench_wall_s() - t0 < BENCH_INFER_MIN_S);
    /* Keep sink live across the loop. */
    if (sink < 0.0)
        n += (size_t)sink;
    if (n_out)
        *n_out = (double)n;
    return bench_wall_s() - t0;
}

#endif
