#ifndef TYPE_NN_BENCH_TIME_H
#define TYPE_NN_BENCH_TIME_H

#include <time.h>

/* Inference is timed for at least this much wall time so a fast model
   never prints 0 us/infer. */
#define BENCH_INFER_MIN_S 0.20

static inline double bench_wall_s(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

#endif
