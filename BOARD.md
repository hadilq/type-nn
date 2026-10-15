type-nn board: graph-type-nn, type-nn, type-nn-overfit, c-mlp
==============================================================

Splits     1 x 70/30 split(s) (xorshift32 Fisher-Yates; split 0 is seed 34972);
           standardisation fitted on the training rows of each split only.
Seeds      5 initialisation seeds per split; every number is the mean over
           all (split, seed) runs, ± is their standard deviation.
Protocol   one harness (bench.c) for every model: same shuffle, same
           per-sample Adam (common.h), same mean-MSE, same epochs/lr,
           same readout F(u) = sign(u) ln(1+|u|). The hold-out is read
           only after training ends.
params     every learnable scalar present after training (dense; no weight
           is skipped for being small). Edge weights of the graph are set
           by feedback, not by gradient, and are not counted.
L0->L      layers (chain) or nodes (graph) at birth -> after training.
E0->E      edges at birth -> after training, x -> first layer included.
or/and     Or-width and And-degree dummies promoted (+) and live items
           pruned (-); L+/L- layers or nodes, E+/E- graph edges.
models     graph-type-nn    type-nn on a DAG, slime-mould edge weights
           type-nn          evidence-driven scaling (BIC on measured MSE)
           type-nn-overfit  displacement-only scaling (frozen reference)
           c-mlp            Linear-ReLU-Linear, fixed hidden width 8 / 16
           c-mlp-matched    the widest c-mlp with params <= graph-type-nn's
verdict    hold_mse difference, 'clear' if it exceeds two standard errors
           (paired over splits when there are several), and the params ratio.
goal       graph-type-nn: hold_mse not clearly worse than type-nn, with
           fewer parameters.
xor        fit only: 4 points have no hold-out.

task        impl             hold_mse       ± hold_acc      ± train_mse  params      ± train_s  us/inf     L0->L     E0->E   or+  or-  and+ and-   L+   L-   E+   E-
--------------------------------------------------------------------------------------------------------------------------------------------------------------------
xor         type-nn               n/a  0.0000      n/a  0.000   0.00000    26.4    3.6   0.012   0.105    1->2.0    1->2.0   0.0  0.0   3.6  0.0  1.0  0.0          
xor         type-nn-overfit       n/a  0.0000      n/a  0.000   0.00000   107.6   73.4   0.012   0.523    1->3.0    1->3.0   2.0  0.0  16.6  0.4  2.0  0.0          
xor         graph-type-nn         n/a  0.0000      n/a  0.000   0.00000    24.8    1.8   0.016   0.115    1->2.0    1->2.0   0.0  0.0   3.2  0.0  1.0  0.0  0.0  0.0
xor         c-mlp                 n/a  0.0000      n/a  0.000   0.00000    33.0    0.0   0.001   0.030    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
xor         c-mlp-matched         n/a  0.0000      n/a  0.000   0.05010    21.0    0.0   0.001   0.023    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: fit only; params 0.94x (fewer)
            graph-type-nn vs c-mlp: fit only; params 0.75x (fewer)
            graph-type-nn vs c-mlp-matched: fit only; params 1.18x (more)
            type-nn vs c-mlp: fit only; params 0.80x (fewer)

iris        type-nn            0.0215  0.0069    0.969  0.012   0.00084   125.4   52.2   0.358   0.517    3->3.8    3->3.8   1.0  1.4  12.8  0.6  0.8  0.0          
iris        type-nn-overfit    0.0249  0.0044    0.960  0.010   0.00181   174.6   29.8   0.066   0.475    3->4.0    3->4.0   2.6  1.4  19.2  1.8  1.0  0.0          
iris        graph-type-nn      0.0218  0.0065    0.969  0.012   0.00085    80.6   16.6   0.243   0.218    3->3.8    4->5.2   2.6  5.6  10.8  3.2  1.0  0.2  1.0  0.4
iris        c-mlp              0.0280  0.0061    0.956  0.000   0.01087    67.0    0.0   0.011   0.043    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
iris        c-mlp-matched      0.0315  0.0035    0.956  0.000   0.01195    75.0    0.0   0.010   0.043    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: type-nn lower hold_mse by 0.0003 (2se 0.0085, over 5 seeds) -> within noise; params 0.64x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0062 (2se 0.0080, over 5 seeds) -> within noise; params 1.20x (more)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0097 (2se 0.0066, over 5 seeds) -> clear; params 1.07x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0065 (2se 0.0083, over 5 seeds) -> within noise; params 1.87x (more)

wine        type-nn            0.0277  0.0048    0.956  0.010   0.00008   169.8   52.3   0.406   0.527    4->5.0    4->5.0   2.0  2.6  15.4  3.2  1.2  0.2          
wine        type-nn-overfit    0.0209  0.0105    0.967  0.020   0.00016   392.6   43.7   0.091   0.718    4->5.0    4->5.0   4.6  0.6  27.4  0.8  1.0  0.0          
wine        graph-type-nn      0.0184  0.0039    0.967  0.008   0.00011   165.8   49.1   0.541   0.439    4->5.4    6->8.2   3.0  7.0  21.0  5.4  1.4  0.0  2.6  0.8
wine        c-mlp              0.0234  0.0056    0.974  0.017   0.00224   275.0    0.0   0.022   0.101    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
wine        c-mlp-matched      0.0261  0.0072    0.981  0.023   0.00309   156.0    0.0   0.014   0.073    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0093 (2se 0.0055, over 5 seeds) -> clear; params 0.98x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0050 (2se 0.0061, over 5 seeds) -> within noise; params 0.60x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0077 (2se 0.0073, over 5 seeds) -> clear; params 1.06x (more)
            type-nn vs c-mlp: c-mlp lower hold_mse by 0.0043 (2se 0.0066, over 5 seeds) -> within noise; params 0.62x (fewer)

wdbc        type-nn            0.0448  0.0048    0.951  0.007   0.00010   162.8   28.7   0.138   0.284    3->4.0    3->4.0   1.2  0.0   5.6  0.0  1.0  0.0          
wdbc        type-nn-overfit    0.0459  0.0030    0.950  0.003   0.00042   162.6   31.2   0.049   0.235    3->3.2    3->3.2   1.2  0.0   5.2  0.6  0.4  0.2          
wdbc        graph-type-nn      0.0398  0.0035    0.954  0.005   0.00077   128.4   55.0   0.207   0.259    3->4.0    4->5.2   2.8  1.4   8.4  1.4  1.0  0.0  0.8  0.4
wdbc        c-mlp              0.0438  0.0017    0.946  0.008   0.00518   513.0    0.0   0.073   0.144    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
wdbc        c-mlp-matched      0.0467  0.0038    0.949  0.008   0.01664    97.0    0.0   0.011   0.048    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0051 (2se 0.0053, over 5 seeds) -> within noise; params 0.79x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0041 (2se 0.0035, over 5 seeds) -> clear; params 0.25x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0069 (2se 0.0046, over 5 seeds) -> clear; params 1.32x (more)
            type-nn vs c-mlp: c-mlp lower hold_mse by 0.0010 (2se 0.0046, over 5 seeds) -> within noise; params 0.32x (fewer)

diabetes    type-nn            0.0329  0.0010      n/a  0.000   0.02733    20.4    7.2   0.046   0.133    2->2.4    2->2.4   0.0  0.0   1.8  0.8  0.8  0.4          
diabetes    type-nn-overfit    0.0371  0.0015      n/a  0.000   0.02404    33.0    8.5   0.027   0.138    2->2.2    2->2.2   0.0  0.0   2.4  0.8  0.4  0.2          
diabetes    graph-type-nn      0.0332  0.0012      n/a  0.000   0.02728    18.8    6.9   0.056   0.163    2->2.2    2->2.4   1.0  1.0   1.8  1.4  0.8  0.6  0.0  0.0
diabetes    c-mlp              0.0488  0.0042      n/a  0.000   0.01354   193.0    0.0   0.029   0.078    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
diabetes    c-mlp-matched      0.0353  0.0027      n/a  0.000   0.02854    13.0    0.0   0.005   0.021    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: type-nn lower hold_mse by 0.0003 (2se 0.0014, over 5 seeds) -> within noise; params 0.92x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0156 (2se 0.0039, over 5 seeds) -> clear; params 0.10x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0021 (2se 0.0027, over 5 seeds) -> within noise; params 1.45x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0158 (2se 0.0038, over 5 seeds) -> clear; params 0.11x (fewer)

ionosphere  type-nn            0.0886  0.0111    0.904  0.014   0.00970   153.8   65.0   0.185   0.088    4->4.4    4->4.4   2.2  0.6   7.2  1.0  1.2  0.8          
ionosphere  type-nn-overfit    0.0813  0.0119    0.913  0.018   0.00433   298.0   60.5   0.070   0.540    4->4.4    4->4.4   4.2  0.2  12.0  0.4  0.6  0.2          
ionosphere  graph-type-nn      0.0862  0.0209    0.902  0.027   0.00779   221.4  133.0   0.461   0.549    4->4.8    6->7.4   4.6  1.2  14.4  2.8  1.0  0.2  2.2  0.6
ionosphere  c-mlp              0.1461  0.0155    0.896  0.018   0.01631   577.0    0.0   0.052   0.166    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
ionosphere  c-mlp-matched      0.1478  0.0332    0.900  0.024   0.01614   217.0    0.0   0.020   0.082    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0023 (2se 0.0212, over 5 seeds) -> within noise; params 1.44x (more)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0599 (2se 0.0233, over 5 seeds) -> clear; params 0.38x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0615 (2se 0.0351, over 5 seeds) -> clear; params 1.02x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0575 (2se 0.0170, over 5 seeds) -> clear; params 0.27x (fewer)

spambase    type-nn            0.0597  0.0055    0.929  0.007   0.02903   318.4   66.8   1.090   0.348    4->5.0    4->5.0   2.4  0.0   7.0  0.2  1.0  0.0          
spambase    type-nn-overfit    0.0514  0.0019    0.936  0.004   0.04230   146.2   24.8   0.269   0.240    4->3.8    4->3.8   2.0  0.4   1.4  0.2  0.0  0.2          
spambase    graph-type-nn      0.0570  0.0050    0.933  0.006   0.03056   335.4  142.8   1.903   0.482    4->5.0    6->6.6   4.4  1.2   9.6  2.0  1.0  0.0  1.8  1.0
spambase    c-mlp              0.0534  0.0016    0.941  0.003   0.03941   945.0    0.0   0.355   0.276    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
spambase    c-mlp-matched      0.0549  0.0020    0.935  0.005   0.04855   296.0    0.0   0.116   0.105    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0027 (2se 0.0067, over 5 seeds) -> within noise; params 1.05x (more)
            graph-type-nn vs c-mlp: c-mlp lower hold_mse by 0.0036 (2se 0.0047, over 5 seeds) -> within noise; params 0.35x (fewer)
            graph-type-nn vs c-mlp-matched: c-mlp-matched lower hold_mse by 0.0021 (2se 0.0048, over 5 seeds) -> within noise; params 1.13x (more)
            type-nn vs c-mlp: c-mlp lower hold_mse by 0.0063 (2se 0.0052, over 5 seeds) -> clear; params 0.34x (fewer)

synth       type-nn            0.0919  0.0253    0.863  0.045   0.07151    79.6   19.3   0.501   0.215    3->4.0    3->4.0   1.0  0.2   3.6  0.2  1.0  0.0          
synth       type-nn-overfit    0.2392  0.0079    0.587  0.022   0.23603    22.2    1.6   0.139   0.114    3->2.4    3->2.4   0.2  0.2   1.2  1.2  0.0  0.6          
synth       graph-type-nn      0.0862  0.0090    0.879  0.012   0.05445   135.0   43.1   1.285   0.433    3->3.6    4->5.2   3.0  0.6   7.8  1.0  1.0  0.4  1.0  0.4
synth       c-mlp              0.1179  0.0053    0.863  0.004   0.08463   289.0    0.0   0.299   0.100    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
synth       c-mlp-matched      0.1210  0.0060    0.843  0.013   0.09322   127.0    0.0   0.072   0.058    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0057 (2se 0.0240, over 5 seeds) -> within noise; params 1.70x (more)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0316 (2se 0.0094, over 5 seeds) -> clear; params 0.47x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0347 (2se 0.0097, over 5 seeds) -> clear; params 1.06x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0260 (2se 0.0231, over 5 seeds) -> clear; params 0.28x (fewer)

mnist       type-nn            0.0863  0.0040    0.184  0.064   0.08564   243.8   10.0   5.392   0.448    8->6.4    8->6.4   4.0 74.0  27.8 17.4  0.8  2.4          
mnist       type-nn-overfit    0.0849  0.0057    0.191  0.100   0.08415   378.0  128.2   0.495   1.029    8->7.8    8->7.8   0.0 57.8  11.8  2.0  0.0  0.2          
mnist       graph-type-nn      0.0837  0.0038    0.209  0.063   0.08258   231.2    4.1  11.090   0.473    8->3.0   19->3.2   5.6 75.2  46.2 23.4  0.8  5.8  3.6  3.2
mnist       c-mlp              0.0382  0.0017    0.854  0.007   0.01287  3322.0    0.0   0.197   1.163    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
mnist       c-mlp-matched      0.0832  0.0007    0.201  0.006   0.08074   217.0    0.0   0.017   0.172    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0026 (2se 0.0050, over 5 seeds) -> within noise; params 0.95x (fewer)
            graph-type-nn vs c-mlp: c-mlp lower hold_mse by 0.0455 (2se 0.0038, over 5 seeds) -> clear; params 0.07x (fewer)
            graph-type-nn vs c-mlp-matched: c-mlp-matched lower hold_mse by 0.0005 (2se 0.0035, over 5 seeds) -> within noise; params 1.07x (more)
            type-nn vs c-mlp: c-mlp lower hold_mse by 0.0481 (2se 0.0039, over 5 seeds) -> clear; params 0.07x (fewer)

mnist_large type-nn            0.0160  0.0010    0.900  0.007   0.00933  6094.6  110.9  67.498   6.134    8->9.0    8->9.0   5.8  2.0  76.8  9.2  1.0  0.0          
mnist_large type-nn-overfit    0.0873  0.0039    0.137  0.054   0.08712   240.6    1.3   1.430   0.484    8->6.2    8->6.2   0.0 63.0   0.6  0.6  0.0  1.8          
mnist_large graph-type-nn      0.0158  0.0011    0.913  0.004   0.01069  4912.6  704.4  81.403   3.704    8->7.8   19->13.4   5.4 45.4  70.8 13.4  1.0  1.2  7.8  2.0
mnist_large c-mlp              0.0228  0.0002    0.904  0.003   0.01702  3322.0    0.0   0.539   1.178    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
mnist_large c-mlp-matched      0.0225  0.0007    0.910  0.002   0.01649  4771.0    0.0   0.767   1.613    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0002 (2se 0.0013, over 5 seeds) -> within noise; params 0.81x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0069 (2se 0.0010, over 5 seeds) -> clear; params 1.48x (more)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0067 (2se 0.0012, over 5 seeds) -> clear; params 1.03x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0068 (2se 0.0009, over 5 seeds) -> clear; params 1.83x (more)

video       type-nn            0.0000  0.0000    1.000  0.000   0.00002   823.2  488.2   0.306   0.968    6->7.0    6->7.0   6.6  1.4  17.2  2.4  1.0  0.0          
video       type-nn-overfit    0.0000  0.0000    1.000  0.000   0.00004  1029.0  359.8   0.093   0.716    6->6.0    6->6.0   9.0  0.6   9.8  0.2  0.0  0.0          
video       graph-type-nn      0.0001  0.0001    1.000  0.000   0.00012   529.2  404.8   0.328   0.388    6->4.8   10->7.2   6.0  3.8  18.0  5.6  1.0  2.2  5.6  2.4
video       c-mlp              0.0004  0.0001    1.000  0.000   0.00039  4129.0    0.0   0.106   1.424    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
video       c-mlp-matched      0.0005  0.0003    1.000  0.000   0.00042   517.0    0.0   0.015   0.232    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: type-nn lower hold_mse by 0.0001 (2se 0.0001, over 5 seeds) -> clear; params 0.64x (fewer)
            graph-type-nn vs c-mlp: graph-type-nn lower hold_mse by 0.0003 (2se 0.0001, over 5 seeds) -> clear; params 0.13x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0003 (2se 0.0003, over 5 seeds) -> clear; params 1.02x (more)
            type-nn vs c-mlp: type-nn lower hold_mse by 0.0004 (2se 0.0001, over 5 seeds) -> clear; params 0.20x (fewer)

video_large type-nn            0.0678  0.0030    0.896  0.003   0.04184   406.8  132.7   0.649   0.452    6->6.8    6->6.8   3.0  1.2   7.8  1.4  1.0  0.2          
video_large type-nn-overfit    0.0641  0.0022    0.897  0.007   0.04284   445.6  141.3   0.236   0.501    6->5.8    6->5.8   2.8  0.8   4.4  0.4  0.0  0.2          
video_large graph-type-nn      0.0645  0.0014    0.897  0.006   0.04292   392.0  149.4   0.982   0.665    6->5.4   10->7.2   4.4  3.0   9.0  4.0  1.0  1.6  2.2  1.0
video_large c-mlp              0.0623  0.0016    0.899  0.001   0.03195  4129.0    0.0   0.369   1.435    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
video_large c-mlp-matched      0.0711  0.0097    0.889  0.011   0.05464   259.0    0.0   0.028   0.152    2->2.0    2->2.0   0.0  0.0   0.0  0.0  0.0  0.0          
            graph-type-nn vs type-nn: graph-type-nn lower hold_mse by 0.0033 (2se 0.0030, over 5 seeds) -> clear; params 0.96x (fewer)
            graph-type-nn vs c-mlp: c-mlp lower hold_mse by 0.0022 (2se 0.0019, over 5 seeds) -> clear; params 0.09x (fewer)
            graph-type-nn vs c-mlp-matched: graph-type-nn lower hold_mse by 0.0066 (2se 0.0088, over 5 seeds) -> within noise; params 1.51x (more)
            type-nn vs c-mlp: c-mlp lower hold_mse by 0.0055 (2se 0.0030, over 5 seeds) -> clear; params 0.10x (fewer)

goal met on 8 / 12 tasks: xor, iris, wine, wdbc, diabetes, mnist, mnist_large, video_large
