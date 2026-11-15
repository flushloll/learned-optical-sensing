# Stereo experiment results

## Supervised: error 100 ms ahead (cm), eyes 10 cm apart

| setup | readings/frame | fixed size: lateral | fixed: depth | random size: lateral | random: depth | random: depth near / mid / far |
|---|---:|---:|---:|---:|---:|---|
| classical, 1 eye (depth from size) | 4096 | 8.6 | 10.5 | 11.9 | 53.5 | 40.0 / 60.4 / 59.2 |
| classical, 2 eyes (triangulation) | 8192 | 12.8 | 15.1 | 11.7 | 13.8 | 5.7 / 15.0 / 20.6 |
| CNN, 1 eye | 4096 | 2.5 | 3.5 | 4.2 | 20.8 | 20.8 / 20.5 / 21.3 |
| CNN, 2 eyes | 8192 | 2.4 | 3.1 | 3.2 | 6.2 | 5.1 / 5.9 / 7.5 |
| masks, 1 eye, R=32 | 32 | 4.6 ± 0.0 | 8.0 ± 0.1 | 7.9 ± 0.1 | 23.1 ± 0.2 | 24.2 / 21.9 / 23.5 |
| masks, 2 separate eyes, R=32 | 32 | 4.6 ± 0.1 | 7.5 ± 0.0 | 7.7 ± 0.1 | 17.0 ± 0.2 | 16.3 / 18.3 / 16.2 |
| masks, binocular (each sees both eyes), R=32 | 32 | 4.2 ± 0.0 | 7.5 ± 0.1 | 7.7 ± 0.1 | 22.9 ± 0.1 | 24.1 / 21.6 / 23.0 |
| masks, 1 eye, R=128 | 128 | 4.0 ± 0.1 | 7.9 ± 0.1 | 7.7 ± 0.0 | 23.0 ± 0.1 | 24.2 / 21.9 / 23.2 |
| masks, 2 separate eyes, R=128 | 128 | 4.0 ± 0.1 | 7.3 ± 0.1 | 7.4 ± 0.0 | 16.4 ± 0.1 | 16.2 / 17.9 / 14.9 |
| masks, binocular (each sees both eyes), R=128 | 128 | 3.8 ± 0.1 | 7.4 ± 0.2 | 7.6 ± 0.1 | 22.8 ± 0.1 | 24.2 / 21.6 / 22.9 |

Near / mid / far = true depth 1-1.67 / 1.67-2.33 / 2.33-3 m. Lateral = error in x, y combined. ± = std over 2 seeds.

## Eye separation (CNN, 2 eyes, random size, error 100 ms ahead, cm)

| eye separation | lateral | depth | depth near / mid / far |
|---:|---:|---:|---|
| one eye | 4.2 | 20.8 | 20.8 / 20.5 / 21.3 |
| 3 cm | 3.5 | 8.4 | 6.4 / 9.0 / 9.8 |
| 10 cm | 3.2 | 6.2 | 5.1 / 5.9 / 7.5 |
| 30 cm | 3.0 | 4.8 | 4.2 / 4.6 / 5.7 |

## Does depth emerge without being taught? (linear probe on the hidden state)

R² of a linear readout from the model's final hidden state (1 = perfect, 0 = no better than guessing the average). Self-supervised models were trained only to predict their own future frames; they never saw a depth label. 'untrained' = same network with random weights, as a control.

| model | depth now | depth +100 ms | ball radius | x now | depth now, RMSE cm |
|---|---:|---:|---:|---:|---:|
| self-supervised CNN, 1 eye | 0.69 | 0.67 | 0.83 | 0.98 | 28.7 |
| self-supervised CNN, 1 eye (untrained) | 0.25 | 0.24 | 0.47 | 0.42 | 44.5 |
| self-supervised CNN, 2 eyes | 0.72 | 0.71 | 0.80 | 0.98 | 26.9 |
| self-supervised CNN, 2 eyes (untrained) | 0.33 | 0.31 | 0.47 | 0.42 | 42.1 |
| CNN, 1 eye | 0.74 | 0.72 | 0.84 | 0.99 | 26.1 |
| CNN, 1 eye (untrained) | 0.25 | 0.24 | 0.47 | 0.42 | 44.5 |
| CNN, 2 eyes | 0.99 | 0.98 | 0.78 | 1.00 | 3.7 |
| CNN, 2 eyes (untrained) | 0.33 | 0.31 | 0.47 | 0.42 | 42.1 |

Supervised CNN rows are trained on 3D positions, so their depth R² is expected to be high; they are included as a ceiling.

