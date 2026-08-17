# Avaliação da detecção de defeitos

## Abordagem: regras

IoU médio (localização): 0.853

| classe | precisão | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| furo_ausente | 0.806 | 0.962 | 0.877 | 25 | 6 | 1 |
| mancha | 0.512 | 0.733 | 0.603 | 22 | 21 | 8 |
| risco | 0.593 | 0.593 | 0.593 | 16 | 11 | 11 |

Matriz de confusão (linha = tipo real, coluna = tipo predito):

| real \ predito | furo_ausente | mancha | risco | nao_detectado |
|---|---|---|---|---|
| furo_ausente | 25 | 0 | 0 | 1 |
| mancha | 6 | 22 | 1 | 1 |
| risco | 0 | 11 | 16 | 0 |

## Abordagem: ml

IoU médio (localização): 0.853

| classe | precisão | recall | F1 | TP | FP | FN |
|---|---|---|---|---|---|---|
| furo_ausente | 0.962 | 0.962 | 0.962 | 25 | 1 | 1 |
| mancha | 0.683 | 0.933 | 0.789 | 28 | 13 | 2 |
| risco | 0.735 | 0.926 | 0.820 | 25 | 9 | 2 |

Matriz de confusão (linha = tipo real, coluna = tipo predito):

| real \ predito | furo_ausente | mancha | risco | nao_detectado |
|---|---|---|---|---|
| furo_ausente | 25 | 0 | 0 | 1 |
| mancha | 1 | 28 | 0 | 1 |
| risco | 0 | 2 | 25 | 0 |
