"""Pré-processamento: escala de cinza, redução de ruído e normalização de iluminação."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def converter_para_cinza(imagem: np.ndarray) -> np.ndarray:
    if imagem.ndim == 2:
        return imagem
    return cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)


def reduzir_ruido(imagem_cinza: np.ndarray, kernel: int = 3) -> np.ndarray:
    # mediana em vez de gaussiana: preserva melhor a borda dos defeitos
    # (que é justamente o que a segmentação depois vai precisar) enquanto
    # ainda limpa o ruído tipo sal-e-pimenta do sensor
    return cv2.medianBlur(imagem_cinza, kernel)


def aplicar_clahe(imagem_cinza: np.ndarray, clip_limit: float = 2.0, tile_grid: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(imagem_cinza)


def preprocessar(
    imagem: np.ndarray,
    denoise_kernel: int = 3,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid: int = 8,
) -> np.ndarray:
    """Pipeline completo de pré-processamento, na ordem cinza -> denoise -> CLAHE.

    CLAHE entra depois do denoise e antes da segmentação: iluminação de chão de
    fábrica raramente é uniforme sobre a chapa toda (reflexo de lâmpada, sombra
    de suporte), e sem equalizar isso um threshold único acaba estourando numa
    ponta da imagem e sumindo defeito na outra. CLAHE local resolve isso sem
    amplificar o ruído que a mediana já tirou antes.
    """
    cinza = converter_para_cinza(imagem)
    sem_ruido = reduzir_ruido(cinza, denoise_kernel)
    equalizado = aplicar_clahe(sem_ruido, clahe_clip_limit, clahe_tile_grid)
    return equalizado
