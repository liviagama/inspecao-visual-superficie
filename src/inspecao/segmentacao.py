"""Segmentação: threshold adaptativo + Canny, combinados e limpos por morfologia."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def threshold_adaptativo(imagem_pre: np.ndarray, block_size: int = 25, c: int = 5) -> np.ndarray:
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(
        imagem_pre,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        c,
    )


def detectar_bordas(imagem_pre: np.ndarray, limiar_baixo: int = 40, limiar_alto: int = 120) -> np.ndarray:
    return cv2.Canny(imagem_pre, limiar_baixo, limiar_alto)


def segmentar(
    imagem_pre: np.ndarray,
    threshold_block_size: int = 35,
    threshold_c: int = 15,
    canny_limiar_baixo: int = 40,
    canny_limiar_alto: int = 120,
    kernel_abertura: int = 3,
    kernel_fechamento: int = 9,
) -> np.ndarray:
    """Combina threshold adaptativo e Canny numa máscara binária limpa.

    O threshold sozinho pega a região inteira do defeito (bom pra área/forma),
    mas o Canny sozinho pega só o contorno, que costuma vir fragmentado em
    riscos finos. Juntando os dois com OR, a região do threshold garante
    massa contínua e o Canny reforça borda em casos onde o defeito é sutil
    demais pro threshold sozinho detectar.

    block_size=35 e C=15 vieram de um teste com 44 imagens sintéticas
    (sementes 42 e 123) comparando o centro de cada contorno detectado contra
    o bbox do ground truth: com C=5 (valor "de manual") a textura da chapa
    virava ruído de fundo e a máscara saía com mais de 20% de pixels brancos;
    C=15 derrubou pra ~6 candidatos por imagem sem perder nenhum defeito real
    (104/104 recall no teste). kernel_fechamento=9 é maior que o usual porque
    mancha e furo_ausente, quando do tamanho do block_size do threshold, saem
    como anel oco em vez de disco cheio — um fechamento pequeno não fecha esse
    buraco, mas como área/perímetro são calculados sobre o contorno externo
    (ver features.py), o anel ainda descreve a forma corretamente contanto que
    feche num laço só, o que passou a acontecer de forma consistente nesse
    kernel.
    """
    mascara_threshold = threshold_adaptativo(imagem_pre, threshold_block_size, threshold_c)
    mascara_canny = detectar_bordas(imagem_pre, canny_limiar_baixo, canny_limiar_alto)
    combinada = cv2.bitwise_or(mascara_threshold, mascara_canny)

    # abertura primeiro remove pontinhos isolados de ruído que sobraram do
    # threshold (testado com kernel 3x3: kernel 5x5 já começava a apagar
    # risco fino de 1px de espessura, então ficou nesse tamanho)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_abertura, kernel_abertura))
    aberta = cv2.morphologyEx(combinada, cv2.MORPH_OPEN, kernel_open)

    # fechamento depois reconecta a borda do Canny com a massa do threshold,
    # que costuma sair com pequenas quebras nas bordas de riscos e manchas
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_fechamento, kernel_fechamento))
    fechada = cv2.morphologyEx(aberta, cv2.MORPH_CLOSE, kernel_close)

    return fechada
