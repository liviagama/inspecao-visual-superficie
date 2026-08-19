"""Desenho do overlay de detecção: contorno, tipo e área sobre a imagem original."""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_COR_POR_TIPO = {
    "risco": (0, 165, 255),       # laranja
    "mancha": (0, 0, 255),        # vermelho
    "furo_ausente": (255, 0, 0),  # azul
}
_COR_PADRAO = (0, 255, 0)

_FONTE = cv2.FONT_HERSHEY_SIMPLEX
_ESCALA_FONTE = 0.6
_ESPESSURA_TEXTO = 2
_ESPESSURA_CONTORNO = 3
_PADDING_ROTULO = 5
_COR_FUNDO_ROTULO = (20, 20, 20)  # quase preto, pra contrastar com qualquer cor de classe


def _retangulos_sobrepostos(a: tuple, b: tuple) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def _desenhar_rotulo(imagem: np.ndarray, texto: str, x: int, y: int, cor: tuple, ocupados: list[tuple]) -> None:
    """Desenha `texto` com fundo preenchido perto de (x, y) e registra a caixa em `ocupados`.

    (x, y) é a posição preferida da linha de base do texto. A caixa é
    reposicionada pra não ultrapassar a borda superior nem lateral da
    imagem, e deslocada verticalmente se colidir com um rótulo já desenhado.
    """
    altura_img, largura_img = imagem.shape[:2]
    (tw, th), baseline = cv2.getTextSize(texto, _FONTE, _ESCALA_FONTE, _ESPESSURA_TEXTO)

    x1 = max(x, _PADDING_ROTULO)
    x1 = min(x1, max(largura_img - tw - 2 * _PADDING_ROTULO, _PADDING_ROTULO))
    y1 = max(y, th + _PADDING_ROTULO)

    caixa = (x1 - _PADDING_ROTULO, y1 - th - _PADDING_ROTULO, x1 + tw + _PADDING_ROTULO, y1 + baseline + _PADDING_ROTULO)
    while any(_retangulos_sobrepostos(caixa, outra) for outra in ocupados):
        deslocamento = (caixa[3] - caixa[1]) + 2
        y1 += deslocamento
        caixa = (caixa[0], caixa[1] + deslocamento, caixa[2], caixa[3] + deslocamento)

    cv2.rectangle(imagem, (caixa[0], caixa[1]), (caixa[2], caixa[3]), _COR_FUNDO_ROTULO, -1)
    cv2.putText(imagem, texto, (x1, y1), _FONTE, _ESCALA_FONTE, cor, _ESPESSURA_TEXTO, cv2.LINE_AA)
    ocupados.append(caixa)


def desenhar_overlay(imagem: np.ndarray, deteccoes: list[dict]) -> np.ndarray:
    """deteccoes: lista de {"tipo": str, "bbox": [x,y,w,h], "area": float, "contorno": ndarray opcional}"""
    saida = imagem.copy()
    ocupados: list[tuple] = []
    for det in deteccoes:
        cor = _COR_POR_TIPO.get(det["tipo"], _COR_PADRAO)
        x, y, w, h = det["bbox"]

        contorno = det.get("contorno")
        if contorno is not None:
            cv2.drawContours(saida, [contorno], -1, cor, _ESPESSURA_CONTORNO)
        else:
            cv2.rectangle(saida, (x, y), (x + w, y + h), cor, _ESPESSURA_CONTORNO)

        rotulo = f"{det['tipo']} ({int(det['area'])}px)"
        _desenhar_rotulo(saida, rotulo, x, y - 10, cor, ocupados)

    return saida
