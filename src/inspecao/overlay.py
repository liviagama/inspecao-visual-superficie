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


def desenhar_overlay(imagem: np.ndarray, deteccoes: list[dict]) -> np.ndarray:
    """deteccoes: lista de {"tipo": str, "bbox": [x,y,w,h], "area": float, "contorno": ndarray opcional}"""
    saida = imagem.copy()
    for det in deteccoes:
        cor = _COR_POR_TIPO.get(det["tipo"], _COR_PADRAO)
        x, y, w, h = det["bbox"]

        contorno = det.get("contorno")
        if contorno is not None:
            cv2.drawContours(saida, [contorno], -1, cor, 2)
        else:
            cv2.rectangle(saida, (x, y), (x + w, y + h), cor, 2)

        rotulo = f"{det['tipo']} ({int(det['area'])}px)"
        origem_texto = (x, max(y - 6, 12))
        cv2.putText(saida, rotulo, origem_texto, cv2.FONT_HERSHEY_SIMPLEX, 0.45, cor, 1, cv2.LINE_AA)

    return saida
