"""Extração de candidatos a defeito e suas características geométricas."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# abaixo disso o contorno costuma ser ruído residual da segmentação, não
# defeito: em 60 imagens sintéticas de teste, contornos de ruído ficaram
# quase todos abaixo de 30px² (só 1 em 210 passou de 50px²) enquanto os
# contornos que batiam com o ground truth raramente ficavam abaixo de 40px².
# Com esse corte o número de candidatos por imagem caiu de ~6.7 pra ~3.2 sem
# perder recall (158/160 defeitos ainda batidos, os 2 que se perdem já eram
# fragmentos residuais de risco fino cortados pela abertura morfológica)
AREA_MINIMA_PADRAO = 40


@dataclass
class Candidato:
    contorno: np.ndarray
    bbox: tuple  # (x, y, w, h)
    area: float
    perimetro: float
    aspecto: float
    circularidade: float
    solidez: float
    extensao: float

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.bbox),
            "area": self.area,
            "perimetro": self.perimetro,
            "aspecto": self.aspecto,
            "circularidade": self.circularidade,
            "solidez": self.solidez,
            "extensao": self.extensao,
        }


def calcular_caracteristicas(contorno: np.ndarray) -> Candidato | None:
    area = cv2.contourArea(contorno)
    if area <= 0:
        return None

    perimetro = cv2.arcLength(contorno, closed=True)
    x, y, w, h = cv2.boundingRect(contorno)
    aspecto = max(w, h) / max(min(w, h), 1)

    circularidade = (4 * np.pi * area) / (perimetro**2) if perimetro > 0 else 0.0
    # circularidade pode passar de 1 por erro de discretização do contorno
    # em círculos pequenos (poligonal versus círculo perfeito); trava em 1
    # pra não distorcer o classificador por regras
    circularidade = min(circularidade, 1.0)

    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    solidez = area / area_hull if area_hull > 0 else 0.0

    extensao = area / (w * h) if w * h > 0 else 0.0

    return Candidato(
        contorno=contorno,
        bbox=(x, y, w, h),
        area=float(area),
        perimetro=float(perimetro),
        aspecto=float(aspecto),
        circularidade=float(circularidade),
        solidez=float(solidez),
        extensao=float(extensao),
    )


def extrair_candidatos(mascara: np.ndarray, area_minima: int = AREA_MINIMA_PADRAO) -> list[Candidato]:
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidatos = []
    for contorno in contornos:
        candidato = calcular_caracteristicas(contorno)
        if candidato is None or candidato.area < area_minima:
            continue
        candidatos.append(candidato)

    logger.debug("Extraídos %d candidatos (de %d contornos brutos)", len(candidatos), len(contornos))
    return candidatos
