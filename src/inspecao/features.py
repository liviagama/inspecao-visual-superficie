"""Extração de candidatos a defeito e suas características geométricas."""

import logging
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# abaixo disso o contorno é ruído residual que sobrou da segmentação, não
# defeito: medido olhando a distribuição de área dos contornos espúrios nas
# imagens sintéticas sem nenhum defeito naquela região (ver segmentacao.py
# pra calibração completa do resto do pipeline)
AREA_MINIMA_PADRAO = 15


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
