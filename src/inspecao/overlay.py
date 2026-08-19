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

# cores usadas só na ilustração de erro de avaliação (docs/img/erro_*.png),
# não no overlay normal de inspeção. Padronizadas pra não significar coisas
# diferentes em figuras diferentes:
COR_GT_NAO_DETECTADO = (255, 0, 255)     # magenta: defeito real que nenhuma detecção achou
COR_DETECCAO_INCORRETA = (255, 255, 0)   # ciano: detecção sem defeito real correspondente

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


def cor_da_classe(tipo: str) -> tuple:
    """Cor (BGR) usada pra uma classe de defeito no overlay normal — azul/laranja/vermelho."""
    return _COR_POR_TIPO.get(tipo, _COR_PADRAO)


def _linha_tracejada(imagem: np.ndarray, pt1: tuple, pt2: tuple, cor: tuple, espessura: int = 2, traco: int = 9, vao: int = 7) -> None:
    pt1 = np.array(pt1, dtype=float)
    pt2 = np.array(pt2, dtype=float)
    dist = np.linalg.norm(pt2 - pt1)
    if dist == 0:
        return
    direcao = (pt2 - pt1) / dist
    passo = traco + vao
    d = 0.0
    while d < dist:
        ini = pt1 + direcao * d
        fim = pt1 + direcao * min(d + traco, dist)
        cv2.line(imagem, tuple(ini.astype(int)), tuple(fim.astype(int)), cor, espessura, cv2.LINE_AA)
        d += passo


def desenhar_marcacao_erro(
    imagem: np.ndarray, bbox: tuple, cor: tuple, rotulo: str, ocupados: list[tuple] | None = None
) -> None:
    """Marca uma região de erro de avaliação (falso positivo ou falso negativo) com um
    retângulo tracejado e um rótulo curto em ASCII (sem acento — a fonte do cv2.putText
    não suporta acentuação).

    Usada só na geração das figuras de análise de erro da documentação
    (`docs/img/erro_falso_positivo.png`, `docs/img/erro_falso_negativo.png`); não faz
    parte do overlay de inspeção normal (`desenhar_overlay`).
    """
    x, y, w, h = bbox
    x1, y1, x2, y2 = x - 6, y - 6, x + w + 6, y + h + 6
    _linha_tracejada(imagem, (x1, y1), (x2, y1), cor)
    _linha_tracejada(imagem, (x2, y1), (x2, y2), cor)
    _linha_tracejada(imagem, (x2, y2), (x1, y2), cor)
    _linha_tracejada(imagem, (x1, y2), (x1, y1), cor)
    _desenhar_rotulo(imagem, rotulo, x1, y1 - 8, cor, ocupados if ocupados is not None else [])


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
