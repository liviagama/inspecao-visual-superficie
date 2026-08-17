"""Gerador de imagens sintéticas de chapa metálica com defeitos e ground truth."""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TIPOS_DEFEITO = ("risco", "mancha", "furo_ausente")


@dataclass
class DefeitoGT:
    tipo: str
    bbox: list  # [x, y, w, h]
    area: int


def gerar_textura_base(largura: int, altura: int, rng: np.random.Generator) -> np.ndarray:
    """Cria uma chapa metálica cinza-azulada com veios de escovação e ruído de sensor."""
    base = rng.normal(150, 8, (altura, largura)).astype(np.float32)

    # veios de escovação: listras horizontais tênues, imitando o acabamento
    # de laminação da chapa. Sem isso a textura fica "plástica" demais e o
    # threshold adaptativo não tem nada de realista pra lidar.
    n_veios = altura // 15
    for _ in range(n_veios):
        y = rng.integers(0, altura)
        base[y, :] += rng.normal(0, 4)

    base = cv2.GaussianBlur(base, (0, 0), sigmaX=1.2)
    ruido_sensor = rng.normal(0, 3.5, (altura, largura))
    base += ruido_sensor

    cinza = np.clip(base, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(cinza, cv2.COLOR_GRAY2BGR)
    # leve tom azulado típico de alumínio/aço sob luz fria de inspeção
    bgr[:, :, 0] = np.clip(bgr[:, :, 0].astype(np.int16) + 6, 0, 255).astype(np.uint8)
    return bgr


def _desenhar_risco(mascara: np.ndarray, rng: np.random.Generator) -> None:
    altura, largura = mascara.shape
    x1, y1 = rng.integers(0, largura), rng.integers(0, altura)
    comprimento = rng.integers(30, 140)
    angulo = rng.uniform(0, 2 * np.pi)
    x2 = int(np.clip(x1 + comprimento * np.cos(angulo), 0, largura - 1))
    y2 = int(np.clip(y1 + comprimento * np.sin(angulo), 0, altura - 1))
    espessura = int(rng.integers(1, 3))
    cv2.line(mascara, (x1, y1), (x2, y2), 255, espessura, lineType=cv2.LINE_AA)


def _desenhar_mancha(mascara: np.ndarray, rng: np.random.Generator) -> None:
    altura, largura = mascara.shape
    cx, cy = rng.integers(20, largura - 20), rng.integers(20, altura - 20)
    raio_base = rng.integers(10, 30)
    # blob irregular: várias manchas circulares deslocadas em volta de um
    # centro, o que dá borda serrilhada sem precisar de nenhuma lib extra
    n_lobulos = rng.integers(5, 10)
    for _ in range(n_lobulos):
        offset_ang = rng.uniform(0, 2 * np.pi)
        offset_dist = rng.uniform(0, raio_base * 0.6)
        lx = int(cx + offset_dist * np.cos(offset_ang))
        ly = int(cy + offset_dist * np.sin(offset_ang))
        raio_lobulo = int(raio_base * rng.uniform(0.4, 0.9))
        cv2.circle(mascara, (lx, ly), raio_lobulo, 255, -1)


def _desenhar_furo_ausente(mascara: np.ndarray, rng: np.random.Generator) -> None:
    altura, largura = mascara.shape
    cx, cy = rng.integers(15, largura - 15), rng.integers(15, altura - 15)
    raio = rng.integers(6, 14)
    cv2.circle(mascara, (int(cx), int(cy)), int(raio), 255, -1)


_DESENHISTAS = {
    "risco": _desenhar_risco,
    "mancha": _desenhar_mancha,
    "furo_ausente": _desenhar_furo_ausente,
}

# intensidade (delta de BGR aplicado sobre a base) por tipo de defeito,
# calibrada visualmente pra parecer plausível numa chapa real
_COR_DEFEITO = {
    "risco": np.array([-45, -45, -45]),
    "mancha": np.array([-70, -40, 20]),  # tom de ferrugem: azul cai, vermelho sobe
    "furo_ausente": np.array([-120, -120, -120]),
}


def inserir_defeito(imagem: np.ndarray, tipo: str, rng: np.random.Generator) -> DefeitoGT | None:
    altura, largura = imagem.shape[:2]
    mascara = np.zeros((altura, largura), dtype=np.uint8)
    _DESENHISTAS[tipo](mascara, rng)

    if tipo != "risco":
        mascara = cv2.GaussianBlur(mascara, (5, 5), 0)
        _, mascara = cv2.threshold(mascara, 60, 255, cv2.THRESH_BINARY)

    area = int(cv2.countNonZero(mascara))
    if area == 0:
        return None

    x, y, w, h = cv2.boundingRect(mascara)
    delta = _COR_DEFEITO[tipo]
    regiao = mascara > 0
    imagem[regiao] = np.clip(imagem[regiao].astype(np.int16) + delta, 0, 255).astype(np.uint8)

    if tipo == "risco":
        imagem_desfoque = cv2.GaussianBlur(imagem, (3, 3), 0)
        imagem[regiao] = imagem_desfoque[regiao]

    return DefeitoGT(tipo=tipo, bbox=[int(x), int(y), int(w), int(h)], area=area)


def gerar_imagem(
    largura: int,
    altura: int,
    n_defeitos_min: int,
    n_defeitos_max: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[DefeitoGT]]:
    imagem = gerar_textura_base(largura, altura, rng)
    n_defeitos = int(rng.integers(n_defeitos_min, n_defeitos_max + 1))
    ground_truth = []
    # TODO: não checa sobreposição entre defeitos. Se dois caírem na mesma
    # região, a área gravada no ground truth do primeiro pode não bater mais
    # com os pixels finais depois que o segundo é desenhado por cima. Com
    # 1-4 defeitos numa chapa de 640x480 isso é raro o suficiente pra não ter
    # aparecido nos testes, mas seria o primeiro ajuste antes de aumentar a
    # densidade de defeitos por imagem.
    for _ in range(n_defeitos):
        tipo = rng.choice(TIPOS_DEFEITO)
        defeito = inserir_defeito(imagem, tipo, rng)
        if defeito is not None:
            ground_truth.append(defeito)
    return imagem, ground_truth


def gerar_dataset(
    saida_imagens: Path,
    saida_ground_truth: Path,
    n_imagens: int,
    largura: int,
    altura: int,
    defeitos_por_imagem: tuple[int, int],
    seed: int,
) -> None:
    saida_imagens.mkdir(parents=True, exist_ok=True)
    saida_ground_truth.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    for i in range(n_imagens):
        nome = f"img_{i:04d}"
        imagem, gt = gerar_imagem(largura, altura, defeitos_por_imagem[0], defeitos_por_imagem[1], rng)
        caminho_imagem = saida_imagens / f"{nome}.png"
        cv2.imwrite(str(caminho_imagem), imagem)

        registro = {"imagem": f"{nome}.png", "defeitos": [asdict(d) for d in gt]}
        caminho_json = saida_ground_truth / f"{nome}.json"
        with open(caminho_json, "w", encoding="utf-8") as f:
            json.dump(registro, f, ensure_ascii=False, indent=2)

    logger.info("Dataset gerado: %d imagens em %s", n_imagens, saida_imagens)
