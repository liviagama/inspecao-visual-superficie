"""Classificação dos candidatos: abordagem por regras e baseline de machine learning."""

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from inspecao.avaliacao import calcular_iou
from inspecao.features import Candidato

logger = logging.getLogger(__name__)

COLUNAS_FEATURES = ["area", "perimetro", "aspecto", "circularidade", "solidez", "extensao"]


@dataclass
class ParametrosRegras:
    circularidade_furo: float = 0.65
    solidez_furo: float = 0.88
    # rebites reais têm diâmetro limitado; acima disso não é furo de rebite,
    # é mancha redonda (calibrado nos dados sintéticos: furo_ausente nunca
    # passa de ~620px de área, enquanto manchas circulares grandes chegam a
    # passar de 1500px)
    area_maxima_furo: float = 700.0
    aspecto_risco: float = 1.8


def classificar_regras(candidato: Candidato, params: ParametrosRegras = ParametrosRegras()) -> str:
    """Classifica um candidato pelas características geométricas, sem aprendizado.

    Ordem importa: primeiro descarta furo (forma mais restritiva: precisa ser
    redondo, sólido E pequeno), depois risco (alongado), o resto vira mancha.
    Riscos quase na diagonal (~45°) têm bbox quase quadrado e aspecto perto de
    1, então caem em "mancha" por essa regra — é uma limitação conhecida, ver
    comparação com o classificador de ML no README.
    """
    if (
        candidato.circularidade >= params.circularidade_furo
        and candidato.solidez >= params.solidez_furo
        and candidato.area <= params.area_maxima_furo
    ):
        return "furo_ausente"
    if candidato.aspecto >= params.aspecto_risco:
        return "risco"
    return "mancha"


def candidato_para_vetor(candidato: Candidato) -> list[float]:
    return [getattr(candidato, nome) for nome in COLUNAS_FEATURES]


def construir_dataset_treino(
    candidatos_por_imagem: list[list[Candidato]],
    ground_truth_por_imagem: list[list[dict]],
    iou_min: float = 0.3,
) -> tuple[np.ndarray, list[str]]:
    """Rotula cada candidato com o tipo do GT mais próximo por IoU.

    Candidatos sem par no ground truth (ruído de segmentação) são
    descartados do treino: o classificador aprende a distinguir tipo de
    defeito, não a decidir se é defeito, papel que já é da segmentação.
    """
    X, y = [], []
    for candidatos, ground_truth in zip(candidatos_por_imagem, ground_truth_por_imagem):
        gt_usado = [False] * len(ground_truth)
        for candidato in candidatos:
            melhor_iou, melhor_idx = 0.0, -1
            for idx, gt in enumerate(ground_truth):
                if gt_usado[idx]:
                    continue
                iou = calcular_iou(candidato.bbox, tuple(gt["bbox"]))
                if iou > melhor_iou:
                    melhor_iou, melhor_idx = iou, idx
            if melhor_iou >= iou_min:
                gt_usado[melhor_idx] = True
                X.append(candidato_para_vetor(candidato))
                y.append(ground_truth[melhor_idx]["tipo"])

    logger.info("Dataset de treino construído: %d exemplos rotulados", len(y))
    return np.array(X), y


def treinar_modelo(X: np.ndarray, y: list[str], tipo_modelo: str = "arvore"):
    if tipo_modelo == "regressao_logistica":
        # area/perímetro ficam na casa das centenas e circularidade/solidez
        # entre 0 e 1; sem escalar, a regressão logística não convergia
        # (STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT mesmo com max_iter=1000)
        classificador = LogisticRegression(max_iter=1000)
        modelo = Pipeline([("escala", StandardScaler()), ("clf", classificador)])
    elif tipo_modelo == "arvore":
        # profundidade limitada a 5: com árvore sem limite o modelo decorava
        # o dataset sintético (treino em 100%, mas piorava um pouco no
        # conjunto de teste separado) por causa do baixo número de exemplos
        modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        raise ValueError(f"tipo_modelo desconhecido: {tipo_modelo}")

    modelo.fit(X, y)
    return modelo


def classificar_ml(modelo, candidato: Candidato) -> str:
    vetor = np.array(candidato_para_vetor(candidato)).reshape(1, -1)
    return modelo.predict(vetor)[0]


def salvar_modelo(modelo, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "wb") as f:
        pickle.dump(modelo, f)


def carregar_modelo(caminho: Path):
    with open(caminho, "rb") as f:
        return pickle.load(f)
