"""Avaliação: IoU, casamento com ground truth, precisão/recall/F1 e matriz de confusão."""

import logging
from collections import Counter

logger = logging.getLogger(__name__)


def calcular_iou(bbox1: tuple, bbox2: tuple) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2

    xi1, yi1 = max(x1, x2), max(y1, y2)
    xi2, yi2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    largura_inter = max(0, xi2 - xi1)
    altura_inter = max(0, yi2 - yi1)
    intersecao = largura_inter * altura_inter

    uniao = w1 * h1 + w2 * h2 - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


def casar_deteccoes_com_gt(deteccoes: list[dict], ground_truth: list[dict], iou_min: float = 0.3) -> list[tuple]:
    """Casamento guloso por IoU decrescente, sem reaproveitar detecção nem GT.

    Não filtra por tipo aqui de propósito: uma detecção com bbox certo mas
    tipo errado ainda deve "consumir" aquele par pra avaliação de
    classificação, em vez de aparecer como falso positivo E o GT como falso
    negativo separado.
    """
    candidatas = []
    for i, det in enumerate(deteccoes):
        for j, gt in enumerate(ground_truth):
            iou = calcular_iou(det["bbox"], gt["bbox"])
            if iou >= iou_min:
                candidatas.append((iou, i, j))
    candidatas.sort(key=lambda t: t[0], reverse=True)

    det_usada = [False] * len(deteccoes)
    gt_usada = [False] * len(ground_truth)
    pares = []
    for iou, i, j in candidatas:
        if not det_usada[i] and not gt_usada[j]:
            det_usada[i] = True
            gt_usada[j] = True
            pares.append((i, j, iou))
    return pares


def calcular_metricas(deteccoes: list[dict], ground_truth: list[dict], iou_min: float = 0.3) -> dict:
    """Precisão/recall/F1 por classe, matriz de confusão e IoU médio.

    deteccoes e ground_truth são listas de {"tipo": str, "bbox": [x,y,w,h]}.
    """
    pares = casar_deteccoes_com_gt(deteccoes, ground_truth, iou_min)
    det_usada = {i for i, _, _ in pares}
    gt_usada = {j for _, j, _ in pares}

    tipos = sorted({d["tipo"] for d in deteccoes} | {g["tipo"] for g in ground_truth})
    matriz = {real: {predito: 0 for predito in tipos + ["nao_detectado"]} for real in tipos}

    for i, j, _ in pares:
        real = ground_truth[j]["tipo"]
        predito = deteccoes[i]["tipo"]
        matriz[real][predito] += 1

    for j, gt in enumerate(ground_truth):
        if j not in gt_usada:
            matriz[gt["tipo"]]["nao_detectado"] += 1

    fp_sem_gt = Counter(deteccoes[i]["tipo"] for i in range(len(deteccoes)) if i not in det_usada)

    metricas_por_classe = {}
    for tipo in tipos:
        tp = matriz[tipo][tipo]
        fn = sum(qtd for predito, qtd in matriz[tipo].items() if predito != tipo)
        fp_classificacao_errada = sum(matriz[outro].get(tipo, 0) for outro in tipos if outro != tipo)
        fp = fp_classificacao_errada + fp_sem_gt.get(tipo, 0)

        precisao = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0.0
        metricas_por_classe[tipo] = {
            "precisao": precisao,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    iou_medio = sum(iou for _, _, iou in pares) / len(pares) if pares else 0.0

    return {
        "por_classe": metricas_por_classe,
        "matriz_confusao": matriz,
        "iou_medio": iou_medio,
        "falsos_positivos_sem_gt": sum(fp_sem_gt.values()),
    }


def formatar_matriz_confusao_md(matriz: dict) -> str:
    tipos_reais = list(matriz.keys())
    colunas = list(next(iter(matriz.values())).keys())

    linhas = ["| real \\ predito | " + " | ".join(colunas) + " |"]
    linhas.append("|" + "---|" * (len(colunas) + 1))
    for real in tipos_reais:
        valores = " | ".join(str(matriz[real][c]) for c in colunas)
        linhas.append(f"| {real} | {valores} |")
    return "\n".join(linhas)


def gerar_relatorio_avaliacao(resultados: dict, caminho_saida) -> None:
    """resultados: {"regras": {...saida de calcular_metricas...}, "ml": {...}}"""
    linhas = ["# Avaliação da detecção de defeitos", ""]

    for nome_abordagem, metricas in resultados.items():
        linhas.append(f"## Abordagem: {nome_abordagem}")
        linhas.append("")
        linhas.append(f"IoU médio (localização): {metricas['iou_medio']:.3f}")
        linhas.append("")
        linhas.append("| classe | precisão | recall | F1 | TP | FP | FN |")
        linhas.append("|---|---|---|---|---|---|---|")
        for tipo, m in metricas["por_classe"].items():
            linhas.append(
                f"| {tipo} | {m['precisao']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                f"| {m['tp']} | {m['fp']} | {m['fn']} |"
            )
        linhas.append("")
        linhas.append("Matriz de confusão (linha = tipo real, coluna = tipo predito):")
        linhas.append("")
        linhas.append(formatar_matriz_confusao_md(metricas["matriz_confusao"]))
        linhas.append("")

    with open(caminho_saida, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    logger.info("Relatório de avaliação salvo em %s", caminho_saida)
