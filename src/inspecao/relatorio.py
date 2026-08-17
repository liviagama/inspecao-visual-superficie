"""Relatório de inspeção por lote: severidade por área e veredito aprovado/reprovado."""

import csv
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def classificar_severidade(area: float, limite_media: float, limite_alta: float) -> str:
    if area >= limite_alta:
        return "alta"
    if area >= limite_media:
        return "media"
    return "baixa"


def montar_linhas_relatorio(
    resultados_por_imagem: dict[str, list[dict]],
    limite_media: float,
    limite_alta: float,
    limite_area_reprovacao: float,
) -> list[dict]:
    """resultados_por_imagem: {nome_imagem: [{"tipo":..., "area":...}, ...]}"""
    linhas = []
    for imagem, deteccoes in resultados_por_imagem.items():
        if not deteccoes:
            linhas.append(
                {
                    "imagem": imagem,
                    "tipo": "-",
                    "area_px": 0,
                    "severidade": "-",
                    "veredito": "aprovado",
                }
            )
            continue

        veredito = "reprovado" if any(d["area"] >= limite_area_reprovacao for d in deteccoes) else "aprovado"
        for d in deteccoes:
            linhas.append(
                {
                    "imagem": imagem,
                    "tipo": d["tipo"],
                    "area_px": round(d["area"], 1),
                    "severidade": classificar_severidade(d["area"], limite_media, limite_alta),
                    "veredito": veredito,
                }
            )
    return linhas


def salvar_csv(linhas: list[dict], caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["imagem", "tipo", "area_px", "severidade", "veredito"])
        escritor.writeheader()
        escritor.writerows(linhas)
    logger.info("Relatório CSV salvo em %s", caminho)


def _tabela_markdown(linhas: list[dict]) -> str:
    # pandas.to_markdown() precisaria do pacote tabulate, fora da lista de
    # dependências do projeto — mais simples montar a tabela na mão aqui
    colunas = ["imagem", "tipo", "area_px", "severidade", "veredito"]
    cabecalho = "| " + " | ".join(colunas) + " |"
    separador = "|" + "---|" * len(colunas)
    corpo = "\n".join("| " + " | ".join(str(linha[c]) for c in colunas) + " |" for linha in linhas)
    return "\n".join([cabecalho, separador, corpo])


def salvar_markdown(linhas: list[dict], caminho: Path) -> None:
    df = pd.DataFrame(linhas)
    veredito_por_imagem = df.groupby("imagem")["veredito"].first()
    n_aprovadas = (veredito_por_imagem == "aprovado").sum()
    n_reprovadas = (veredito_por_imagem == "reprovado").sum()

    partes = [
        "# Relatório de inspeção por lote",
        "",
        f"Imagens inspecionadas: {len(veredito_por_imagem)} "
        f"({n_aprovadas} aprovadas, {n_reprovadas} reprovadas)",
        "",
        _tabela_markdown(linhas),
        "",
    ]
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(partes))
    logger.info("Relatório Markdown salvo em %s", caminho)
