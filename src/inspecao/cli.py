"""Interface de linha de comando do pipeline de inspeção visual de superfície."""

import argparse
import glob
import json
import logging
import time
from pathlib import Path

import cv2
import yaml

from inspecao import avaliacao, classificador, dataset, features, overlay, preproc, relatorio, segmentacao

logger = logging.getLogger("inspecao")

EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".bmp")


def configurar_logging(nivel: str) -> None:
    # formato enxuto de propósito: os subcomandos de avaliação e benchmark
    # imprimem tabelas de resultado via logger.info, e prefixo de timestamp
    # atrapalharia quem for copiar esses números pra outro lugar
    logging.basicConfig(level=getattr(logging, nivel.upper()), format="%(message)s")


def carregar_config(caminho: Path) -> dict:
    with open(caminho, encoding="utf-8") as f:
        return yaml.safe_load(f)


def listar_imagens(entrada: str, arquivo_lote: str | None) -> list[Path]:
    if arquivo_lote:
        with open(arquivo_lote, encoding="utf-8") as f:
            return [Path(linha.strip()) for linha in f if linha.strip()]

    caminho = Path(entrada)
    if caminho.is_dir():
        arquivos = []
        for ext in EXTENSOES_IMAGEM:
            arquivos.extend(sorted(caminho.glob(f"*{ext}")))
        return arquivos
    return [caminho]


def processar_imagem(imagem, cfg: dict) -> tuple[list, dict]:
    tempos = {}

    t0 = time.perf_counter()
    pre = preproc.preprocessar(
        imagem,
        cfg["preproc"]["denoise_kernel"],
        cfg["preproc"]["clahe_clip_limit"],
        cfg["preproc"]["clahe_tile_grid"],
    )
    tempos["preproc"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    mask = segmentacao.segmentar(pre, **cfg["segmentacao"])
    tempos["segmentacao"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    candidatos = features.extrair_candidatos(mask, cfg["features"]["area_minima"])
    tempos["features"] = time.perf_counter() - t0

    return candidatos, tempos


def classificar_candidatos(candidatos: list, modo: str, modelo, cfg: dict) -> list[dict]:
    deteccoes = []
    params_regras = classificador.ParametrosRegras(**cfg["classificador_regras"])
    for c in candidatos:
        if modo == "ml":
            tipo = classificador.classificar_ml(modelo, c)
        else:
            tipo = classificador.classificar_regras(c, params_regras)
        deteccoes.append({"tipo": tipo, "bbox": list(c.bbox), "area": c.area, "contorno": c.contorno})
    return deteccoes


def cmd_gerar_dataset(args, cfg: dict) -> None:
    cfg_ds = cfg["dataset"]
    dataset.gerar_dataset(
        saida_imagens=Path(args.saida_imagens),
        saida_ground_truth=Path(args.saida_gt),
        n_imagens=args.n_imagens or cfg_ds["n_imagens"],
        largura=args.largura or cfg_ds["largura"],
        altura=args.altura or cfg_ds["altura"],
        defeitos_por_imagem=tuple(cfg_ds["defeitos_por_imagem"]),
        seed=args.seed if args.seed is not None else cfg_ds["seed"],
    )


def cmd_treinar_ml(args, cfg: dict) -> None:
    arquivos_gt = sorted(glob.glob(f"{args.dataset_gt}/*.json"))
    if not arquivos_gt:
        raise FileNotFoundError(f"Nenhum ground truth encontrado em {args.dataset_gt}")

    candidatos_por_imagem, gt_por_imagem = [], []
    for caminho_gt in arquivos_gt:
        with open(caminho_gt, encoding="utf-8") as f:
            registro = json.load(f)
        imagem = cv2.imread(str(Path(args.dataset_imagens) / registro["imagem"]))
        candidatos, _ = processar_imagem(imagem, cfg)
        candidatos_por_imagem.append(candidatos)
        gt_por_imagem.append(registro["defeitos"])

    X, y = classificador.construir_dataset_treino(candidatos_por_imagem, gt_por_imagem, cfg["avaliacao"]["iou_min_match"])
    modelo = classificador.treinar_modelo(X, y, args.tipo_modelo)
    classificador.salvar_modelo(modelo, Path(args.saida_modelo))
    logger.info("Modelo treinado (%s) com %d exemplos, salvo em %s", args.tipo_modelo, len(y), args.saida_modelo)


def cmd_inspecionar(args, cfg: dict) -> None:
    imagens = listar_imagens(args.entrada, args.lote)
    modelo = classificador.carregar_modelo(Path(args.modelo_ml)) if args.classificador == "ml" else None

    saida_overlays = Path(args.saida_overlays)
    saida_overlays.mkdir(parents=True, exist_ok=True)
    resultados_por_imagem = {}

    for caminho_imagem in imagens:
        imagem = cv2.imread(str(caminho_imagem))
        if imagem is None:
            logger.warning("Não foi possível ler %s, pulando", caminho_imagem)
            continue

        candidatos, _ = processar_imagem(imagem, cfg)
        deteccoes = classificar_candidatos(candidatos, args.classificador, modelo, cfg)
        resultados_por_imagem[caminho_imagem.name] = deteccoes

        saida_com_overlay = overlay.desenhar_overlay(imagem, deteccoes)
        cv2.imwrite(str(saida_overlays / caminho_imagem.name), saida_com_overlay)
        logger.info("%s: %d defeito(s) encontrado(s)", caminho_imagem.name, len(deteccoes))

    linhas = relatorio.montar_linhas_relatorio(
        resultados_por_imagem,
        cfg["relatorio"]["area_severidade_media"],
        cfg["relatorio"]["area_severidade_alta"],
        cfg["relatorio"]["limite_area_reprovacao"],
    )
    saida_lote = Path(args.saida_lote)
    relatorio.salvar_csv(linhas, saida_lote / "relatorio.csv")
    relatorio.salvar_markdown(linhas, saida_lote / "relatorio.md")


def cmd_avaliar(args, cfg: dict) -> None:
    arquivos_gt = sorted(glob.glob(f"{args.dataset_gt}/*.json"))
    if not arquivos_gt:
        raise FileNotFoundError(f"Nenhum ground truth encontrado em {args.dataset_gt}")

    modelo_ml = classificador.carregar_modelo(Path(args.modelo_ml)) if args.modelo_ml else None
    params_regras = classificador.ParametrosRegras(**cfg["classificador_regras"])

    det_regras_total, det_ml_total, gt_total = [], [], []
    for caminho_gt in arquivos_gt:
        with open(caminho_gt, encoding="utf-8") as f:
            registro = json.load(f)
        imagem = cv2.imread(str(Path(args.dataset_imagens) / registro["imagem"]))
        candidatos, _ = processar_imagem(imagem, cfg)

        for c in candidatos:
            det_regras_total.append({"tipo": classificador.classificar_regras(c, params_regras), "bbox": list(c.bbox)})
            if modelo_ml is not None:
                det_ml_total.append({"tipo": classificador.classificar_ml(modelo_ml, c), "bbox": list(c.bbox)})
        gt_total.extend(registro["defeitos"])

    iou_min = cfg["avaliacao"]["iou_min_match"]
    resultados = {"regras": avaliacao.calcular_metricas(det_regras_total, gt_total, iou_min)}
    if modelo_ml is not None:
        resultados["ml"] = avaliacao.calcular_metricas(det_ml_total, gt_total, iou_min)

    avaliacao.gerar_relatorio_avaliacao(resultados, Path(args.saida))

    for nome_abordagem, metricas in resultados.items():
        logger.info("=== %s ===", nome_abordagem)
        logger.info("IoU médio: %.3f", metricas["iou_medio"])
        for tipo, m in metricas["por_classe"].items():
            logger.info(
                "  %-14s precisao=%.3f recall=%.3f f1=%.3f (tp=%d fp=%d fn=%d)",
                tipo, m["precisao"], m["recall"], m["f1"], m["tp"], m["fp"], m["fn"],
            )


def cmd_benchmark(args, cfg: dict) -> None:
    imagens = listar_imagens(args.entrada, None)[: args.n_imagens]
    if not imagens:
        raise FileNotFoundError(f"Nenhuma imagem encontrada em {args.entrada}")

    acumulado = {"preproc": 0.0, "segmentacao": 0.0, "features": 0.0}
    tempo_total = 0.0
    for caminho_imagem in imagens:
        imagem = cv2.imread(str(caminho_imagem))
        t0 = time.perf_counter()
        _, tempos = processar_imagem(imagem, cfg)
        tempo_total += time.perf_counter() - t0
        for etapa, duracao in tempos.items():
            acumulado[etapa] += duracao

    n = len(imagens)
    logger.info("=== Benchmark (%d imagens) ===", n)
    for etapa, duracao in acumulado.items():
        logger.info("  %-12s %.2f ms/imagem", etapa, (duracao / n) * 1000)
    logger.info("  %-12s %.2f ms/imagem", "total", (tempo_total / n) * 1000)
    logger.info("  throughput aproximado: %.1f imagens/s", n / tempo_total)


def montar_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspecao", description="Pipeline de inspeção visual de superfície")
    parser.add_argument("--config", default="config.yaml", help="Caminho do arquivo de configuração YAML")
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="comando", required=True)

    p_ds = subparsers.add_parser("gerar-dataset", help="Gera imagens sintéticas com ground truth")
    p_ds.add_argument("--saida-imagens", default="dados/sinteticas")
    p_ds.add_argument("--saida-gt", default="dados/ground_truth")
    p_ds.add_argument("--n-imagens", type=int, default=None)
    p_ds.add_argument("--largura", type=int, default=None)
    p_ds.add_argument("--altura", type=int, default=None)
    p_ds.add_argument("--seed", type=int, default=None)
    p_ds.set_defaults(func=cmd_gerar_dataset)

    p_tr = subparsers.add_parser("treinar-ml", help="Treina o classificador de ML sobre um dataset com ground truth")
    p_tr.add_argument("--dataset-imagens", default="dados/sinteticas")
    p_tr.add_argument("--dataset-gt", default="dados/ground_truth")
    p_tr.add_argument("--tipo-modelo", choices=["arvore", "regressao_logistica"], default="arvore")
    p_tr.add_argument("--saida-modelo", default="dados/modelo_ml.pkl")
    p_tr.set_defaults(func=cmd_treinar_ml)

    p_insp = subparsers.add_parser("inspecionar", help="Roda o pipeline numa imagem, pasta ou lote")
    p_insp.add_argument("entrada", nargs="?", default=None, help="Imagem única ou pasta de imagens")
    p_insp.add_argument("--lote", default=None, help="Arquivo texto com um caminho de imagem por linha")
    p_insp.add_argument("--classificador", choices=["regras", "ml"], default="regras")
    p_insp.add_argument("--modelo-ml", default="dados/modelo_ml.pkl")
    p_insp.add_argument("--saida-overlays", default="saida/overlays")
    p_insp.add_argument("--saida-lote", default="saida/lote")
    p_insp.set_defaults(func=cmd_inspecionar)

    p_av = subparsers.add_parser("avaliar", help="Avalia regras e ML contra o ground truth sintético")
    p_av.add_argument("--dataset-imagens", default="dados/sinteticas")
    p_av.add_argument("--dataset-gt", default="dados/ground_truth")
    p_av.add_argument("--modelo-ml", default=None, help="Caminho do modelo treinado; se omitido, avalia só as regras")
    p_av.add_argument("--saida", default="avaliacao/resultados.md")
    p_av.set_defaults(func=cmd_avaliar)

    p_bm = subparsers.add_parser("benchmark", help="Mede tempo por estágio do pipeline")
    p_bm.add_argument("entrada", default="dados/sinteticas", nargs="?")
    p_bm.add_argument("--n-imagens", type=int, default=30)
    p_bm.set_defaults(func=cmd_benchmark)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = montar_parser()
    args = parser.parse_args(argv)
    configurar_logging(args.log_level)
    cfg = carregar_config(Path(args.config))
    args.func(args, cfg)


if __name__ == "__main__":
    main()
