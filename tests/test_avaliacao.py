from inspecao.avaliacao import calcular_iou, calcular_metricas, casar_deteccoes_com_gt


def test_iou_caixas_identicas_e_um():
    assert calcular_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_caixas_sem_sobreposicao_e_zero():
    assert calcular_iou((0, 0, 10, 10), (100, 100, 10, 10)) == 0.0


def test_iou_sobreposicao_parcial():
    # bbox2 desloca metade do bbox1 em x: intersecao 5x10=50, uniao 150
    iou = calcular_iou((0, 0, 10, 10), (5, 0, 10, 10))
    assert abs(iou - 50 / 150) < 1e-9


def test_casar_deteccoes_e_guloso_sem_reuso():
    deteccoes = [{"tipo": "risco", "bbox": [0, 0, 10, 10]}]
    gt = [{"tipo": "risco", "bbox": [1, 1, 10, 10]}, {"tipo": "risco", "bbox": [50, 50, 10, 10]}]
    pares = casar_deteccoes_com_gt(deteccoes, gt, iou_min=0.3)

    assert len(pares) == 1
    assert pares[0][1] == 0  # casou com o gt mais próximo, não com o distante


def test_metricas_deteccao_perfeita():
    deteccoes = [{"tipo": "risco", "bbox": [0, 0, 10, 10]}]
    gt = [{"tipo": "risco", "bbox": [0, 0, 10, 10]}]
    metricas = calcular_metricas(deteccoes, gt)

    assert metricas["por_classe"]["risco"]["precisao"] == 1.0
    assert metricas["por_classe"]["risco"]["recall"] == 1.0
    assert metricas["iou_medio"] == 1.0


def test_metricas_falso_positivo_sem_gt_correspondente():
    deteccoes = [{"tipo": "mancha", "bbox": [0, 0, 10, 10]}]
    gt = []
    metricas = calcular_metricas(deteccoes, gt)

    assert metricas["por_classe"]["mancha"]["precisao"] == 0.0
    assert metricas["por_classe"]["mancha"]["fp"] == 1
    assert metricas["falsos_positivos_sem_gt"] == 1


def test_metricas_falso_negativo_sem_deteccao():
    deteccoes = []
    gt = [{"tipo": "furo_ausente", "bbox": [0, 0, 10, 10]}]
    metricas = calcular_metricas(deteccoes, gt)

    assert metricas["por_classe"]["furo_ausente"]["recall"] == 0.0
    assert metricas["por_classe"]["furo_ausente"]["fn"] == 1


def test_metricas_bbox_certo_mas_tipo_errado_conta_fp_e_fn():
    deteccoes = [{"tipo": "mancha", "bbox": [0, 0, 10, 10]}]
    gt = [{"tipo": "risco", "bbox": [0, 0, 10, 10]}]
    metricas = calcular_metricas(deteccoes, gt)

    assert metricas["por_classe"]["risco"]["fn"] == 1
    assert metricas["por_classe"]["mancha"]["fp"] == 1
    assert metricas["por_classe"]["risco"]["tp"] == 0
