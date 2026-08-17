import cv2
import numpy as np

from inspecao.features import calcular_caracteristicas, extrair_candidatos


def _mascara_com_quadrado(lado: int = 40) -> np.ndarray:
    mascara = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mascara, (30, 30), (30 + lado, 30 + lado), 255, -1)
    return mascara


def _mascara_com_circulo(raio: int = 20) -> np.ndarray:
    mascara = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mascara, (50, 50), raio, 255, -1)
    return mascara


def test_quadrado_tem_aspecto_um_e_extensao_alta():
    mascara = _mascara_com_quadrado(40)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidato = calcular_caracteristicas(contornos[0])

    assert candidato is not None
    assert candidato.aspecto == 1.0
    assert candidato.extensao > 0.95  # quadrado preenche quase todo o próprio bbox
    assert candidato.bbox == (30, 30, 41, 41)  # cv2.rectangle inclui os dois cantos


def test_circulo_tem_circularidade_e_solidez_altas():
    mascara = _mascara_com_circulo(20)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidato = calcular_caracteristicas(contornos[0])

    assert candidato is not None
    # discretização do contorno poligonal nunca chega em 1.0 mesmo pra um
    # círculo perfeito; 0.85 já separa bem de formas alongadas (ver risco)
    assert candidato.circularidade > 0.85
    assert candidato.solidez > 0.95


def test_retangulo_alongado_tem_aspecto_alto():
    mascara = np.zeros((100, 200), dtype=np.uint8)
    cv2.rectangle(mascara, (10, 45), (190, 55), 255, -1)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidato = calcular_caracteristicas(contornos[0])

    assert candidato is not None
    assert candidato.aspecto > 10
    assert candidato.circularidade < 0.3


def test_extrair_candidatos_filtra_por_area_minima():
    mascara = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(mascara, (20, 20), 2, 255, -1)   # ruído bem pequeno
    cv2.circle(mascara, (70, 70), 15, 255, -1)  # defeito de verdade

    candidatos_sem_filtro = extrair_candidatos(mascara, area_minima=1)
    candidatos_filtrados = extrair_candidatos(mascara, area_minima=40)

    assert len(candidatos_sem_filtro) == 2
    assert len(candidatos_filtrados) == 1


def test_contorno_degenerado_retorna_none():
    contorno_degenerado = np.array([[[0, 0]], [[0, 0]], [[0, 0]]])
    assert calcular_caracteristicas(contorno_degenerado) is None
