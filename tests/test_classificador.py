import numpy as np

from inspecao.classificador import ParametrosRegras, candidato_para_vetor, classificar_regras
from inspecao.features import Candidato

_CONTORNO_DUMMY = np.zeros((1, 1, 2), dtype=np.int32)


def _candidato(area, perimetro, aspecto, circularidade, solidez, extensao) -> Candidato:
    return Candidato(
        contorno=_CONTORNO_DUMMY,
        bbox=(0, 0, 10, 10),
        area=area,
        perimetro=perimetro,
        aspecto=aspecto,
        circularidade=circularidade,
        solidez=solidez,
        extensao=extensao,
    )


def test_classifica_furo_ausente_quando_redondo_solido_e_pequeno():
    candidato = _candidato(area=400, perimetro=70, aspecto=1.05, circularidade=0.85, solidez=0.95, extensao=0.7)
    assert classificar_regras(candidato) == "furo_ausente"


def test_mancha_redonda_grande_nao_vira_furo():
    # mesma forma do furo, mas grande demais pra ser rebite ausente
    candidato = _candidato(area=1500, perimetro=150, aspecto=1.05, circularidade=0.85, solidez=0.95, extensao=0.7)
    assert classificar_regras(candidato) != "furo_ausente"


def test_classifica_risco_quando_alongado():
    candidato = _candidato(area=300, perimetro=250, aspecto=5.0, circularidade=0.1, solidez=0.7, extensao=0.15)
    assert classificar_regras(candidato) == "risco"


def test_classifica_mancha_quando_irregular_e_nao_alongada():
    candidato = _candidato(area=600, perimetro=140, aspecto=1.2, circularidade=0.3, solidez=0.5, extensao=0.35)
    assert classificar_regras(candidato) == "mancha"


def test_parametros_customizados_mudam_o_limiar():
    candidato = _candidato(area=300, perimetro=250, aspecto=2.0, circularidade=0.1, solidez=0.7, extensao=0.15)
    params_padrao = ParametrosRegras()
    params_mais_estritos = ParametrosRegras(aspecto_risco=2.5)

    assert classificar_regras(candidato, params_padrao) == "risco"
    assert classificar_regras(candidato, params_mais_estritos) == "mancha"


def test_candidato_para_vetor_segue_ordem_das_colunas():
    candidato = _candidato(area=1, perimetro=2, aspecto=3, circularidade=4, solidez=5, extensao=6)
    assert candidato_para_vetor(candidato) == [1, 2, 3, 4, 5, 6]
