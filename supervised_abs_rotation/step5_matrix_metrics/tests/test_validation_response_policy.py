import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import matrix_io
from validation_response_policy import FINAL_RESPONSE_SOURCES,final_Y_eligible,response_fields


def test_only_actual_choice_or_neural_outputs_can_qualify_for_final_validation():
    assert set(matrix_io.CHANNELS)==set(FINAL_RESPONSE_SOURCES)=={'choice_observed','logit_nn','p_nn','choice_nn'}
    for channel in matrix_io.CHANNELS:
        assert final_Y_eligible(channel) and response_fields(channel)['final_Y_eligible']
    for channel in ('logit_true','p_true','logit_formula','p_formula','oracle_logit','unknown'):
        assert not final_Y_eligible(channel)
        assert not response_fields(channel)['analytic_logit_or_probability_can_count_as_success']
