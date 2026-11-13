"""Final-validation Y sources explicitly authorized by the user."""
FINAL_RESPONSE_SOURCES={
    'logit_nn':'frozen_neural_network_binary_logit',
    'p_nn':'frozen_neural_network_probability',
    'choice_nn':'frozen_neural_network_argmax_choice',
    'choice_observed':'actual_observed_choice',
}


def final_Y_eligible(channel):return channel in FINAL_RESPONSE_SOURCES


def response_fields(channel):
    return dict(final_Y_eligible=final_Y_eligible(channel),
        Y_source=FINAL_RESPONSE_SOURCES.get(channel,'auxiliary_or_unverified_response'),
        analytic_logit_or_probability_can_count_as_success=False)
