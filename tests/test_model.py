import numpy as np
import pytest
from nic.pipeline import load_model, gpt2_probs_sequence

VOCAB_SIZE = 50_257

@pytest.fixture(scope="module")
def model_and_device():
    return load_model()

def test_probs_shape(model_and_device):
    model = model_and_device
    probs = gpt2_probs_sequence([15496, 995], model)
    assert probs.shape == (2, VOCAB_SIZE)