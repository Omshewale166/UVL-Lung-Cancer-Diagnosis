"""Architecture contract tests for the CNN described in the paper."""

import pytest

tf = pytest.importorskip("tensorflow")

from lung_cancer.model import build_model


def test_model_geometry_and_parameter_count() -> None:
    model = build_model()
    assert model.input_shape == (None, 256, 256, 1)
    assert model.output_shape == (None, 3)
    assert [layer.output.shape[1:] for layer in model.layers[:6]] == [
        (254, 254, 64), (254, 254, 64), (127, 127, 64),
        (125, 125, 64), (62, 62, 64), (246016,),
    ]
    assert model.get_layer("conv2d").count_params() == 640
    assert model.get_layer("conv2d_1").count_params() == 36928
    assert model.get_layer("dense").count_params() == 3936272
    assert model.get_layer("class_probabilities").count_params() == 51
    assert model.count_params() == 3973891


def test_weights_checkpoint_round_trip(tmp_path) -> None:
    from lung_cancer.model import load_model

    checkpoint = tmp_path / "model.weights.h5"
    original = build_model(compile_model=False)
    original.save_weights(checkpoint)
    restored = load_model(checkpoint)
    assert restored.input_shape == original.input_shape
    assert restored.output_shape == original.output_shape
    for expected, actual in zip(original.get_weights(), restored.get_weights()):
        assert (expected == actual).all()
