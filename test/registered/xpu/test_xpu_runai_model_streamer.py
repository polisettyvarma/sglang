"""Weights streamed by Run:ai Model Streamer onto XPU must match the default loader.

Usage:
python3 -m unittest test_xpu_runai_model_streamer.TestXPURunaiModelStreamer
"""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST, CustomTestCase

register_xpu_ci(est_time=180, suite="stage-b-test-1-gpu-xpu")

PROMPTS = [
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 16}


def _generate(load_format: str) -> list[str]:
    engine = sgl.Engine(
        model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
        load_format=load_format,
        device="xpu",
        mem_fraction_static=0.5,
    )
    try:
        outputs = engine.generate(PROMPTS, SAMPLING_PARAMS)
    finally:
        engine.shutdown()
    return [output["text"] for output in outputs]


class TestXPURunaiModelStreamer(CustomTestCase):
    def test_greedy_output_matches_default_loader(self):
        expected = _generate("auto")
        actual = _generate("runai_streamer")
        self.assertTrue(all(expected), expected)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
