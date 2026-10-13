"""Distributed Run:ai streaming over XCCL must engage on XPU and load correct weights.

Usage:
python3 -m unittest test_xpu_runai_model_streamer_distributed.TestXPURunaiDistributedStreaming
"""

import json
import tempfile
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_xpu_ci(est_time=300, suite="nightly-xpu-kernel-main-2-gpu", nightly=True)

PROMPTS = [
    "The capital of France is",
    "The future of AI is",
]
SAMPLING_PARAMS = {"temperature": 0, "max_new_tokens": 16}
# Logged by runai_model_streamer's DistributedStreamer on rank 0 once the
# broadcast path is taken; every fallback logs "fallback to non distributed".
DISTRIBUTED_LOG = "Using distributed streaming between 2 processes"
FALLBACK_LOG = "fallback to non distributed streaming"


def _streamer_supports_xpu() -> bool:
    try:
        from runai_model_streamer.distributed_streamer.distributed_streamer import (
            DistributedStreamer,
        )
    except ImportError:
        return False
    # Added with XCCL support in dsx-ai-factory/model-streamer#205.
    return hasattr(DistributedStreamer, "get_xpu_free_memory")


def _generate(load_args: list[str]) -> tuple[list[str], str]:
    with tempfile.TemporaryFile("w+") as stdout, tempfile.TemporaryFile("w+") as stderr:
        process = popen_launch_server(
            DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--device", "xpu", "--tp-size", "2", *load_args],
            env={
                "RUNAI_STREAMER_LOG_LEVEL": "INFO",
                "RUNAI_STREAMER_LOG_TO_STDERR": "1",
            },
            return_stdout_stderr=(stdout, stderr),
        )
        try:
            response = requests.post(
                DEFAULT_URL_FOR_TEST + "/generate",
                json={"text": PROMPTS, "sampling_params": SAMPLING_PARAMS},
            )
            response.raise_for_status()
        finally:
            kill_process_tree(process.pid)
        stdout.seek(0)
        stderr.seek(0)
        logs = stdout.read() + stderr.read()
    return [output["text"] for output in response.json()], logs


@unittest.skipUnless(
    _streamer_supports_xpu(),
    "installed runai_model_streamer lacks XPU distributed streaming "
    "(dsx-ai-factory/model-streamer#205)",
)
class TestXPURunaiDistributedStreaming(CustomTestCase):
    def test_distributed_streaming_matches_default_loader(self):
        expected, _ = _generate(["--load-format", "auto"])
        actual, logs = _generate(
            [
                "--load-format",
                "runai_streamer",
                "--model-loader-extra-config",
                json.dumps({"distributed": True}),
            ]
        )
        self.assertIn(DISTRIBUTED_LOG, logs)
        self.assertNotIn(FALLBACK_LOG, logs)
        self.assertTrue(all(expected), expected)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
