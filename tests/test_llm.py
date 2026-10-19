import importlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import requests


class OpenAIEndpointTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {"LLM_API_KEY": "test-key"}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        from core import api_config, llm
        self.config = importlib.reload(api_config)
        self.llm = importlib.reload(llm)

    def response(self, usage=None):
        payload = {"choices": [{"message": {"content": " SELECT 1 "}}]}
        if usage is not None:
            payload["usage"] = usage
        response = requests.Response()
        response.status_code = 200
        response._content = json.dumps(payload).encode()
        response.headers["Content-Type"] = "application/json"
        return response

    @patch("requests.sessions.Session.request")
    def test_request_uses_openai_route_model_and_auth(self, request):
        request.return_value = self.response({"prompt_tokens": 3, "completion_tokens": 2})
        self.assertEqual(self.llm.api_func("hello"), ("SELECT 1", 3, 2))
        args, kwargs = request.call_args
        self.assertEqual(args[0].lower(), "post")
        self.assertEqual(args[1], "http://localhost:20128/v1/chat/completions")
        self.assertEqual(kwargs["headers"]["Authorization"], "Bearer test-key")
        body = json.loads(kwargs["data"])
        self.assertEqual(body, {
            "model": "cx/gpt-6-luna",
            "messages": [{"role": "user", "content": "hello"}],
        })

    @patch("requests.sessions.Session.request")
    def test_missing_usage_and_log_output(self, request):
        request.return_value = self.response()
        with tempfile.TemporaryDirectory() as directory:
            self.llm.init_log_path(str(Path(directory) / "log.txt"))
            self.assertEqual(self.llm.safe_call_llm("hello", idx=1), "SELECT 1")
            trace = json.loads((Path(directory) / "api_trace.json").read_text())
            self.assertEqual(trace["prompt_token"], 0)
            self.assertEqual(trace["response_token"], 0)
            self.assertEqual(trace["idx"], 1)

    def test_environment_overrides(self):
        with patch.dict(os.environ, {
            "LLM_BASE_URL": "http://localhost:9000/v1/",
            "LLM_MODEL": "custom-model",
            "LLM_API_KEY": "custom-key",
        }):
            importlib.reload(self.config)
            self.assertEqual(self.config.openai.api_base, "http://localhost:9000/v1")
            self.assertEqual(self.config.openai.api_key, "custom-key")
            self.assertEqual(self.config.MODEL_NAME, "custom-model")
            self.assertEqual(self.config.openai.api_type, "open_ai")
            self.assertIsNone(self.config.openai.api_version)

    @patch("requests.sessions.Session.request")
    @patch("core.llm.time.sleep")
    def test_missing_key_fails_without_retry_or_request(self, sleep, request):
        with patch.dict(os.environ, {"LLM_API_KEY": ""}):
            importlib.reload(self.config)
            with self.assertRaisesRegex(ValueError, "LLM_API_KEY"):
                self.llm.safe_call_llm("hello")
        sleep.assert_not_called()
        request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
