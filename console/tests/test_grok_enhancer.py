import unittest
from unittest.mock import Mock, patch
import requests
from grok_enhancer import enhance_prompt, INSTRUCTIONS


class EnhancerTests(unittest.TestCase):
    def test_custom_system_prompt_replaces_default_and_blank_restores_default(self):
        response = Mock(status_code=200)
        response.json.return_value = {'status': 'completed', 'output': [
            {'type': 'message', 'role': 'assistant', 'content': [
                {'type': 'output_text', 'text': 'Enhanced robot prompt.'}]}]}
        for instructions, expected in [('Keep it short.\nPreserve references.', 'Keep it short.\nPreserve references.'),
                                       ('   ', INSTRUCTIONS)]:
            with patch('grok_enhancer.requests.post', return_value=response) as post:
                enhance_prompt({'prompt': 'A robot'}, {'grok_api_key': 'key', 'grok_system_prompt': instructions})
            self.assertEqual(post.call_args.kwargs['json']['instructions'], expected)
            self.assertFalse(post.call_args.kwargs['json']['store'])

    def test_invalid_system_prompt_never_calls_provider(self):
        for instructions in (None, ['instructions'], 'x' * 60001):
            with patch('grok_enhancer.requests.post') as post:
                with self.assertRaises(ValueError):
                    enhance_prompt({'prompt': 'A robot'}, {'grok_api_key': 'key', 'grok_system_prompt': instructions})
                post.assert_not_called()

    def test_input_validation_does_not_call_provider(self):
        for data, cfg in [({'prompt': ''}, {'grok_api_key': 'key'}),
                          ({'prompt': 'x'}, {}),
                          ({'prompt': 'x' * 60001}, {'grok_api_key': 'key'})]:
            with patch('grok_enhancer.requests.post') as post:
                with self.assertRaises(ValueError):
                    enhance_prompt(data, cfg)
                post.assert_not_called()

    def test_failures_do_not_leak_provider_body_or_retry(self):
        for status in [401, 403, 429, 500]:
            with patch('grok_enhancer.requests.post', return_value=Mock(
                    status_code=status, text='secret-key provider details')) as post:
                with self.assertRaises(RuntimeError) as caught:
                    enhance_prompt({'prompt': 'A robot'}, {'grok_api_key': 'secret-key'})
                self.assertNotIn('secret-key', str(caught.exception))
                post.assert_called_once()

    def test_timeout_is_manual_retry(self):
        with patch('grok_enhancer.requests.post', side_effect=requests.Timeout('secret')) as post:
            with self.assertRaises(TimeoutError):
                enhance_prompt({'prompt': 'A robot'}, {'grok_api_key': 'key'})
            post.assert_called_once()

    def test_incomplete_and_empty_responses_rejected(self):
        for result in [{'status': 'incomplete', 'output': []},
                       {'status': 'completed', 'output': []}, {'bad': 'shape'}]:
            response = Mock(status_code=200)
            response.json.return_value = result
            with patch('grok_enhancer.requests.post', return_value=response):
                with self.assertRaises(RuntimeError):
                    enhance_prompt({'prompt': 'A robot'}, {'grok_api_key': 'key'})
