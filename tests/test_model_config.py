"""Model configuration compatibility; no model requests."""
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from schema_study.cli import model_config


class ModelConfigTests(unittest.TestCase):
    def load(self, document, selector):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'models.json'
            path.write_text(json.dumps(document), encoding='utf-8')
            return model_config(path, selector)

    def test_legacy_backend_alias_and_secret_exclusion(self):
        entry = dict(backend='deepseek', model='deepseek-flash',
                     base_url='https://example.invalid/v1', api_key='test-secret')
        with patch.dict(os.environ, {}, clear=True):
            model = self.load([entry], 'deepseek')
            self.assertEqual(model['model'], 'deepseek-flash')
            self.assertEqual(model['api_type'], 'chat_completions')
            self.assertEqual(os.environ[model['api_key_env']], 'test-secret')
            self.assertNotIn('test-secret', json.dumps(model))
            self.assertNotIn('api_key', model)
            self.assertNotIn('backend', model)

    def test_explicit_options_and_exact_id_take_precedence(self):
        entries = [dict(backend='service', model='first', base_url='https://example.invalid/v1',
                        api_type='responses', api_key_env='MY_KEY', reasoning_effort='medium'),
                   dict(backend='service', model='second', base_url='https://example.invalid/v1')]
        model = self.load(entries, 'first')
        self.assertEqual(model['api_type'], 'responses')
        self.assertEqual(model['api_key_env'], 'MY_KEY')
        self.assertEqual(model['reasoning_effort'], 'medium')
        with self.assertRaisesRegex(ValueError, 'matched 2 entries'):
            self.load(entries, 'service')

    def test_current_presets_unchanged(self):
        model = self.load({'defaults': {'reasoning_effort':'medium'}, 'models': {
            'deepseek': dict(model='test-model',provider='compatible',api_type='responses',
                             base_url='https://example.invalid/v1',api_key_env='MODEL_API_KEY')}}, 'deepseek')
        self.assertEqual(model['api_type'], 'responses')
        self.assertEqual(model['reasoning_effort'], 'medium')
        self.assertEqual(model['api_key_env'], 'MODEL_API_KEY')

    def test_new_format_inline_key_without_environment_variable(self):
        entry = dict(model='test-model', provider='compatible', api_type='responses',
                     base_url='https://example.invalid/v1', api_key='test-inline-secret')
        with patch.dict(os.environ, {}, clear=True):
            model = self.load({'models': {'deepseek':entry}}, 'deepseek')
            self.assertEqual(os.environ[model['api_key_env']], 'test-inline-secret')
            self.assertNotIn('api_key', model)
            self.assertNotIn('test-inline-secret', json.dumps(model))

    def test_inline_key_overrides_named_process_variable(self):
        entry = dict(model='test-model', provider='compatible', api_type='responses',
                     base_url='https://example.invalid/v1', api_key='new-test-secret', api_key_env='MY_KEY')
        with patch.dict(os.environ, {'MY_KEY':'old-test-secret'}, clear=True):
            model = self.load({'models': {'deepseek':entry}}, 'deepseek')
            self.assertEqual(model['api_key_env'], 'MY_KEY')
            self.assertEqual(os.environ['MY_KEY'], 'new-test-secret')

    def test_literal_in_environment_field_is_not_echoed(self):
        entry = dict(model='test-model', provider='compatible', api_type='responses',
                     base_url='https://example.invalid/v1', api_key_env='test-secret-with-dashes')
        with self.assertRaisesRegex(ValueError, 'put a literal API key in api_key instead') as caught:
            self.load({'models': {'deepseek':entry}}, 'deepseek')
        self.assertNotIn('test-secret-with-dashes', str(caught.exception))

    def test_invalid_inline_key_type(self):
        entry = dict(model='test-model', provider='compatible', api_type='responses',
                     base_url='https://example.invalid/v1', api_key=123)
        with self.assertRaisesRegex(ValueError, 'api_key must be a string'):
            self.load({'models': {'deepseek':entry}}, 'deepseek')

    def test_invalid_shape_has_actionable_error(self):
        for document in ([], [1], 'invalid', {}):
            with self.subTest(document=document), self.assertRaisesRegex(ValueError, 'Expected'):
                self.load(document, 'deepseek')


if __name__ == '__main__':
    unittest.main()
