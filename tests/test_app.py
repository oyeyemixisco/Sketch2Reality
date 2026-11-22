"""Smoke tests for the local-only Flask prototype."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as sketch_app


class Sketch2FutureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.media = Path(self.temp.name) / 'media'
        self.media.mkdir()
        self.history = Path(self.temp.name) / 'history.json'
        self.media_patch = patch.object(sketch_app.config, 'MEDIA_DIR', self.media)
        self.history_patch = patch.object(sketch_app.config, 'HISTORY_FILE', self.history)
        self.media_patch.start()
        self.history_patch.start()
        self.backend_patch = patch.object(sketch_app.config, 'BACKEND_MODE', 'mock')
        self.backend_patch.start()
        self.client = sketch_app.app.test_client()

    def tearDown(self):
        self.backend_patch.stop()
        self.media_patch.stop()
        self.history_patch.stop()
        self.temp.cleanup()

    def test_home_and_health(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sketch2Future', response.data)
        health = self.client.get('/api/health').json
        self.assertEqual(health['mode'], 'local_demo')
        self.assertFalse(health['gpu_connected'])

    def make_payload(self, approach='a1'):
        image = Image.new('RGB', (220, 180), 'white')
        stream = io.BytesIO()
        image.save(stream, 'PNG')
        stream.seek(0)
        return {'sketch': (stream, 'drawing.png'), 'approach': approach,
                'preprocessing': 'auto', 'seed': '42', 'steps': '28',
                'prompt': 'A scenic drawing'}

    def test_generate_and_explain_and_media(self):
        response = self.client.post('/api/generate', data=self.make_payload())
        self.assertEqual(response.status_code, 200)
        data = response.json
        self.assertEqual(data['mode'], 'mock')
        self.assertIn('NOT', data['explanation']['summary'].upper())
        self.assertIn('No diffusion model', data['metadata']['model_executed'])
        self.assertEqual(self.client.get(data['output_url']).status_code, 200)
        meta = self.client.get(f"/api/runs/{data['run_id']}/metadata")
        self.assertEqual(meta.status_code, 200)
        self.assertIn(b'"execution_mode": "mock"', meta.data)
        history = self.client.get('/api/history').json
        self.assertEqual(history['count'], 1)

    def test_compare_three_mock_approaches(self):
        response = self.client.post('/api/compare', data=self.make_payload())
        self.assertEqual(response.status_code, 200)
        self.assertEqual([r['approach'] for r in response.json['runs']], ['a1','a2','a3'])
        self.assertEqual(self.client.get('/api/history').json['count'], 3)
        self.assertEqual(self.client.delete('/api/history').status_code, 200)
        self.assertEqual(self.client.get('/api/history').json['count'], 0)

    def test_reject_missing_or_invalid_sketch(self):
        self.assertEqual(self.client.post('/api/generate', data={}).status_code, 400)
        response = self.client.post('/api/generate', data={
            'sketch': (io.BytesIO(b'bad image'), 'test.png')
        })
        self.assertEqual(response.status_code, 400)

if __name__ == '__main__':
    unittest.main()
