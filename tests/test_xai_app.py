"""Local HTTP contract tests. Kaggle GPU is mocked and never downloaded."""
import base64
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import app as studio


def encode(img):
    b = io.BytesIO()
    img.save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


def fake_run(label=None):
    png = encode(Image.new('RGB', (96, 96), 'white'))
    return {
        'approach': 'a2',
        'model_label': 'Approach 2 · SD 1.5 + ControlNet + LoRA',
        'mode': 'ai_inference',
        'source_png_b64': png,
        'conditioning_png_b64': png,
        'output_png_b64': png,
        'metadata': {'execution_mode':'ai_inference','preprocessing_selected':'canny',
                     'preprocessing_auto_metrics': {}, 'settings': {'seed':42},
                     'model_executed': 'test-base', 'generation_ms':3},
        'explanation': {'summary':'Stub', 'preprocessing_reason':'Testing',
                        'model_role':'Test', 'prompt_role':'Test', 'limitation':'Test'},
        'lab_label': label,
    }


class XAIApplicationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.patches = [
            patch.object(studio.config, 'MEDIA_DIR', Path(self.tmp.name)/'media'),
            patch.object(studio.config, 'HISTORY_FILE', Path(self.tmp.name)/'history.json'),
            patch.object(studio.config, 'BACKEND_MODE', 'kaggle'),
        ]
        for p in self.patches: p.start()
        studio.config.MEDIA_DIR.mkdir(parents=True, exist_ok=True)
        self.client = studio.app.test_client()

    def tearDown(self):
        for p in reversed(self.patches): p.stop()
        self.tmp.cleanup()

    @staticmethod
    def payload(mode='auto', approach='a2'):
        b = io.BytesIO()
        Image.new('RGB', (96, 96), 'white').save(b, 'PNG')
        b.seek(0)
        return {'sketch': (b, 'test.png'), 'preprocessing': mode, 'approach': approach,
                'seed': '42', 'steps':'30', 'prompt':'A house', 'lora_scale':'.7'}

    def test_generate_lineart_forwarded_and_metadata_saved(self):
        with patch.object(studio, 'send_image', return_value=fake_run()) as sender:
            response = self.client.post('/api/generate', data=self.payload('lineart'))
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json['mode'], 'ai_inference')
            self.assertEqual(self.client.get(response.json['output_url']).status_code, 200)
            self.assertEqual(self.client.get('/api/history').json['count'], 1)
            self.assertEqual(sender.call_args.args[2], 'lineart')

    def test_lora_lab_four_runs_saved_and_differences_reported(self):
        labels = ['A · Text only','B · Text + LoRA','C · Text + ControlNet','D · Text + ControlNet + LoRA']
        with patch.object(studio, 'send_image', return_value={'runs': [fake_run(x) for x in labels], 'notice':'Research'}):
            response = self.client.post('/api/ablation', data=self.payload('auto'))
            self.assertEqual(response.status_code, 200, response.get_data(as_text=True))
            self.assertEqual(len(response.json['runs']), 4)
            self.assertEqual(response.json['pixel_differences']['A_vs_B'], 0)
            self.assertEqual(self.client.get('/api/history').json['count'], 4)

    def test_lora_lab_does_not_fabricate_gpu_results_in_mock(self):
        with patch.object(studio.config, 'BACKEND_MODE', 'mock'):
            response = self.client.post('/api/ablation', data=self.payload())
            self.assertEqual(response.status_code, 503)
            self.assertIn('Kaggle', response.json['error'])


if __name__ == '__main__':
    unittest.main()
