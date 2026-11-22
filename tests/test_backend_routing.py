"""Pure-image routing and API-contract tests. No GPU or HF downloads required."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kaggle_backend import kaggle_api_server as server


class DummyLineart:
    def __call__(self, image, coarse=False):
        # Pure stub used only for routing tests; does NOT validate learned Lineart quality.
        return Image.new('RGB', image.size, (180, 180, 180))


class BackendRoutingTests(unittest.TestCase):
    def setUp(self):
        self.image = Image.new('RGB', (512, 512), 'white')
        d = ImageDraw.Draw(self.image)
        d.rectangle((70, 70, 430, 430), outline='black', width=5)
        d.line((70, 70, 250, 20, 430, 70), fill='black', width=4)

    def test_sd15_auto_returns_supported_method(self):
        result = server.preprocess(self.image, 'auto', 'a1')
        source, guidance, mode, metrics, reason, fallback = result
        self.assertIn(mode, {'canny', 'scribble', 'lineart'})
        self.assertEqual(source.size, guidance.size)
        self.assertIn('stroke_density', metrics)
        self.assertFalse(fallback)
        self.assertTrue(reason)

    def test_manual_lineart_for_both_sd15_approaches(self):
        with patch.object(server, '_LINEART_DETECTOR', DummyLineart()):
            for approach in ('a1', 'a2'):
                with self.subTest(approach=approach):
                    _, guidance, method, _, _, fallback = server.preprocess(self.image, 'lineart', approach)
                    self.assertEqual(method, 'lineart')
                    self.assertEqual(guidance.getpixel((0, 0)), (180, 180, 180))
                    self.assertFalse(fallback)

    def test_sd15_manual_canny_scribble(self):
        for mode in ('canny', 'scribble'):
            with self.subTest(mode=mode):
                _, guidance, method, _, _, _ = server.preprocess(self.image, mode, 'a1')
                self.assertEqual(method, mode)
                self.assertEqual(guidance.mode, 'RGB')

    def test_sdxl_original_auto_and_manual_routes(self):
        for mode in ('auto', 'canny', 'scribble'):
            with self.subTest(mode=mode):
                source, guidance, selected, metrics, reason, fallback = server.preprocess(self.image, mode, 'a3')
                self.assertIn(selected, {'canny', 'scribble'})
                self.assertEqual(source.size, (1024, 1024))
                self.assertEqual(guidance.size, source.size)
                self.assertIn('canny_density', metrics)
                self.assertFalse(fallback)

    def test_sdxl_lineart_is_rejected_for_single_generate_and_falls_back_in_comparison(self):
        with self.assertRaisesRegex(ValueError, 'Lineart'):
            server.preprocess(self.image, 'lineart', 'a3')
        _, _, method, _, reason, fallback = server.preprocess(self.image, 'lineart', 'a3', comparing=True)
        self.assertIn(method, {'canny', 'scribble'})
        self.assertTrue(fallback)
        self.assertIn('SDXL', reason)


if __name__ == '__main__':
    unittest.main()
