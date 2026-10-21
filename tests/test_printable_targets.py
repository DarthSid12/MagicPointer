"""Print geometry and dictionary compatibility for the decorative wall cards."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from magicpointer.calibration import targets
from magicpointer.cli import parser


class PrintableTargetsTests(unittest.TestCase):
    def test_default_pair_is_a4_and_keeps_exact_codes_and_quiet_zones(self):
        self.assertEqual(parser().parse_args(['targets']).marker_mm, 80.)
        with tempfile.TemporaryDirectory() as directory:
            targets(directory)
            root = ET.parse(Path(directory) / 'markers-pair.svg').getroot()
            self.assertEqual((root.get('width'), root.get('height')), ('210.0mm', '297.0mm'))
            groups = root.findall('{*}g')
            self.assertEqual(len(groups), 2)
            for marker_id, card in enumerate(groups):
                rects = list(card.find(f"{{*}}g[@id='aruco-pattern-{marker_id}']"))
                # Rasterize the SVG's exact pattern coordinates at 6 pixels/mm.
                image = np.full((804, 780), 255, np.uint8)
                for rect in rects:
                    x, y, w, h = [float(rect.get(k)) for k in ('x', 'y', 'width', 'height')]
                    image[round(y*6):round((y+h)*6), round(x*6):round((x+w)*6)] = 0
                dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
                _, ids, _ = cv2.aruco.ArucoDetector(dictionary).detectMarkers(image)
                self.assertEqual(ids.flatten().tolist(), [marker_id])
                self.assertAlmostEqual(max(float(r.get('x'))+float(r.get('width')) for r in rects)
                                       - min(float(r.get('x')) for r in rects), 80.)
                white = next(r for r in card.findall('{*}rect') if r.get('fill') == 'white')
                self.assertAlmostEqual(min(float(r.get('x')) for r in rects)-float(white.get('x')), 80/6)
                self.assertAlmostEqual(float(white.get('width')), 80*4/3)

    def test_explicit_large_markers_are_not_silently_shrunk_to_a4(self):
        with tempfile.TemporaryDirectory() as directory:
            targets(directory, marker_mm=200.)
            root = ET.parse(Path(directory) / 'markers-pair.svg').getroot()
            self.assertGreater(float(root.get('width')[:-2]), 210)
            self.assertGreater(float(root.get('height')[:-2]), 297)
            card = ET.parse(Path(directory) / 'marker-0.svg').getroot()
            self.assertAlmostEqual(float(card.get('width')[:-2]), 325.)
