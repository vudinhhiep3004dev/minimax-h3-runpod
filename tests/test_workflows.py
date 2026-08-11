"""Fast schema tests; no CUDA, weights, or Runpod credentials required."""

from __future__ import annotations

import sys
import types
import unittest
from pathlib import Path



# The worker's Runpod SDK is intentionally only installed in the GPU image.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.modules.setdefault("runpod", types.SimpleNamespace())

import handler  # noqa: E402
from h3_pipeline import duration_to_num_frames, normalize_workflow, resolve_resolution  # noqa: E402


class WorkflowSchemaTests(unittest.TestCase):
    def test_workflow_aliases_and_frame_grid(self):
        self.assertEqual(normalize_workflow("i2va"), "fl2va")
        self.assertEqual(normalize_workflow("r2va"), "ref2va")
        frames, duration = duration_to_num_frames(8)
        self.assertEqual(frames % 17, 5)
        self.assertTrue(5 <= duration <= 15)


    def test_text_job_remains_backward_compatible(self):
        spec = handler._normalize_one({"prompt": "a fox", "quality": "draft"})
        self.assertEqual(spec["workflow"], "t2va")
        self.assertEqual(spec["mode"], "t2va")
        self.assertIsNone(spec["image"])


    def test_image_job_infers_i2va(self):
        spec = handler._normalize_one(
            {"prompt": "a fox runs", "image": "https://example.test/fox.png"}
        )
        self.assertEqual(spec["workflow"], "fl2va")
        self.assertEqual(spec["mode"], "i2va")


    def test_ref_job_preserves_order_and_type(self):
        spec = handler._normalize_one(
            {
                "prompt": "use the subject",
                "workflow": "ref2va",
                "references": [
                    {"type": "video", "url": "https://example.test/motion.mp4"},
                    {"type": "image", "url": "https://example.test/subject.png"},
                ],
            }
        )
        self.assertEqual(spec["workflow"], "ref2va")
        self.assertEqual([item["type"] for item in spec["references"]], ["video", "image"])


    def test_invalid_mixed_modes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "not references"):
            handler._normalize_one(
                {
                    "prompt": "bad",
                    "workflow": "fl2va",
                    "image": "https://example.test/a.png",
                    "references": [{"type": "image", "url": "https://example.test/b.png"}],
                }
            )


    def test_resolution_contract(self):
        self.assertEqual(resolve_resolution("9:16", "draft"), (544, 960))


if __name__ == "__main__":
    unittest.main()
