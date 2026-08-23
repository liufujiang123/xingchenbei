import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import ascend_perf_analyze as apa


class AscendPerfAnalyzeTest(unittest.TestCase):
    def write_source(self, text):
        tmp = tempfile.TemporaryDirectory()
        path = pathlib.Path(tmp.name) / "kernel.cpp"
        path.write_text(text, encoding="utf-8")
        return tmp, path

    def test_static_vector_scan_is_hypothesis(self):
        tmp, path = self.write_source(
            """
            AscendC::GlobalTensor<half> x;
            AscendC::LocalTensor<half> ub;
            AscendC::DataCopy(ub, x, 128);
            AscendC::SetFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::WaitFlag<AscendC::HardEvent::V_MTE2>(0);
            AscendC::Add(ub, ub, ub, 128);
            """
        )
        self.addCleanup(tmp.cleanup)
        result = apa.analyze([path], limit=3)
        self.assertEqual(result["operator_class"], "vector")
        self.assertEqual(result["confidence"], "static_hypothesis")
        self.assertIn("pipeline", result["static_risk_tags"])
        self.assertIn("memory", result["static_risk_tags"])
        self.assertEqual(result["observed_bottlenecks"], [])

    def test_profile_markers_override_static_class(self):
        tmp, path = self.write_source(
            """
            Matmul<MatmulType<TPosition::GM, CubeFormat::ND, half>> mm;
            TileMmad(mm);
            """
        )
        self.addCleanup(tmp.cleanup)
        profile = "\n".join(
            [
                "HARNESS_OPERATOR_CLASS=mixed_cv",
                "HARNESS_BOTTLENECKS=pipeline,synchronization",
                "HARNESS_PROFILE_NOTE=measured wait bubbles",
            ]
        )
        result = apa.analyze([path], profile_text=profile, limit=8)
        self.assertEqual(result["operator_class"], "mixed_cv")
        self.assertEqual(result["operator_class_source"], "profile_marker")
        self.assertEqual(result["confidence"], "profile_observed")
        self.assertIn("pipeline", result["observed_bottlenecks"])
        self.assertTrue(result["conflicts"])

    def test_advanced_patterns_hidden_by_default(self):
        tmp, path = self.write_source("AscendC::Add(dst, src0, src1, 128);")
        self.addCleanup(tmp.cleanup)
        profile = "HARNESS_OPERATOR_CLASS=vector\nHARNESS_BOTTLENECKS=compute,latency\n"
        core = apa.analyze([path], profile_text=profile, include_advanced=False, limit=50)
        advanced = apa.analyze([path], profile_text=profile, include_advanced=True, limit=50)
        core_ids = {item["id"] for item in core["candidates"]}
        advanced_ids = {item["id"] for item in advanced["candidates"]}
        self.assertNotIn("vector.microapi_register_kernel", core_ids)
        self.assertIn("vector.microapi_register_kernel", advanced_ids)


if __name__ == "__main__":
    unittest.main()
