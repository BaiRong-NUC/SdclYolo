import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("scope_audit",
    Path(__file__).resolve().parents[1]/"scripts/evaluation/audit_research_scope.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class ResearchScopeAuditTests(unittest.TestCase):
    def test_stop_string_is_not_a_positive_lead(self):
        self.assertFalse(audit.positive_lead(["STOP_POSITIVE_SIGNAL_RECIPE"], ["POSITIVE"]))

    def test_retains_distinct_seed_task_outcomes(self):
        result = dict(decision=dict(ranking_task="PROMISING_HEAD_SEED_REPEAT",
                                    regret_weight="POSITIVE_EXTRA_HEAD_SEED_SIGNAL",
                                    registered_policy=dict(not_a_decision=True)))
        self.assertEqual(len(audit.outcome_strings(result)), 2)
        self.assertTrue(audit.positive_lead(audit.outcome_strings(result), ["POSITIVE"]))

    def test_untraced_positive_prevents_complete_audit(self):
        rows = {1: dict(decisions=["PROMISING_POINT_ESTIMATE"])}
        protocol = dict(positive_keyword_detection=["PROMISING"], positive_lead_dispositions={})
        with self.assertRaisesRegex(ValueError, "Untraced positive"):
            audit.trace_leads(rows, protocol)

    def test_unknown_decision_schema_is_not_silently_a_stop(self):
        with self.assertRaises(ValueError):
            audit.outcome_strings(dict())


if __name__ == "__main__":
    unittest.main()
