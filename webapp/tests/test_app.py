import unittest
from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
from webapp.app import create_app
from webapp.model import FEATURES, load_model
from sklearn.model_selection import train_test_split


class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = create_app().test_client()
        cls.example = {
            "startup_name": "Demo startup", "funding_total_usd": "5000000",
            "funding_rounds": "3", "founded_at_year": "2008",
            "first_funding_at_year": "2009", "last_funding_at_year": "2014",
            "country_code": "USA", "first_category": "Software"
        }

    def test_page_and_health(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b'name="viewport"', page.data)
        self.assertIn(b"Northstar Labs", page.data)
        self.assertIn(b"Harbor Cart", page.data)
        self.assertIn(b"Seedling Health", page.data)
        self.assertIn(b"Clear the form", page.data)
        self.assertIn(b"Content-Security-Policy", str(page.headers).encode())
        self.assertEqual(self.client.get("/health").json["status"], "ok")

    def test_prediction_and_name_do_not_change_model_score(self):
        response = self.client.post("/predict", json=self.example)
        self.assertEqual(response.status_code, 200)
        result = response.json
        self.assertAlmostEqual(result["acquired_score"] + result["closed_score"], 1)
        self.assertEqual(result["prediction"], int(result["acquired_score"] > 0.5))
        self.assertEqual(result["known_features"], 7)
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        renamed = dict(self.example, startup_name="<script>alert(1)</script>")
        self.assertEqual(self.client.post("/predict", json=renamed).json["acquired_score"], result["acquired_score"])

    def test_missing_values_and_newer_dates(self):
        response = self.client.post("/predict", json={"funding_total_usd": "0"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["known_features"], 1)
        self.assertTrue(response.json["warnings"])
        newer = dict(self.example, last_funding_at_year=str(date.today().year))
        self.assertTrue(any("2015" in warning for warning in self.client.post("/predict", json=newer).json["warnings"]))

    def test_invalid_numeric_inputs(self):
        for value in ["NaN", "Infinity", "-1", True, {}, [], "1e999"]:
            with self.subTest(value=value):
                response = self.client.post("/predict", json=dict(self.example, funding_total_usd=value))
                self.assertEqual(response.status_code, 400)
                self.assertIn("funding_total_usd", response.json["errors"])
        for field, value in [("funding_rounds", "2.5"), ("funding_rounds", "0"), ("founded_at_year", "1700"), ("last_funding_at_year", "9999")]:
            self.assertEqual(self.client.post("/predict", json=dict(self.example, **{field: value})).status_code, 400)

    def test_invalid_categories_dates_and_empty_payload(self):
        for data in [{}, {"startup_name": "Name only"}, [], None]:
            self.assertEqual(self.client.post("/predict", json=data).status_code, 400)
        for field in ["country_code", "first_category"]:
            self.assertEqual(self.client.post("/predict", json=dict(self.example, **{field: "not-a-real-option"})).status_code, 400)
        reversed_dates = dict(self.example, first_funding_at_year="2015", last_funding_at_year="2010")
        self.assertEqual(self.client.post("/predict", json=reversed_dates).status_code, 400)
        self.assertEqual(self.client.post("/predict", json=dict(self.example, founded_at_year="2015")).status_code, 400)
        self.assertEqual(self.client.post("/predict", data="bad", content_type="application/json").status_code, 400)
        self.assertEqual(self.client.post("/predict", data="x" * 20000, content_type="application/json").status_code, 413)

    def test_artifact_matches_notebook_holdout(self):
        root = Path(__file__).resolve().parents[2]
        df = pd.read_csv(root / "data" / "cleaned_startup_data.csv").sort_values("startup_id").reset_index(drop=True)
        _, X_test = train_test_split(df[FEATURES], test_size=0.2, stratify=df["target"], random_state=42)
        model, metadata = load_model()
        reference = pd.read_csv(root / "results" / "random_forest_predictions.csv")
        np.testing.assert_array_equal(reference["startup_id"], df.loc[X_test.index, "startup_id"])
        np.testing.assert_array_equal(reference["predicted"], model.predict(X_test))
        np.testing.assert_allclose(reference["score"], model.predict_proba(X_test)[:, 1], atol=1e-12, rtol=0)
        self.assertEqual(metadata["train_rows"], 9429)


if __name__ == "__main__":
    unittest.main()
