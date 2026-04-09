import pandas as pd
from analysis import compute_kpis, run_statistical_tests


def test_compute_kpis_returns_expected_keys():
    df = pd.DataFrame({
        "order_id": [1, 1, 2, 2],
        "line_revenue": [100, 50, 80, 20],
        "order_month": ["2026-01", "2026-01", "2026-02", "2026-02"],
        "city": ["Amman", "Amman", "Irbid", "Irbid"],
        "registration_month": ["2025-12", "2025-12", "2026-01", "2026-01"],
        "category": ["Electronics", "Electronics", "Fashion", "Fashion"]
    })

    kpis = compute_kpis(df)

    assert "total_revenue" in kpis
    assert "monthly_revenue" in kpis
    assert "avg_order_value" in kpis
    assert "revenue_by_city" in kpis
    assert "cohort_revenue" in kpis
    assert "revenue_by_category" in kpis
    assert "order_values" in kpis


def test_total_revenue_is_correct():
    df = pd.DataFrame({
        "order_id": [1, 1, 2],
        "line_revenue": [100, 50, 25],
        "order_month": ["2026-01", "2026-01", "2026-02"],
        "city": ["Amman", "Amman", "Irbid"],
        "registration_month": ["2025-12", "2025-12", "2026-01"],
        "category": ["Electronics", "Electronics", "Fashion"]
    })

    kpis = compute_kpis(df)
    assert kpis["total_revenue"] == 175


def test_statistical_tests_return_dict():
    df = pd.DataFrame({
        "order_id": [1, 2, 3, 4, 5, 6],
        "line_revenue": [100, 110, 200, 210, 300, 320],
        "city": ["Amman", "Amman", "Irbid", "Irbid", "Zarqa", "Zarqa"],
        "category": ["Electronics", "Electronics", "Fashion", "Fashion", "Home", "Home"]
    })

    results = run_statistical_tests(df)
    assert isinstance(results, dict)