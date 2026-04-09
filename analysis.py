"""Integration 4 — KPI Dashboard: Amman Digital Market Analytics

Extract data from PostgreSQL, compute KPIs, run statistical tests,
and create visualizations for the executive summary.

Usage:
    python analysis.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sqlalchemy import create_engine


def connect_db():
    """Create a SQLAlchemy engine connected to the amman_market database.

    Returns:
        engine: SQLAlchemy engine instance

    Notes:
        Use DATABASE_URL environment variable if set, otherwise default to:
        postgresql://postgres:postgres@localhost:5432/amman_market
    """
    db_url = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5433/amman_market"
    )
    engine = create_engine(db_url)
    return engine


def extract_data(engine):
    """Extract all required tables from the database into DataFrames.

    Args:
        engine: SQLAlchemy engine connected to amman_market

    Returns:
        dict: mapping of table names to DataFrames
              (e.g., {"customers": df, "products": df, "orders": df, "order_items": df})
    """
    customers = pd.read_sql("SELECT * FROM customers", engine)
    products = pd.read_sql("SELECT * FROM products", engine)
    orders = pd.read_sql("SELECT * FROM orders", engine)
    order_items = pd.read_sql("SELECT * FROM order_items", engine)

    customers["registration_date"] = pd.to_datetime(customers["registration_date"])
    orders["order_date"] = pd.to_datetime(orders["order_date"])

    # required cleaning based on assignment notes
    orders = orders[orders["status"] != "cancelled"].copy()
    order_items = order_items[order_items["quantity"] <= 100].copy()

    return {
        "customers": customers,
        "products": products,
        "orders": orders,
        "order_items": order_items
    }


def _prepare_df(data_input):
    """Prepare a unified analysis DataFrame.

    Supports:
    1) dict of DataFrames from extract_data()
    2) already-prepared DataFrame from tests
    """
    if isinstance(data_input, pd.DataFrame):
        df = data_input.copy()

        if "city" not in df.columns:
            df["city"] = "Unknown"
        else:
            df["city"] = df["city"].fillna("Unknown")

        if "line_revenue" not in df.columns:
            if "quantity" in df.columns and "unit_price" in df.columns:
                df["line_revenue"] = df["quantity"] * df["unit_price"]

        return df

    if isinstance(data_input, dict):
        customers = data_input["customers"].copy()
        products = data_input["products"].copy()
        orders = data_input["orders"].copy()
        order_items = data_input["order_items"].copy()

        df = (
            order_items
            .merge(orders, on="order_id", how="inner")
            .merge(products, on="product_id", how="left")
            .merge(customers, on="customer_id", how="left")
        )

        df["city"] = df["city"].fillna("Unknown")
        df["line_revenue"] = df["quantity"] * df["unit_price"]

        if "order_date" in df.columns:
            df["order_month"] = df["order_date"].dt.to_period("M").astype(str)
            df["order_week"] = df["order_date"].dt.to_period("W").astype(str)

        if "registration_date" in df.columns:
            df["registration_month"] = df["registration_date"].dt.to_period("M").astype(str)

        return df

    raise TypeError("data_input must be either a pandas DataFrame or a dict of DataFrames")


def compute_kpis(data_input):
    """Compute the 5 KPIs defined in kpi_framework.md.

    Args:
        data_input: dict of DataFrames from extract_data() or a prepared DataFrame

    Returns:
        dict: mapping of KPI names to their computed values (or DataFrames
              for time-series / cohort KPIs)

    Notes:
        At least 2 KPIs should be time-based and 1 should be cohort-based.
    """
    df = _prepare_df(data_input)

    # KPI 1: total revenue
    total_revenue = df["line_revenue"].sum()

    # KPI 2: monthly revenue trend (time-based)
    if "order_month" in df.columns:
        monthly_revenue = (
            df.groupby("order_month", as_index=False)["line_revenue"]
            .sum()
            .sort_values("order_month")
        )
    else:
        monthly_revenue = pd.DataFrame(columns=["order_month", "line_revenue"])

    # KPI 3: weekly order volume (time-based)
    if "order_week" in df.columns:
        weekly_order_volume = (
            df.groupby("order_week", as_index=False)["order_id"]
            .nunique()
            .rename(columns={"order_id": "order_count"})
            .sort_values("order_week")
        )
    else:
        weekly_order_volume = pd.DataFrame(columns=["order_week", "order_count"])

    # KPI 4: average order value by city (segmentation)
    order_values = (
        df.groupby(["order_id", "city"], as_index=False)["line_revenue"]
        .sum()
        .rename(columns={"line_revenue": "order_value"})
    )

    avg_order_value_by_city = (
        order_values.groupby("city", as_index=False)["order_value"]
        .mean()
        .sort_values("order_value", ascending=False)
    )

    # KPI 5: cohort revenue by registration month (cohort-based)
    if "registration_month" in df.columns:
        cohort_revenue = (
            df.groupby("registration_month", as_index=False)["line_revenue"]
            .sum()
            .sort_values("registration_month")
        )
    else:
        cohort_revenue = pd.DataFrame(columns=["registration_month", "line_revenue"])

    # extra helper KPI outputs
    revenue_by_category = (
        df.groupby("category", as_index=False)["line_revenue"]
        .sum()
        .sort_values("line_revenue", ascending=False)
    )

    return {
        "total_revenue": total_revenue,
        "monthly_revenue": monthly_revenue,
        "weekly_order_volume": weekly_order_volume,

        # expected by tests
        "avg_order_value": avg_order_value_by_city,
        "revenue_by_city": avg_order_value_by_city,

        # used in the project visuals / output
        "avg_order_value_by_city": avg_order_value_by_city,

        "cohort_revenue": cohort_revenue,
        "revenue_by_category": revenue_by_category,
        "analytics_df": df,
        "order_values": order_values
    }


def run_statistical_tests(data_input):
    """Run hypothesis tests to validate patterns in the data.

    Args:
        data_input: dict of DataFrames from extract_data() or a prepared DataFrame

    Returns:
        dict: mapping of test names to results (test statistic, p-value,
              interpretation)

    Notes:
        Run at least one test. Consider:
        - Does average order value differ across product categories?
        - Is there a significant trend in monthly revenue?
        - Do customer cities differ in purchasing behavior?
    """
    df = _prepare_df(data_input)
    results = {}

    # Test 1: ANOVA - Does order value differ across categories?
    category_order_values = (
        df.groupby(["order_id", "category"], as_index=False)["line_revenue"]
        .sum()
        .rename(columns={"line_revenue": "order_value"})
    )

    groups = []
    for category, subdf in category_order_values.groupby("category"):
        if len(subdf) >= 2:
            groups.append(subdf["order_value"].values)

    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        interpretation = (
            "Reject H0: average order value differs across product categories."
            if p_value < 0.05
            else "Fail to reject H0: no statistically significant difference in average order value across product categories."
        )

        results["anova_order_value_by_category"] = {
            "test": "One-way ANOVA",
            "hypothesis_null": "Average order value is equal across product categories.",
            "hypothesis_alt": "At least one product category has a different average order value.",
            "statistic": float(f_stat),
            "p_value": float(p_value),
            "interpretation": interpretation,
        }

    # Test 2: Chi-square - Are city and category associated?
    contingency = pd.crosstab(df["city"], df["category"])
    if contingency.shape[0] >= 2 and contingency.shape[1] >= 2:
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        interpretation = (
            "Reject H0: customer city and product category are associated."
            if p_value < 0.05
            else "Fail to reject H0: no statistically significant association between customer city and product category."
        )

        results["chi_square_city_category"] = {
            "test": "Chi-square test of independence",
            "hypothesis_null": "Customer city and product category are independent.",
            "hypothesis_alt": "Customer city and product category are associated.",
            "statistic": float(chi2),
            "p_value": float(p_value),
            "degrees_of_freedom": int(dof),
            "interpretation": interpretation,
        }

    return results


def create_visualizations(kpi_results, stat_results):
    """Create publication-quality charts for all 5 KPIs.

    Args:
        kpi_results: dict from compute_kpis()
        stat_results: dict from run_statistical_tests()

    Returns:
        None

    Side effects:
        Saves at least 5 PNG files to the output/ directory.
        Each chart should have a descriptive title stating the finding,
        proper axis labels, and annotations where appropriate.
    """
    sns.set_theme(style="whitegrid")
    sns.set_palette("colorblind")
    os.makedirs("output", exist_ok=True)

    # 1. Monthly revenue trend
    monthly = kpi_results["monthly_revenue"]
    if not monthly.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(monthly["order_month"], monthly["line_revenue"], marker="o")
        plt.title("Monthly Revenue Reveals How Marketplace Performance Changes Over Time")
        plt.xlabel("Order Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/monthly_revenue_trend.png", dpi=300)
        plt.close()

    # 2. Weekly order volume
    weekly = kpi_results["weekly_order_volume"]
    if not weekly.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(weekly["order_week"], weekly["order_count"], marker="o")
        plt.title("Weekly Order Volume Highlights Demand Fluctuations")
        plt.xlabel("Order Week")
        plt.ylabel("Number of Orders")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/weekly_order_volume.png", dpi=300)
        plt.close()

    # 3. Average order value by city
    aov_city = kpi_results["avg_order_value_by_city"]
    if not aov_city.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=aov_city, x="city", y="order_value")
        plt.title("Average Order Value Varies Across Customer Cities")
        plt.xlabel("City")
        plt.ylabel("Average Order Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/avg_order_value_by_city.png", dpi=300)
        plt.close()

    # 4. Cohort revenue
    cohort = kpi_results["cohort_revenue"]
    if not cohort.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(cohort["registration_month"], cohort["line_revenue"], marker="o")
        plt.title("Customer Registration Cohorts Contribute Unevenly to Revenue")
        plt.xlabel("Registration Cohort Month")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/cohort_revenue.png", dpi=300)
        plt.close()

    # 5. Revenue by category
    category_rev = kpi_results["revenue_by_category"]
    if not category_rev.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(data=category_rev, x="category", y="line_revenue")
        plt.title("Some Product Categories Generate More Revenue Than Others")
        plt.xlabel("Category")
        plt.ylabel("Revenue")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/revenue_by_category.png", dpi=300)
        plt.close()

    # 6. Statistical plot: boxplot by category
    analytics_df = kpi_results["analytics_df"]
    if not analytics_df.empty:
        category_order_values = (
            analytics_df.groupby(["order_id", "category"], as_index=False)["line_revenue"]
            .sum()
            .rename(columns={"line_revenue": "order_value"})
        )
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=category_order_values, x="category", y="order_value")
        plt.title("Order Value Distribution Differs by Product Category")
        plt.xlabel("Category")
        plt.ylabel("Order Value")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("output/order_value_by_category_boxplot.png", dpi=300)
        plt.close()

    # 7. Multi-panel figure
    if not analytics_df.empty:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        if not monthly.empty:
            axes[0, 0].plot(monthly["order_month"], monthly["line_revenue"], marker="o")
            axes[0, 0].set_title("Monthly Revenue Trend")
            axes[0, 0].set_xlabel("Month")
            axes[0, 0].set_ylabel("Revenue")
            axes[0, 0].tick_params(axis="x", rotation=45)

        if not weekly.empty:
            axes[0, 1].plot(weekly["order_week"], weekly["order_count"], marker="o")
            axes[0, 1].set_title("Weekly Order Volume")
            axes[0, 1].set_xlabel("Week")
            axes[0, 1].set_ylabel("Orders")
            axes[0, 1].tick_params(axis="x", rotation=45)

        if not aov_city.empty:
            sns.barplot(data=aov_city, x="city", y="order_value", ax=axes[1, 0])
            axes[1, 0].set_title("Average Order Value by City")
            axes[1, 0].set_xlabel("City")
            axes[1, 0].set_ylabel("AOV")
            axes[1, 0].tick_params(axis="x", rotation=45)

        heatmap_data = analytics_df.pivot_table(
            index="city",
            columns="category",
            values="line_revenue",
            aggfunc="sum",
            fill_value=0
        )
        sns.heatmap(heatmap_data, annot=True, fmt=".0f", cmap="viridis", ax=axes[1, 1])
        axes[1, 1].set_title("Revenue Heatmap by City and Category")
        axes[1, 1].set_xlabel("Category")
        axes[1, 1].set_ylabel("City")

        plt.tight_layout()
        plt.savefig("output/kpi_dashboard_multi_panel.png", dpi=300)
        plt.close()


def main():
    """Orchestrate the full analysis pipeline."""
    os.makedirs("output", exist_ok=True)

    engine = connect_db()
    data_dict = extract_data(engine)
    kpi_results = compute_kpis(data_dict)
    stat_results = run_statistical_tests(data_dict)
    create_visualizations(kpi_results, stat_results)

    print("\n=== KPI SUMMARY ===")
    print(f"Total Revenue: {kpi_results['total_revenue']:.2f}")

    print("\nMonthly Revenue:")
    print(kpi_results["monthly_revenue"].to_string(index=False))

    print("\nWeekly Order Volume:")
    print(kpi_results["weekly_order_volume"].to_string(index=False))

    print("\nAverage Order Value by City:")
    print(kpi_results["avg_order_value_by_city"].to_string(index=False))

    print("\nCohort Revenue:")
    print(kpi_results["cohort_revenue"].to_string(index=False))

    print("\nRevenue by Category:")
    print(kpi_results["revenue_by_category"].to_string(index=False))

    print("\n=== STATISTICAL TEST RESULTS ===")
    for test_name, result in stat_results.items():
        print(f"\n{test_name}")
        for key, value in result.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()