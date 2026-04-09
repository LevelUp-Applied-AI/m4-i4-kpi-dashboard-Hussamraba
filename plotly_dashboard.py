import os
import plotly.express as px

from analysis import connect_db, extract_data, compute_kpis


def build_dashboard():
    os.makedirs("output", exist_ok=True)

    engine = connect_db()
    data = extract_data(engine)
    kpis = compute_kpis(data)

    # Monthly Revenue
    fig1 = px.line(
        kpis["monthly_revenue"],
        x="order_month",
        y="line_revenue",
        title="Monthly Revenue Trend"
    )

    # Weekly Orders
    fig2 = px.line(
        kpis["weekly_order_volume"],
        x="order_week",
        y="order_count",
        title="Weekly Order Volume"
    )

    # AOV by City
    fig3 = px.bar(
        kpis["avg_order_value_by_city"],
        x="city",
        y="order_value",
        title="Average Order Value by City"
    )

    # Cohort Revenue
    fig4 = px.line(
        kpis["cohort_revenue"],
        x="registration_month",
        y="line_revenue",
        title="Cohort Revenue"
    )

    # Revenue by Category
    fig5 = px.bar(
        kpis["revenue_by_category"],
        x="category",
        y="line_revenue",
        title="Revenue by Category"
    )

    # Save all into one HTML
    with open("output/dashboard.html", "w") as f:
        f.write(fig1.to_html(full_html=False))
        f.write(fig2.to_html(full_html=False))
        f.write(fig3.to_html(full_html=False))
        f.write(fig4.to_html(full_html=False))
        f.write(fig5.to_html(full_html=False))

    print("Dashboard saved to output/dashboard.html")


if __name__ == "__main__":
    build_dashboard()