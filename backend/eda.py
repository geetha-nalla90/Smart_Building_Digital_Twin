"""Create a reproducible dataset profile, report, and EDA visualizations."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from backend.config import EDA_DIR, get_dataset_path
from backend.data_pipeline import TARGET_COLUMN, load_dataset


def generate_eda(dataset_path: Path | str | None = None, output_dir: Path = EDA_DIR) -> Path:
    path = Path(dataset_path) if dataset_path else get_dataset_path()
    data = load_dataset(path).sort_values("Date").reset_index(drop=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    fig, axis = plt.subplots(figsize=(9, 5))
    sns.histplot(data[TARGET_COLUMN], kde=True, ax=axis)
    axis.set(title="Total energy consumption distribution", xlabel="Energy consumption")
    fig.tight_layout()
    fig.savefig(output_dir / "energy_distribution.png", dpi=150)
    plt.close(fig)

    for column, filename, title in [
        ("Occupancy", "occupancy_vs_energy.png", "Occupancy vs energy"),
        ("Temperature", "temperature_vs_energy.png", "Temperature vs energy"),
    ]:
        fig, axis = plt.subplots(figsize=(8, 5))
        sns.scatterplot(
            data=data,
            x=column,
            y=TARGET_COLUMN,
            alpha=0.6,
            s=28,
            ax=axis,
        )
        axis.set(title=title)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=150)
        plt.close(fig)

    load_columns = ["Lighting_Load", "HVAC_Load", "Computer_Load", "Other_Load"]
    fig, axis = plt.subplots(figsize=(9, 5))
    data[load_columns].mean().sort_values().plot.barh(ax=axis)
    axis.set(title="Average individual load", xlabel="Average load")
    fig.tight_layout()
    fig.savefig(output_dir / "individual_load_analysis.png", dpi=150)
    plt.close(fig)

    numeric = data.select_dtypes(include="number")
    fig, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=axis)
    axis.set_title("Numeric feature correlation")
    fig.tight_layout()
    fig.savefig(output_dir / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    by_date = data.assign(Date=pd.to_datetime(data["Date"])).set_index("Date")
    monthly = by_date[TARGET_COLUMN].resample("MS").mean()
    fig, axes = plt.subplots(2, 1, figsize=(11, 8))
    monthly.plot(ax=axes[0], marker="o")
    axes[0].set(title="Monthly average energy", ylabel="Energy")
    data.groupby("Hour")[TARGET_COLUMN].mean().plot(ax=axes[1], marker="o")
    axes[1].set(title="Average energy by hour", xlabel="Hour of day", ylabel="Energy")
    fig.tight_layout()
    fig.savefig(output_dir / "time_based_analysis.png", dpi=150)
    plt.close(fig)

    numeric_q1 = numeric.quantile(0.25)
    numeric_q3 = numeric.quantile(0.75)
    iqr = numeric_q3 - numeric_q1
    outliers = (numeric.lt(numeric_q1 - 1.5 * iqr) | numeric.gt(numeric_q3 + 1.5 * iqr)).sum()
    report_lines = [
        f"# Exploratory Data Analysis: {path.name}",
        "",
        "## Dataset overview",
        f"- Rows: {len(data)}",
        f"- Columns: {len(data.columns)}",
        f"- Date range: {pd.to_datetime(data['Date']).min().date()} to {pd.to_datetime(data['Date']).max().date()}",
        f"- Unique Building_ID values: {data['Building_ID'].nunique()}",
        f"- Exact duplicate rows: {data.duplicated().sum()}",
        "",
        "## Data quality",
        "### Column types",
        data.dtypes.to_frame("dtype").to_markdown(),
        "",
        "### Missing values",
        data.isna().sum().to_frame("missing").to_markdown(),
        "",
        "### IQR outlier counts (exploratory flags; rows are retained)",
        outliers.to_frame("outlier_count").to_markdown(),
        "",
        "## Numeric summary",
        data.describe().transpose().to_markdown(),
        "",
        "## Interpretation",
        "- Building_ID is unique for every row and is excluded from model features to avoid memorizing identifiers.",
        "- Date is transformed into month, day-of-week, day-of-year, and cyclic hour/month features.",
        "- The small IQR-flagged values are retained because clipping would alter valid energy/load observations.",
        "- Dataset rows are chronological observations; model validation uses the later 20% as a holdout.",
        "",
        "## Visualizations",
    ]
    report_lines.extend(f"- [{file.stem}](./{file.name})" for file in sorted(output_dir.glob("*.png")))
    report_path = output_dir / "eda_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved EDA report: {report_path}")
    return report_path


if __name__ == "__main__":
    generate_eda()
