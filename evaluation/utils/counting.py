import re

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error


def squared_pearson_r2(y_true, y_pred):
    true = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    if true.size < 2:
        return 0.0
    true_centered = true - true.mean()
    pred_centered = pred - pred.mean()
    denominator = np.linalg.norm(true_centered) * np.linalg.norm(pred_centered)
    if denominator == 0 or not np.isfinite(denominator):
        return 0.0
    correlation = float(np.dot(true_centered, pred_centered) / denominator)
    if not np.isfinite(correlation):
        return 0.0
    return correlation * correlation


def parse_digit_response(x: str):
    from evaluation.utils.scoring import scalar_count
    value = scalar_count(x)
    return value if value is not None else -1


def calculate_counting_metrics(result_json, result_json_no_refusal):
    rr = (result_json["parsed_response"] == -1).mean()

    mape = mean_absolute_percentage_error(
        y_true=result_json["count"],
        y_pred=result_json["parsed_response"].replace(-1, 0)
    )
    if len(result_json_no_refusal):
        mape_no_refusal = mean_absolute_percentage_error(
            y_true=result_json_no_refusal["count"],
            y_pred=result_json_no_refusal["parsed_response"],
        )
        mae_no_refusal = mean_absolute_error(result_json_no_refusal["count"], result_json_no_refusal["parsed_response"])
        r2_no_refusal = squared_pearson_r2(result_json_no_refusal["count"], result_json_no_refusal["parsed_response"])
    else:
        mape_no_refusal = mae_no_refusal = r2_no_refusal = float('nan')
    mae = mean_absolute_error(result_json["count"], result_json["parsed_response"].replace(-1, 0))
    r2 = squared_pearson_r2(result_json["count"], result_json["parsed_response"].replace(-1, 0))
    return rr, (mape, mape_no_refusal), (mae, mae_no_refusal), (r2, r2_no_refusal)


def plot_scatter(df, ax=None):
    import seaborn as sns

    df = df.copy()
    df["parsed_response"] = df["parsed_response"].replace(-1, 0)
    df["Predicted Count"] = df["parsed_response"]
    df["True Count"] = df["count"]

    r2 = squared_pearson_r2(
        df["Predicted Count"], df["True Count"].replace(-1, 0)
    )

    max_limit = max(float(df["parsed_response"].max()), float(df["count"].max()), 1.0)
    cmap = sns.color_palette("blend:#397CB8,#CFE2F3", as_cmap=True)

    if not ax:
        fig, ax = plt.subplots(figsize=(6, 6))

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    sns.regplot(
        data=df,
        x="Predicted Count",
        y="True Count",
        ci=None,
        robust=True,
        truncate=False,
        scatter_kws=dict(marker="x", color="k"),
        line_kws=dict(color="red", linestyle="--"),
        ax=ax
    )


    if len(df) >= 3 and df["parsed_response"].nunique() > 1 and df["count"].nunique() > 1:
        try:
            sns.kdeplot(
                data=df,
                x="Predicted Count",
                y="True Count",
                levels=5,
                fill=True,
                alpha=0.6,
                cut=2,
                ax=ax,
                cmap=cmap,
                warn_singular=False,
            )
        except (ValueError, np.linalg.LinAlgError):
            pass
    ax.text(
        0.95, 0.95, f'$R^2 = {r2:.2f}$',
        verticalalignment='top', horizontalalignment='right',
        transform=ax.transAxes,
        color='black', fontsize=24
    )
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    return ax
