from pathlib import Path

import sqlite3
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import numpy as np

# =========================
# CONFIGURATION
# =========================
DB_PATH = Path("/tmp/optuna.db")

STUDY_NAME = "loss_weights_optimization_split_pitch"
STUDY_ID = None  # optional override (set to an int to use directly)

# =========================
# LOAD DATA (ONCE)
# =========================
conn = sqlite3.connect(DB_PATH)

if STUDY_ID is None:
    study_row = pd.read_sql_query(
        "SELECT study_id FROM studies WHERE study_name = ?", conn, params=(STUDY_NAME,)
    )
    if study_row.empty:
        raise ValueError(f"Study '{STUDY_NAME}' not found.")
    STUDY_ID = int(study_row.iloc[0]["study_id"])

query = """
SELECT
    t.trial_id,
    p.param_name,
    p.param_value,
    v.objective,
    v.value
FROM trials t
JOIN trial_params p ON t.trial_id = p.trial_id
JOIN trial_values v ON t.trial_id = v.trial_id
WHERE t.state = 'COMPLETE'
  AND t.study_id = ?
"""

df_raw = pd.read_sql_query(query, conn, params=(STUDY_ID,))
conn.close()

# 1. Pivot objectives safely
obj_df = (
    df_raw.pivot_table(index="trial_id", columns="objective", values="value")
    .rename(columns={0: "obj0", 1: "obj1"})
    .reset_index()
)
# Safely enforce columns in case some trials/objectives are missing
obj_df = obj_df.reindex(columns=["trial_id", "obj0", "obj1"])

# 2. Extract and deduplicate parameters (Fixes the Cartesian Product bug)
df_params = df_raw[["trial_id", "param_name", "param_value"]].drop_duplicates()

# 3. Create a LONG dataframe (for the Boxplots)
df_long = df_params.merge(obj_df, on="trial_id")
params = sorted(df_params["param_name"].unique())

# 4. Create a WIDE dataframe (for the Parallel Coordinates Plot)
df_wide = df_params.pivot(
    index="trial_id", columns="param_name", values="param_value"
).reset_index()
df_wide = df_wide.merge(obj_df, on="trial_id")

# Clean up parameter names for a cleaner UI (optional but recommended for PCP)
clean_params = [p.replace("model.loss_weights.", "") for p in params]
df_wide = df_wide.rename(columns=dict(zip(params, clean_params)))

# =========================
# INITIAL SCATTER FIGURE
# =========================
scatter_fig = go.Figure(
    data=[
        go.Scatter(
            x=obj_df["obj0"],
            y=obj_df["obj1"],
            mode="markers",
            customdata=obj_df["trial_id"].tolist(),
            marker=dict(size=6),
        )
    ]
)

scatter_fig.update_layout(
    dragmode="select", xaxis_title="Objective 0", yaxis_title="Objective 1", title="Select trials"
)

# =========================
# DASH APP
# =========================
app = Dash(__name__)

app.layout = html.Div(
    [
        html.H3("Optuna Interactive Parameter Explorer"),
        html.Div(
            [
                html.Div(
                    dcc.Graph(id="scatter", figure=scatter_fig),
                    style={"width": "20%", "display": "inline-block"},
                ),
                html.Div(
                    dcc.Graph(id="parcoords"), style={"width": "80%", "display": "inline-block"}
                ),
            ]
        ),
        dcc.Graph(id="boxplot"),
    ]
)


# =========================
# CALLBACK
# =========================
@app.callback(
    [Output("boxplot", "figure"), Output("parcoords", "figure")],
    Input("scatter", "selectedData"),
)
def update_plots(selectedData):

    # --- 1. Handle Selection ---
    if not selectedData or not selectedData.get("points"):
        selected_ids = obj_df["trial_id"].values
    else:
        # Use pointIndex to map safely back to the dataframe
        selected_indices = [p["pointIndex"] for p in selectedData["points"]]
        selected_ids = obj_df.iloc[selected_indices]["trial_id"].values

    # --- 2. Filter Data ---
    filtered_long = df_long[df_long["trial_id"].isin(selected_ids)]
    filtered_wide = df_wide[df_wide["trial_id"].isin(selected_ids)]
    filtered_wide = filtered_wide.sort_values(by="obj0", ascending=False)

    # --- 3. Build Boxplot ---
    box_fig = go.Figure()
    for param in params:
        values = filtered_long.loc[filtered_long["param_name"] == param, "param_value"]
        if not values.empty:
            box_fig.add_trace(
                go.Box(
                    y=values, name=param.replace("model.loss_weights.", ""), boxpoints="outliers"
                )
            )

    box_fig.update_layout(
        title=f"Parameter distributions ({len(selected_ids)} selected trials)",
        yaxis_title="Parameter value",
        xaxis_title="Parameter",
        showlegend=False,
        yaxis_type="log",
        height=600,
    )

    # --- 4. Build Parallel Coordinates Plot (using graph_objects) ---
    pcp_dimensions = clean_params + ["obj1", "obj0"]

    dimensions_list = []
    for col in pcp_dimensions:
        if col in ["obj0", "obj1"]:
            # Linear axes for objectives (Keep dynamic full global range)
            full_min = df_wide[col].astype(float).min()
            full_max = df_wide[col].astype(float).max()

            dimensions_list.append(
                dict(label=col, values=filtered_wide[col].astype(float), range=[full_min, full_max])
            )
        else:
            # Logarithmic axes for parameters (Fixed range 0.001 to 10.0)
            raw_filtered_data = filtered_wide[col].astype(float)

            # Hardcoded bounds in log space
            full_min_log = np.log10(0.001)  # Evaluates to -3.0
            full_max_log = np.log10(10.0)  # Evaluates to 1.0

            # Create ticks exactly at powers of 10: [-3, -2, -1, 0, 1]
            tick_vals_log = np.arange(full_min_log, full_max_log + 1)

            # Format tick text nicely (0.001, 0.01, 0.1, 1, 10)
            tick_text = [f"{10**val:g}" for val in tick_vals_log]

            dimensions_list.append(
                dict(
                    label=col,
                    values=np.log10(raw_filtered_data),
                    range=[full_min_log, full_max_log],  # <--- Fixed range applied here
                    tickvals=tick_vals_log,
                    ticktext=tick_text,
                )
            )

    # Color values for the filtered lines
    color_vals = filtered_wide["obj0"].astype(float)
    print(color_vals)

    # Global bounds for the colorbar so it doesn't change
    full_color_min = df_wide["obj0"].astype(float).min()
    full_color_max = df_wide["obj0"].astype(float).max()

    pcp_fig = go.Figure(
        data=go.Parcoords(
            labelangle=30,
            labelside="bottom",
            line=dict(
                color=color_vals,
                colorscale="Plasma",
                showscale=True,
                cmin=full_color_min,
                cmax=full_color_max,
            ),
            dimensions=dimensions_list,
        )
    )

    pcp_fig.update_layout(
        title=f"Parallel Coordinates ({len(selected_ids)} selected trials)",
        margin=dict(l=50, r=50, t=60, b=100),
    )

    return box_fig, pcp_fig


# =========================
# RUN
# =========================
if __name__ == "__main__":
    app.run(debug=True)
