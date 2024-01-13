import pandas as pd
import plotly.graph_objects as go
import json

FILES = [
    ".cache/xlmr-normal-v2_b128+s64_intrinsic_results_u0.01.json",
    ".cache/xlmr-normal-v2_b256+s64_intrinsic_results_u0.01.json",
    ".cache/xlmr-normal-v2_b512+s64_intrinsic_results_u0.01.json",
]
NAME = "test"


def darken_color(color, factor):
    """Darken a given RGB color by a specified factor."""
    r, g, b, a = color
    r = max(int(r * factor), 0)
    g = max(int(g * factor), 0)
    b = max(int(b * factor), 0)
    return (r, g, b, a)


def plot_violin_from_json(files, name):
    # Prepare data
    data = {"score": [], "metric": [], "file": [], "x": [], "lang": []}
    spacing = 1.0  # Space between groups of metrics
    violin_width = 0.5 / len(files)  # Width of each violin

    # Base colors for each metric
    base_colors = {"u": (0, 123, 255, 0.6), "t": (40, 167, 69, 0.6), "punct": (255, 193, 7, 0.6)}
    color_darkening_factor = 0.8  # Factor to darken color for each subsequent file

    # Compute x positions and prepare colors for each file within each metric group
    x_positions = {}
    colors = {}
    for i, metric in enumerate(["u", "t", "punct"]):
        x_positions[metric] = {}
        colors[metric] = {}
        base_color = base_colors[metric]
        for j, file in enumerate(files):
            x_positions[metric][file] = i * spacing + spacing / (len(files) + 1) * (j + 1)
            colors[metric][file] = "rgba" + str(darken_color(base_color, color_darkening_factor**j))

    for file in files:
        with open(file, "r") as f:
            content = json.load(f)
            for lang, scores in content.items():
                for dataset, values in scores.items():
                    for metric in ["u", "t", "punct"]:
                        data["score"].append(values[metric])
                        data["metric"].append(metric)
                        data["file"].append(
                            file.split("/")[-1].split(".")[0]
                        )  # Use file base name without extension for legend
                        data["x"].append(x_positions[metric][file])  # Use computed x position
                        data["lang"].append(lang)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create violin plots
    fig = go.Figure()
    for metric in ["u", "t", "punct"]:
        metric_df = df[df["metric"] == metric]
        for file in files:
            file_name = file.split("/")[-1].split(".")[0]
            file_df = metric_df[metric_df["file"] == file_name]
            if not file_df.empty:
                fig.add_trace(
                    go.Violin(
                        y=file_df["score"],
                        x=file_df["x"],
                        name=file_name if metric == "u" else "",  # Only show legend for 'u' to avoid duplicates
                        legendgroup=file_name,
                        line_color=colors[metric][file],
                        box_visible=True,
                        meanline_visible=True,
                        width=violin_width,
                        points="all",
                        hovertext=file_df["lang"],
                    )
                )

    # Update layout
    # Calculate the center positions for each metric group
    center_positions = [sum(values.values()) / len(values) for key, values in x_positions.items()]
    fig.update_layout(
        title="Violin Plots of Scores by Metric",
        xaxis=dict(title="Metric", tickvals=center_positions, ticktext=["u", "t", "punct"]),
        yaxis_title="Scores",
        violingap=0,
        violingroupgap=0,
        violinmode="overlay",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black", size=14),
        title_x=0.5,
        legend_title_text="File",
        xaxis_showgrid=False,
        yaxis_showgrid=True,
        xaxis_zeroline=False,
        yaxis_zeroline=False,
        margin=dict(l=40, r=40, t=40, b=40),
    )

    # Update axes lines
    fig.update_xaxes(showline=True, linewidth=1, linecolor="gray")
    fig.update_yaxes(showline=True, linewidth=1, linecolor="gray")

    # Simplify and move legend to the top
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig.show()

    # Save the figure as HTML
    html_filename = f"{name}.html"
    fig.write_html(html_filename)
    print(f"Plot saved as {html_filename}")

    # Save the figure as PNG
    png_filename = f"{name}.png"
    fig.write_image(png_filename)
    print(f"Plot saved as {png_filename}")


plot_violin_from_json(FILES, NAME)
