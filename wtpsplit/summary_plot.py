import pandas as pd
import plotly.graph_objects as go
import json

FILES = [
    # "xlm-normal-p-v2_auxp0.3_n0.9_b512_s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-9l-v2_L_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.0_b512_s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-aux0.0001_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-aux0.1_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-auxp0.1_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-auxp0.1_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-auxp0.1_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.1_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-auxp0.1_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b8+s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b16+s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2-bs256_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b8+s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b16+s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_0.1auxp0.2_bs256_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b8+s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b16+s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_b512+s128_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b8+s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b16+6_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b32+16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b64+32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.2_bs256_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b128_s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b16_s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b256_s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b64_s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.3_n0.9_b8_s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.4_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.4_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.4_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.4_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_auxp0.4_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.3_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.5_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b16+s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.7_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b8+s4_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b16+s8_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b32+s16_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b64+s32_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-p-v2_n0.9_b512+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-v2_b128+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-v2_b256+s64_intrinsic_results_u0.01.json",
    # "xlmr-normal-v2_b512+s64_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b16+s8_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b32+s16_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b64+s32_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b128+s64_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b256+s64_intrinsic_results_u0.01.json",
    "xlmr-normal-noaux_b512+s64_intrinsic_results_u0.01.json",
    # "evaluation/evaluation_results/wtp-canine-s-3l-no-adapters_intrinsic_results.json",
    # "evaluation/evaluation_results/wtp-canine-s-3l_intrinsic_results.json"
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
    data = {"score": [], "metric": [], "file": [], "x": [], "lang": [], "dataset": []}
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
        file_path = ".cache/" + file if not file.startswith("evaluation") else file
        with open(file_path, "r") as f:
            print(file_path)
            content = json.load(f)
            for lang, scores in content.items():
                for dataset, values in scores.items():
                    for metric in ["u", "t", "punct"]:
                        data["score"].append(values[metric])
                        data["metric"].append(metric)
                        data["file"].append(file.split("/")[-1])  # Use file base name without extension for legend
                        data["x"].append(x_positions[metric][file])  # Use computed x position
                        data["lang"].append(lang)
                        data["dataset"].append(dataset)

    # Convert to DataFrame
    df = pd.DataFrame(data)

    # Create violin plots
    fig = go.Figure()
    for metric in ["u", "t", "punct"]:
        metric_df = df[df["metric"] == metric]
        for file in files:
            file_name = file.split("/")[-1]
            show_name = file_name.replace("xlmr-normal-", "").replace("_intrinsic_results_u0.01.json", "")
            file_df = metric_df[metric_df["file"] == file_name]
            if not file_df.empty:
                fig.add_trace(
                    go.Violin(
                        y=file_df["score"],
                        x=file_df["x"],
                        name=show_name if metric == "u" else "",  # Only show legend for 'u' to avoid duplicates
                        legendgroup=show_name,
                        line_color=colors[metric][file],
                        box_visible=True,
                        meanline_visible=True,
                        width=violin_width,
                        points="all",
                        hovertext=file_df["lang"] + " " + file_df["dataset"] + "<br>" + file_df["score"].astype(str),
                        hovertemplate=show_name + "<br>%{hovertext}<extra></extra>",
                        # hoveron="points+violins+kde"
                    )
                )

    # Update layout
    # Calculate the center positions for each metric group
    center_positions = [sum(values.values()) / len(values) for key, values in x_positions.items()]
    fig.update_layout(
        title=name,
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
    fig.update_yaxes(showline=True, linewidth=1, linecolor="gray", dtick=0.1, range=[0.01, 1.09])

    # Simplify and move legend to the top
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

    fig.show()

    # Save the figure as HTML
    html_filename = f"{name}.html"
    fig.write_html(html_filename)
    print(f"Plot saved as {html_filename}")

    # Save the figure as PNG
    # png_filename = f"{name}.png"
    # fig.write_image(png_filename)
    # print(f"Plot saved as {png_filename}")


plot_violin_from_json(FILES, NAME)
