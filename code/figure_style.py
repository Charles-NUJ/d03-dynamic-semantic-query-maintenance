"""Shared publication style for D03 result figures."""

FIGURE_SIZE = (3.50, 5.35)
SUBPLOT_ADJUST = {
    "left": 0.185,
    "right": 0.985,
    "bottom": 0.090,
    "top": 0.820,
    "hspace": 0.430,
}

RC_PARAMS = {
    "font.family": "serif",
    "font.size": 8.0,
    "axes.labelsize": 8.2,
    "axes.titlesize": 8.4,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.fontsize": 6.2,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

# A semantic method keeps one label, gray level, line, and marker everywhere.
# Our method is darkest. Literature adaptations and controls use lighter grays.
METHOD_STYLES = {
    "signature_degree": {
        "label": "Signature-degree repair (ours)",
        "color": "0.05",
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.95,
        "markersize": 4.2,
        "filled": True,
    },
    "input_change_id": {
        "label": "Input-change/ID (ours)",
        "color": "0.30",
        "linestyle": "--",
        "marker": "s",
        "linewidth": 1.65,
        "markersize": 4.0,
        "filled": False,
    },
    "input_change_random": {
        "label": "Input-change random (ours)",
        "color": "0.44",
        "linestyle": ":",
        "marker": "D",
        "linewidth": 1.55,
        "markersize": 3.9,
        "filled": False,
    },
    "proxy_degree": {
        "label": "Proxy-degree [13]",
        "color": "0.60",
        "linestyle": "-.",
        "marker": "^",
        "linewidth": 1.50,
        "markersize": 4.0,
        "filled": False,
    },
    "link_degree": {
        "label": "Link-degree ablation (ours)",
        "color": "0.52",
        "linestyle": "-",
        "marker": "v",
        "linewidth": 1.50,
        "markersize": 4.0,
        "filled": False,
    },
    "svc_uniform": {
        "label": "SVC-style uniform [11]",
        "color": "0.78",
        "linestyle": ":",
        "marker": "X",
        "linewidth": 1.45,
        "markersize": 3.9,
        "filled": False,
    },
}


def label_for(style_key):
    return METHOD_STYLES[style_key]["label"]


def line_style_for(style_key):
    spec = METHOD_STYLES[style_key]
    color = spec["color"]
    return {
        "color": color,
        "linestyle": spec["linestyle"],
        "marker": spec["marker"],
        "linewidth": spec["linewidth"],
        "markersize": spec["markersize"],
        "markerfacecolor": color if spec["filled"] else "white",
        "markeredgecolor": color,
        "markeredgewidth": 0.9,
    }


def format_axis(axis, xlabel, ylabel):
    axis.set_xlabel(xlabel, labelpad=1.0)
    axis.set_ylabel(ylabel, labelpad=1.0)
    axis.grid(axis="y", color="0.88", linewidth=0.55, linestyle="-")
    axis.set_axisbelow(True)
    axis.tick_params(pad=1.0, top=False, right=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("0.15")
        spine.set_linewidth(0.75)
