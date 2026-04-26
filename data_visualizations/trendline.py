from enum import Enum
import os
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, NullFormatter
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

CSV_PATH = r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-17/alldata-nobins.csv"
OUTPUT_PATH = (
    r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/3-15/"
)

class PlanktonType(Enum):
    # the variable names in the CSV file
    DIAT = "diatoms_hirata"
    DINO = "dinoflagellates_hirata"
    GREEN = "greenalgae_hirata"
    PRYM = "prymnesiophytes_hirata"
    CHLOR = "chlor_a"


COLORS = {
    PlanktonType.DIAT: (126 / 255, 33 / 255, 148 / 255),
    PlanktonType.DINO: (229 / 255, 155 / 255, 15 / 255),
    PlanktonType.GREEN: (0 / 255, 210 / 255, 0),
    PlanktonType.PRYM: (0 / 255, 95 / 255, 185 / 255),
    PlanktonType.CHLOR: (0 / 255, 0 / 255, 0 / 255),
}

NAMES = {
    PlanktonType.DIAT: "Diatoms",
    PlanktonType.DINO: "Dinoflagellates",
    PlanktonType.GREEN: "Green Algae",
    PlanktonType.PRYM: "Prymnesiophytes",
    PlanktonType.CHLOR: "Total Chlorophyll-A",
}


def read_csv(csv_path: str) -> pd.DataFrame:
    """Reads csv file as dataframe object"""
    # search for the row that contains all the variable names
    with open(csv_path, "r") as f:
        num_lines_before_header = 0
        for line in f:
            if line.startswith("date"):
                break
            num_lines_before_header += 1
        else:
            raise ValueError(f"Header row starting with 'date' not found in {csv_path}")

    # read in the csv file, starting with the row that contains all the variable names
    df = pd.read_csv(
        csv_path,
        sep=" ",
        skiprows=num_lines_before_header,
        header=0,
        comment="#",
        on_bad_lines="skip",
    )
    df.columns = [c.split(":")[0] for c in df.columns]

    return df


def extract_data(
    df: pd.DataFrame, plankton_type: PlanktonType
) -> List[Dict[str, np.ndarray]]:
    """Extract and align chlorophyll-related plankton data across all regions
    for a single plankton type.

    This function takes a DataFrame containing date-stamped plankton
    measurements and returns a list of region-specific dictionaries.
    Each dictionary contains NumPy arrays of the average, minimum, and
    maximum chlorophyll values aligned to a common date index. Missing
    values for dates with no observations in a region are filled with NaN.
    """
    data = []

    all_dates = df["date"].unique()
    n_dates = len(all_dates)
    date_to_idx = {
        date: i for i, date in enumerate(sorted(all_dates))
    }  # map each date to an index

    region = df["region"]
    for region_num in range(1, 5, 1):
        region_dict = {}

        mask = region == region_num
        region_df = df.loc[mask]  # only select data from that region

        full_avg = np.full(n_dates, np.nan)
        full_min = np.full(n_dates, np.nan)
        full_max = np.full(n_dates, np.nan)

        indices = (
            region_df["date"].map(date_to_idx).values
        )  # map each date from data in the region to an index

        full_avg[indices] = region_df[plankton_type.value + "_avg"].values
        full_min[indices] = region_df[plankton_type.value + "_min"].values
        full_max[indices] = region_df[plankton_type.value + "_max"].values

        region_dict["avg"] = np.array(full_avg)
        region_dict["min"] = np.array(full_min)
        region_dict["max"] = np.array(full_max)

        data.append(region_dict)
    return data


def max_range(data: List[Dict[str, np.ndarray]]) -> float:
    """
    Returns the largest difference between the maximum and average across all the regions
    """
    max_range = 0
    for region_num in range(1, 5):
        max_values = data[region_num - 1]["max"]
        avg_values = data[region_num - 1]["avg"]
        max_range = max(max_range, max(max_values) - np.nanmean(avg_values))
    print(max_range)
    return max_range


def prepare_subplots(
    data: List[Dict[str, np.ndarray]],
) -> Tuple[bool, plt.Figure, np.ndarray]:
    """
    Prepare a matplotlib figure and subplots for multi-region trendline plotting.

    This function determines whether a broken-axis layout is needed based on the
    range of the data. If the maximum values are significantly larger than the
    averages (max - avg > 2), a broken-axis layout is created with twice as many
    subplot rows to allow zooming on high values. Otherwise, a simple single-axis
    layout is used."""
    if (
        max_range(data) > 2
    ):  # if the max data points are much larger than the avg data points, split the y-axis
        broken_axis = True
        fig, axs = plt.subplots(
            4 * 2,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [1, 2] * 4, "hspace": 0.5},
            figsize=(20, 6),
        )
    else:
        broken_axis = False
        fig, axs = plt.subplots(
            4,
            1,
            sharex=True,
            gridspec_kw={"height_ratios": [1] * 4, "hspace": 0.2},
            figsize=(20, 6),
        )

    return broken_axis, fig, axs


def lighten_color(
    color: tuple[float, float, float], amount: float = 0.7
) -> tuple[float, float, float]:
    """
    Lighten an RGB color by a given amount.
    `amount` should be between 0 (no change) and 1 (white).
    """
    r, g, b = color
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b)


def style_y_axis(ax_top, ax_bottom):
    """
    Style the y-axes of a broken-axis plot for better readability and consistency.

    This function formats the y-axis of both the top and bottom subplots:
    - Sets y-axis labels to display one decimal place.
    - Sets evenly spaced y-ticks, ensuring inclusion of minimum and maximum values.
    - Highlights the top-most tick on the top subplot by making it bold and slightly larger.
    - Ensures that the bottom of the top axis (just above the axis break) is included in the ticks."""
    ax_top.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )  # show only 2 decimal places on the y axis
    ax_bottom.yaxis.set_major_formatter(
        FormatStrFormatter("%.1f")
    )  # show only 2 decimal places on the y axis}

    # bottom subplot
    bottom_ylim = ax_bottom.get_ylim()
    bottom_ticks = list(np.linspace(bottom_ylim[0], bottom_ylim[1], 3))
    ax_bottom.set_yticks(bottom_ticks)

    # top subplot
    top_ylim = ax_top.get_ylim()
    top_ticks = list(np.linspace(top_ylim[0], top_ylim[1], 3))
    ax_top.set_yticks(top_ticks)

    labels = ax_top.get_yticklabels()
    labels[-1].set_fontweight("bold")
    labels[-1].set_fontsize(8)


def find_outliers(data: np.ndarray, z_score_cutoff: float) -> np.ndarray:
    """
    Returns boolean array corresponding to outliers in data
    outliers defined as having an z_score greater than z_score_cutoff

    """
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    z_scores = (data - data_mean) / data_std
    outliers = z_scores > z_score_cutoff
    return outliers


def draw_outliers(ax, dates, avg_values, outliers, valid_points):
    """
    Highlight outlier points on a line plot that exceed the y-axis range."""
    ax.scatter(
        dates[outliers & valid_points],
        avg_values[outliers & valid_points],
        marker="^",  # upward triangle
        color="red",
        s=50,
        label="Outlier above axis",
        clip_on=False,
    )


def draw_broken_axis_markings(ax_top, ax_bottom):
    """
    Draw diagonal "break" marks on a pair of axes to indicate a broken y-axis.

    This function adds small diagonal lines at the corners of the top and bottom axes
    to visually represent a discontinuity in the y-axis, commonly used in plots where
    part of the y-axis is omitted."""
    # Add diagonal lines to indicate broken axis
    d = 0.015  # size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color="k", clip_on=False)
    ax_top.plot((-0.005, +d), (-0.1 - d, -0.1 + d), **kwargs)  # top-left diagonal
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    kwargs.update(transform=ax_bottom.transAxes)  # switch to bottom axes
    ax_bottom.plot((-0.005, +d), (1.1 - d, 1.1 + d), **kwargs)  # bottom-left diagonal
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal


def compute_segments(valid_indices, dates):
    """Compute continuous segments of valid data indices for plotting."""
    segments = []
    start = None
    for i, valid in enumerate(valid_indices):
        if valid and start is None:
            start = i
        elif not valid and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(dates)))
    return segments


def draw_trendline(
    ax_top: Axes,
    ax_bottom: Axes,
    data: List[Dict[str, np.ndarray]],
    dates: np.ndarray,
    broken_axis: bool,
    region_num: int,
    plankton_type: PlanktonType,
) -> None:
    """
    Draw a trendline with min-max range and missing data highlights on one or two subplots.

    This function visualizes the average, minimum, and maximum values of a dataset over time.
    It supports both a single axis view and a "broken axis" view with a top subplot
    zoomed on higher values and a bottom subplot showing averages and smaller values.

    Parameters
    ----------
    ax_top : matplotlib.axes.Axes
        The top Axes object for plotting (used in broken_axis mode or single-axis mode).
    ax_bottom : matplotlib.axes.Axes
        The bottom Axes object for plotting in broken_axis mode.
    data : list of dict
        A list where each element corresponds to a region and contains keys:
            'min' : array-like of minimum values
            'max' : array-like of maximum values
            'avg' : array-like of average values
    dates : array-like of datetime
        The dates corresponding to the data points.
    broken_axis : bool
        If True, create a broken-axis plot with a zoomed top subplot and a bottom subplot.
        If False, use a single axis (ax_top) for plotting the full range.
    region_num : int
        The region index (1-based) used to select the appropriate dataset from `data`.

    Behavior
    --------
    - Fills the area between min and max values with light colored shading.
    - Plots the average values as a colored trendline with circular markers.
    - Shades days with missing average values in gray.
    - When `broken_axis=True`, adds diagonal lines to indicate the axis break.
    - Formats y-axis labels to show two decimal places.
    - Hides x-axis tick labels for all but region 4.
    """

    min_values = data[region_num - 1]["min"]
    max_values = data[region_num - 1]["max"]
    avg_values = data[region_num - 1]["avg"]

    nan_indices = np.isnan(avg_values)
    valid_indices = ~nan_indices

    outliers = find_outliers(avg_values, 3)
    mask = valid_indices & ~outliers

    # calculate the min and max of the y axises
    bottom_max = max(avg_values[mask]) * 1.2
    top_max = max(max_values[mask]) * 1.2
    top_min = bottom_max
    # top_min=max(max(max_values[mask]) * 0.6,bottom_max)

    # set all outlier values to the max of the y axis
    avg_values[outliers] = bottom_max

    # calculate segments of valid indices
    # to be used to plot discontinuous fill between
    segments = compute_segments(valid_indices, dates)

    if broken_axis:
        # --- Top subplot: zoomed on max values ---
        for start, end in segments:
            ax_top.fill_between(
                dates[start:end],
                min_values[start:end],
                max_values[start:end],
                alpha=0.5,
                color=lighten_color(COLORS[plankton_type]),
            )  # draw the light colored min-max filling

        ax_top.set_ylim(top_min, top_max)
        # set y limit of top section of the broken_axis
        ax_top.spines["bottom"].set_visible(False)  # disable x-axis
        ax_top.tick_params(bottom=False)

        # --- Bottom subplot: zoomed on average and min ---
        for start, end in segments:
            ax_bottom.fill_between(
                dates[start:end],
                min_values[start:end],
                max_values[start:end],
                alpha=0.5,
                color=lighten_color(COLORS[plankton_type]),
            )
        ax_bottom.plot(
            dates[valid_indices],
            avg_values[valid_indices],
            color=COLORS[plankton_type],
            marker="o",
            markersize=3,
        )
        ax_bottom.set_ylim(0, bottom_max)
        ax_bottom.spines["top"].set_visible(False)

        draw_broken_axis_markings(ax_top, ax_bottom)

    else:
        # plot max min range
        for start, end in segments:
            ax_top.fill_between(
                dates[start:end],
                min_values[start:end],
                max_values[start:end],
                color=lighten_color(COLORS[plankton_type]),
                alpha=0.5,
            )

        # plot trendline
        ax_top.plot(
            dates[valid_indices],
            avg_values[valid_indices],
            label="Trendline",
            color=COLORS[plankton_type],
            marker="o",
            markersize=3,
            markerfacecolor="black",
        )

        ax_top.set_ylim(0, top_max)

    # plot gray bars for no data days
    for d in dates[nan_indices]:
        ax_top.axvspan(
            d - pd.Timedelta(days=3), d + pd.Timedelta(days=3), color="gray", alpha=0.3
        )
        if broken_axis:
            ax_bottom.axvspan(
                d - pd.Timedelta(days=3),
                d + pd.Timedelta(days=3),
                color="gray",
                alpha=0.3,
            )

    draw_outliers(ax_bottom, dates, avg_values, outliers, valid_indices)

    style_y_axis(ax_top, ax_bottom)

    # remove tick marks from all but the bottom axis of region 4
    if region_num != 4:
        ax_bottom.tick_params(axis="x", which="both", labelbottom=False, length=0)
        ax_top.tick_params(axis="x", which="both", labelbottom=False, length=0)
    else:
        if broken_axis:
            ax_top.tick_params(axis="x", which="both", labelbottom=False, length=0)


def label_month_year(ax_bottom: Axes, df: pd.DataFrame) -> None:
    """
    Customize the x-axis of a matplotlib plot to label specific months and years.

    This function sets minor ticks at the start of every month (without labels),
    highlights January with a longer, thicker tick, labels selected months
    (April, July, October) with abbreviated month names, and writes the year
    below the month labels."""
    # tick mark for every month
    ax_bottom.xaxis.set_minor_locator(
        mdates.MonthLocator()
    )  # sets tick mark at start of month
    ax_bottom.xaxis.set_minor_formatter(
        NullFormatter()
    )  # set no labels at the start of the month
    ax_bottom.tick_params(axis="x", which="minor", length=8)

    # label Jan and July
    ax_bottom.xaxis.set_major_locator(
        mdates.MonthLocator(bymonth=[4, 7, 10], bymonthday=15)
    )  # mid-July or mid-Jan
    ax_bottom.xaxis.set_major_formatter(
        mdates.DateFormatter("%b")
    )  # show "Jul" or "Jan"
    ax_bottom.tick_params(
        axis="x", which="major", length=0, pad=8
    )  # adjust tick mark appearance

    # makes the tick mark at January longer and bolder
    minor_ticks = ax_bottom.xaxis.get_minor_ticks()
    minor_locs = (
        ax_bottom.xaxis.get_minorticklocs()
    )  # positions in matplotlib date numbers
    for tick, loc in zip(minor_ticks, minor_locs):
        tick_date = mdates.num2date(loc)
        if tick_date.month == 1:
            tick.tick1line.set_markersize(18)  # bottom tick
            tick.tick1line.set_markeredgewidth(1.5)  # thickness

    # write the year underneath the month labels
    dates = pd.to_datetime(df["date"])
    years = list(range(dates.dt.year.min(), dates.dt.year.max() + 1))
    year_positions = [pd.Timestamp(year=y, month=6, day=15) for y in years]
    for pos, y in zip(year_positions, years):
        ax_bottom.annotate(
            str(y),
            xy=(pos, 0),  # x in data coordinates, y on x-axis line
            xycoords=("data", "axes fraction"),  # y fraction along axis, x in data
            xytext=(0, -20),  # offset in points below axis
            textcoords="offset points",  # offset units in points (screen units)
            ha="center",
            va="top",
            fontweight="bold",
            fontsize=8,
            clip_on=False,
        )
        
def generate_year_ticks(ax_bottom: Axes, df: pd.DataFrame) -> None:
    """
    Customize the x-axis of a matplotlib plot to
    draw minor tick marks at the start of each year
    """
    xmin = pd.to_datetime(df['date']).min()
    xmax = pd.to_datetime(df['date']).max()
    jan1_dates = pd.date_range(start=xmin, end=xmax, freq='YS')  # 'YS' = Year Start
    ax_bottom_secondary = ax_bottom.secondary_xaxis('bottom')
    ax_bottom_secondary.set_xticks(jan1_dates)
    ax_bottom_secondary.tick_params(length=6, width=1.5)
    ax_bottom_secondary.xaxis.set_major_formatter(NullFormatter())



def draw_legend(plt) -> None:
    """
    Add a custom figure-level legend to a matplotlib plot.

    This function creates a legend with predefined elements representing:
        - The minimum-maximum range (light colored patch)
        - A trendline (colored line)
        - Cloudy or insufficient data (gray patch)
        - Outlier above axis(red arrow)

    The legend is placed at the bottom center of the figure,
    slightly below the axes, and is arranged in 3 columns."""
    legend_elements = [
        Patch(
            facecolor=lighten_color(COLORS[plankton_type]),
            alpha=0.5,
            edgecolor="none",
            label="Min-Max Range",
        ),
        plt.Line2D([0], [0], color=COLORS[plankton_type], label="Trendline"),
        Patch(
            facecolor="gray",
            edgecolor="none",
            alpha=0.3,
            label="Cloudy/Insufficient Data",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="^",
            color="red",
            linestyle="None",
            markersize=6,
            label="Outlier above axis",
        ),
    ]

    plt.figlegend(
        handles=legend_elements, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.05)
    )


def generate_trendline(
    plankton_type: PlanktonType, df: pd.DataFrame, save: bool
) -> None:
    """Generate a multi-region trendline plot for a given plankton type.

    This function creates a 4-panel figure (one subplot per region) showing:
      - the minimum–maximum chlorophyll concentration range (shaded band),
      - the average concentration trendline,
      - gray intervals where no data exists for a region.
    """
    dates = np.sort(
        pd.to_datetime(df["date"].unique())
    )  # unique ordered dates along the x-axis
    data = extract_data(df, plankton_type)

    broken_axis, fig, axs = prepare_subplots(data)

    fig.supylabel("Chlorophyll-A Concentration (mg/m^3)", x=0.08, fontsize=10)
    fig.suptitle(f" {NAMES[plankton_type]}")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 6,
            "figure.titlesize": 12,
            "axes.titlesize": 8,
            "legend.fontsize": 10,
        }
    )

    for i, region_num in enumerate(range(1, 5)):
        if broken_axis:
            ax_top = axs[i * 2]
            ax_bottom = axs[i * 2 + 1]
        else:
            ax_top = ax_bottom = axs[i]

        ax_top.set_title(
            "Region " + str(region_num), x=1.05, y=-0.5 if broken_axis else 0.5
        )

        draw_trendline(
            ax_top, ax_bottom, data, dates, broken_axis, region_num, plankton_type
        )
        
        generate_year_ticks(ax_bottom, df)

    label_month_year(ax_bottom, df)
    draw_legend(plt)

    if save:
        fig.savefig(
            OUTPUT_PATH + f"trendline-{plankton_type.value}.png", bbox_inches="tight"
        )


##############
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("No CSV files found in the folder.")
    else:
        df = read_csv(CSV_PATH)

        for plankton_type in PlanktonType:
            generate_trendline(plankton_type, df, save=True)
