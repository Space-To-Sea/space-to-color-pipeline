from enum import Enum
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
import matplotlib.transforms as mtransforms
CSV_PATH = r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-17/alldata-nobins.csv"
OUTPUT_PATH = (
    r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-24/"
)


class PlanktonType(Enum):
    # the variable names in the CSV file
    DIAT = "diatoms_hirata"
    DINO = "dinoflagellates_hirata"
    GREEN = "greenalgae_hirata"
    PRYM = "prymnesiophytes_hirata"
    CHLOR = "chlor_a"


LEGEND_NAMES = {
    PlanktonType.DIAT: "Diatoms",
    PlanktonType.DINO: "Dinoflagellates",
    PlanktonType.GREEN: "Green Algae",
    PlanktonType.PRYM: "Prymnesiophytes",
    PlanktonType.CHLOR: "Other",
}

COLORS = {
    PlanktonType.DIAT: (126 / 255, 33 / 255, 148 / 255),
    PlanktonType.DINO: (255 / 255, 156 / 255, 17 / 255),
    PlanktonType.GREEN: (0 / 255, 210 / 255, 0),
    PlanktonType.PRYM: (0 / 255, 95 / 255, 185 / 255),
    PlanktonType.CHLOR: (255 / 255, 182 / 255, 193 / 255),
}

Y_START = 0.6
Y_STEP = 0.06
HEIGHTS = {
    PlanktonType.DIAT: Y_START,
    PlanktonType.DINO: Y_START + Y_STEP,
    PlanktonType.GREEN: Y_START + Y_STEP * 2,
    PlanktonType.PRYM: Y_START + Y_STEP * 3,
    PlanktonType.CHLOR: Y_START + Y_STEP * 4,
}


##############
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
            raise ValueError(f"Header row starting with 'date' not found in {CSV_PATH}")

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
    
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    return df


def extract_data(df: pd.DataFrame):
    """
    Extracts plankton concentration data from a DataFrame and
    organizes it by region and plankton type.

    Args:
        df (pd.DataFrame): Input DataFrame containing plankton measurements.

    Returns:
        dict: A dictionary mapping each `PlanktonType` to a list of 1D numpy arrays,
        one per region.
        Each array has length equal to the number of unique dates
        and contains the average chlorophyll-a
        concentrations for that plankton type in that region.
        Missing dates are represented as `NaN`.
    """
    all_data = {}
    all_data[PlanktonType.DIAT] = []
    all_data[PlanktonType.DINO] = []
    all_data[PlanktonType.GREEN] = []
    all_data[PlanktonType.PRYM] = []
    all_data[PlanktonType.CHLOR] = []

    all_dates = df["date"].unique()
    n_dates = len(all_dates)
    date_to_idx = {
        date: i for i, date in enumerate(all_dates)
    }  # maps each date to an index

    for region_num in range(1, 5, 1):
        mask = df["region"] == region_num
        region_df = df.loc[mask].copy()

        full_diat = np.full(n_dates, np.nan)
        full_dino = np.full(n_dates, np.nan)
        full_green = np.full(n_dates, np.nan)
        full_prym = np.full(n_dates, np.nan)
        full_chlor = np.full(n_dates, np.nan)

        indices = region_df["date"].map(date_to_idx).values

        full_diat[indices] = region_df[PlanktonType.DIAT.value + "_avg"].values
        full_dino[indices] = region_df[PlanktonType.DINO.value + "_avg"].values
        full_green[indices] = region_df[PlanktonType.GREEN.value + "_avg"].values
        full_prym[indices] = region_df[PlanktonType.PRYM.value + "_avg"].values
        full_chlor[indices] = region_df[PlanktonType.CHLOR.value + "_avg"].values

        all_data[PlanktonType.DIAT].append(full_diat)
        all_data[PlanktonType.DINO].append(full_dino)
        all_data[PlanktonType.GREEN].append(full_green)
        all_data[PlanktonType.PRYM].append(full_prym)
        all_data[PlanktonType.CHLOR].append(full_chlor)
    return all_data, all_dates


def find_outliers(data, z_score_cutoff):
    """
    Returns boolean array corresponding to outliers in data
    outliers defined as having an z_score greater than z_score_cutoff

    """
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    z_scores = (data - data_mean) / data_std
    outliers = z_scores > z_score_cutoff
    return outliers


def draw_outliers(data, ax, outliers, x_axis, y_max, region, bar_width=3, mask=None):
    """
    Draws outliers as a hollow bar with a red star on top, with numbers underneath
    """
    if mask is None:
        mask = slice(None)

    ax.bar(
        x_axis[outliers],
        y_max,
        bottom=0,
        width=bar_width,
        facecolor="none",
        edgecolor="gray",
        linewidth=0.5,
    )
    ax.scatter(
        x_axis[outliers],
        np.full(np.sum(outliers), y_max),
        color="darkred",
        alpha=0.8,
        s=100,
        marker="*",
        clip_on=False,
    )


def draw_bars(
    data, ax, outliers, x_axis, y_max, region, bar_width=3, mask=None, text=False
):
    """
    Draw a stacked bar chart of plankton data for a given region.

    Each plankton type is plotted as a stacked bar segment.
    Values are taken from `data` for the specified region
    and optionally filtered using `mask`.
    Missing values (NaNs) are indicated with a faint gray overlay,
    and outliers are excluded from plotting.

    text=True or False toggles whether numeric labels are drawn
    centered within sufficiently tall bar segments and positioned above small segments.

    Parameters
    ----------
    data : dict
        Mapping from PlanktonType to region-indexed numeric arrays.
        Each entry should be indexable as data[plankton_type][region - 1].
    ax : matplotlib.axes.Axes
        The axes object on which to draw the stacked bars.
    outliers : numpy.ndarray (bool)
        Boolean array indicating which positions should be excluded from
        plotting due to being outliers.
    x_axis : array-like
        X positions for the bars.
    y_max : float
        Maximum y-axis value, used for positioning small-value text labels
        and missing-data overlays.
    region : int
        1-based region index used to select the appropriate data slice.
    bar_width : float, optional
        Width of each bar. Default is 8.
    mask : slice or array-like of bool, optional
        Subset selector applied to the data and x_axis. If None, all values
        are included.
    text : bool, optional
        If True, numeric value labels are drawn on the bars.

    Returns
    -------
    None
        This function modifies the provided Axes object in place.
    """

    if mask is None:
        mask = slice(None)

    bottom = np.zeros(len(x_axis))
    # graph the data for each plankton type
    for plankton_type in PlanktonType:
        height = np.array(data[plankton_type][region - 1])[mask]
        hatch = ""
        if plankton_type == PlanktonType.CHLOR: 
        # to plot the remaining chlorophyll-a concentration,
        #subtract the diatoms, dinoflagellates, green algae, and pyrmseniophytes
        #from the total chlorophyll-a concentration
            height = height - bottom
            hatch = "///"
            height = np.maximum(height, 0) #replaces negative values of height with 0
        valid = ~np.isnan(height) & ~outliers
        missing = np.isnan(height)

        ax.bar(
            x_axis[valid],
            height[valid],
            bottom=bottom[valid],
            width=bar_width,
            hatch=hatch,
            color=COLORS[plankton_type],
            label=LEGEND_NAMES[plankton_type],
            linewidth=0,
        )

        if text:
            for i in np.where(valid)[0]:
                if height[i] > 0.3:
                    r, g, b = COLORS[plankton_type][:3]
                    ax.text(
                        x_axis[i],
                        bottom[i] + height[i] / 2,
                        f"{height[i]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="white"
                        if (0.299 * r + 0.587 * g + 0.114 * b) < 0.5
                        else "black",
                    )
                else:
                    value = data[plankton_type][region - 1][mask][i]
                    if plankton_type == PlanktonType.CHLOR:
                        value -= sum(
                            data[p][region - 1][mask][i]
                            for p in PlanktonType
                            if p != PlanktonType.CHLOR
                        )
                    ax.text(
                        x_axis[i],
                        y_max * HEIGHTS[plankton_type],
                        f"{value:.2f}",
                        ha="center",
                        va="bottom",
                        color=COLORS[plankton_type],
                        fontsize=5,
                    )
        bottom[valid] += height[valid]

        ax.bar(
            x_axis[missing], y_max, facecolor="lightgray", alpha=0.03, width=bar_width
        )


def calc_y_max(data, outliers):
    """
    Calculates y_max, which is 110% of the max value in data (ignoring outliers)
    """
    y_max = np.nanmax(data[~outliers])
    y_max *= 1.1
    return y_max


def label_months(ax, y_max):
    """
    Labels months Jan-Dec on the x axis
    """

    month_starts = {
        "Jan": 0,  # Jan 1
        "Feb": 31,  # Feb 1
        "Mar": 59,  # Mar 1
        "Apr": 90,  # Apr 1
        "May": 120,  # May 1
        "Jun": 151,  # Jun 1
        "Jul": 181,  # Jul 1
        "Aug": 212,  # Aug 1
        "Sep": 243,  # Sep 1
        "Oct": 273,  # Oct 1
        "Nov": 304,  # Nov 1
        "Dec": 334,  # Dec 1
    }

    months = list(month_starts.keys())
    starts = list(month_starts.values())

    for i, month in enumerate(months):
        start = starts[i]
        if i < len(months) - 1:
            end = starts[i + 1]
        else:
            end = 365  # December ends at day 365
        mid = (start + end) / 2  # midpoint of the month

        ax.text(mid, -0.25 * y_max, month, ha="center", va="top", fontsize=10)


def label_dates(ax, x_axis, dates):
    """Labels x-axis with year-month-day dates, rotated 90 degrees
    """
    ax.set_xticks(x_axis)
    ax.set_xticklabels([d.strftime("%Y-%m-%d") for d in dates], rotation=90, fontsize=6)


def style_yaxis(ax, y_max, n_ticks=5):
    """Sets the y-axis limit to y_max
    Draws and labels n_ticks on the y axis
    """
    ax.set_ylim(0, y_max)
    yticks = np.linspace(0, y_max, n_ticks)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))


def draw_legend(fig, ax):
    """
    Create and place a figure-level legend for the plankton bar chart.

    This function collects the legend handles and labels from the provided
    Axes object, appends an additional entry representing missing data
    ("No data (cloud cover)"), and places a combined legend at the figure
    level in the upper-right corner.
    """
    handles, labels = ax.get_legend_handles_labels()
    handles.append(
        Patch(facecolor="lightgray", alpha=0.2, label="No data (cloud cover)")
    )
    handles.append(Line2D([0], [0], marker='*', color='darkred', linestyle='None', markersize=12, label='Outlier'))
    fig.legend(
        handles=handles,
        labels=[h.get_label() for h in handles],
        loc="upper right",
        bbox_to_anchor=(1.00, 0.98),
    )
    
def add_month_year_labels(ax,x_axis,y_max,dates):
    # write the years and Jan/July underneath the x axis
    years = np.array([d.year for d in dates])
    months = np.array([d.month for d in dates])
    unique_years, first_indices = np.unique(years, return_index=True)
    # y offset factor
    y_offset_factor = -0.15
    month_offset = -0.03
    # Place text at first index of each year
    for idx, year in zip(first_indices, unique_years):
        ax.text(
            x_axis[idx],
            y_offset_factor * y_max,
            str(year),
            ha="center",
            va="top",
            rotation=0,
            fontsize=8
        )
        ax.text(
            x_axis[idx]-2,
            y_offset_factor/12 * y_max,
            "|",
            ha="center",
            va="top",
            rotation=0,
            fontsize=8
        )
    seen = set()
    for i, (year, month) in enumerate(zip(years, months)):
        if (year, month) not in seen and month in (4,8):
            seen.add((year, month))
            ax.text(
                x_axis[i]+5,
                month_offset * y_max,
                "Apr" if month==4 else "Aug",
                ha="center",
                va="top",
            )
                

def generate_bar_graph_smaller(df):
    """
    """
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    years = np.sort(df["date"].dt.year.unique())
    
    for i in range(0, len(years), 5):
        year_block = years[i:i+5]
    
        block_df = df[df["date"].dt.year.isin(year_block)]
    
        data, dates = extract_data(block_df)
    
        generate_bar_graph_longterm(
            data,
            dates,
            save=True,
            save_name=f"{year_block[0]}-{year_block[-1]}"
        )

def generate_bar_graph_longterm(data, dates, save, save_name=""):
    """Generates bar graph displaying plankton concentration in each region"""
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(hspace=0.5, left=0.15)

    fig.suptitle("Phytoplankton Chlorophyll-a Concentration", fontweight="bold",y=0.95)
    fig.supylabel("Chlorophyll-A Concentration (mg per m^3)", x=0.08,fontsize=10)
    
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 6,
        "axes.titlesize": 10,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "figure.titlesize": 12,
    })

    dates = pd.to_datetime(dates)
    x_spacing = 3
    x_axis = np.arange(1, len(dates) + 1) * x_spacing

    # graphs the data for each region
    for region in range(1, 5, 1):
        ax = fig.add_subplot(4, 1, region)
        ax.set_title("Region " + f"{region}")

        outliers = find_outliers(data[PlanktonType.CHLOR][region - 1], 3)
        y_max = calc_y_max(data[PlanktonType.CHLOR][region - 1], outliers)

        style_yaxis(ax, y_max, 5)

        # bold the top label of each y-axis
        labels = ax.get_yticklabels()
        labels[-1].set_fontweight("bold")
        labels[-1].set_fontsize(8)

        ax.xaxis.set_visible(False)

        add_month_year_labels(ax,x_axis,y_max,dates)

        bar_width = 3
        draw_bars(
            data, ax, outliers, x_axis, y_max, region, bar_width=bar_width, text=False
        )
        draw_outliers(
            data, ax, outliers, x_axis, y_max, region, bar_width=bar_width, mask=None
        )

    draw_legend(fig, ax)

    if save:
        fig.savefig(OUTPUT_PATH + "bargraph"+save_name+".png", bbox_inches="tight")


##############
if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("No CSV files found in the folder.")
    else:
        df = read_csv(CSV_PATH)
        
        generate_bar_graph_smaller(df)

        # data, dates = extract_data(df)

        # generate_bar_graph_longterm(data,dates,save=True)