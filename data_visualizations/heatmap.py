from enum import Enum
from datetime import datetime
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm

CSV_PATH = r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-17/alldata-nobins.csv"
OUTPUT_PATH = (
    r"C:/Users/raine/Data/School/MIT/Freshman Year/UROP/CSV/lyze/samples/2-17/"
)


class PlanktonType(Enum):
    # the variable names in the CSV file
    DIAT = "diatoms_hirata"
    DINO = "dinoflagellates_hirata"
    GREEN = "greenalgae_hirata"
    PRYM = "prymnesiophytes_hirata"


NAMES = {
    PlanktonType.DIAT: "Diatoms",
    PlanktonType.DINO: "Dinoflagellates",
    PlanktonType.GREEN: "Green Algae",
    PlanktonType.PRYM: "Prymnesiophytes",
}

COLORS = {
    PlanktonType.DIAT: (126 / 255, 33 / 255, 148 / 255),
    PlanktonType.DINO: (255 / 255, 156 / 255, 17 / 255),
    PlanktonType.GREEN: (0 / 255, 210 / 255, 0),
    PlanktonType.PRYM: (0 / 255, 95 / 255, 185 / 255),
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


def extract_years(dates: list):
    """Extracts unique years from a list of date objects, in the order they appear"""
    years = []
    for date in dates:
        if date.year not in years:
            years.append(date.year)
    return years


def create_dates_dict(dates, years):
    """Organizes a list of date objects into a dictionary keyed by year,
    normalizing each date to a reference year (2000)
    """
    dates_dict = {year: [] for year in years}
    for date in dates:
        dates_dict[date.year].append(date.replace(year=2000))
    for year in dates_dict.keys():
        dates_dict[year].sort()
    return dates_dict


def extract_data(df: pd.DataFrame, all_data, dates_dict):
    """Populates a nested data structure with plankton concentration values, organized
    by region > plankton type > year."""
    for region_num in range(1, 5, 1):
        mask = df["region"] == region_num
        region_df = df.loc[
            mask
        ].copy()  # selects only the data corresponding to the region

        region_df["date"] = pd.to_datetime(region_df["date"])

        for year in dates_dict.keys():
            dates_in_year = dates_dict[year]
            date_to_idx = {
                date: i for i, date in enumerate(sorted(dates_in_year))
            }  # maps each date to an index

            year_mask = region_df["date"].dt.year == year
            year_df = region_df.loc[
                year_mask
            ]  # selects only the data from a specific region and year
            for plankton_type in PlanktonType:
                full_list = np.full(len(dates_in_year), np.nan)

                # convert all the data from the region and year to indices corresponding to all collected dates in the year
                normalized_dates = year_df["date"].apply(lambda d: d.replace(year=2000))
                indices = normalized_dates.map(date_to_idx).values

                full_list[indices] = year_df[plankton_type.value + "_avg"].values
                # populate full_list with all collected data for that region and year
                # nan datapoints represent dates in which data was collected for other regions, but not this region

                all_data[region_num][plankton_type][year].extend(full_list.tolist())


def luminance(datapoint, cmap, norm):
    """Calculate the perceived luminance of a datapoint based on a colormap.

    Parameters
    ----------
    datapoint : float
        The numeric value to evaluate (e.g., a data value in the heatmap).
    cmap : matplotlib.colors.Colormap
        A Matplotlib colormap used to convert the datapoint to an RGBA color.
    norm : matplotlib.colors.Normalize
        A normalization function that scales the datapoint into the range [0, 1]
        for the colormap.

    Returns
    -------
    float
        The perceived luminance of the datapoint, computed from the RGB channels
        using the standard Rec. 601 luma formula:
        L = 0.299 * R + 0.587 * G + 0.114 * B.
        Range is [0, 1].
    """
    rgba = cmap(norm(datapoint))
    r, g, b = rgba[:3]
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return luminance


def find_outliers(data, z_score_cutoff=3):
    """
    returns boolean array corresponding to outliers in data
    outliers defined as having an z_score greater than z_score_cutoff

    """
    data_mean = np.nanmean(data)
    data_std = np.nanstd(data)
    z_scores = (data - data_mean) / data_std
    outliers = z_scores > z_score_cutoff
    return outliers


def fill_data_grid(data_grid, years, dates_dict, all_data,region,plankton_type):
    """populates data_grid with the years"""
    for year_idx, year in enumerate(years):
        dates = pd.to_datetime(dates_dict[year])
        values = np.array(all_data[region][plankton_type][year])

        for month in range(1, 13):
            mask = dates.month == month
            valid = values[mask][~np.isnan(values[mask])]
            if len(valid) > 0:
                data_grid[year_idx, month - 1] = np.nanmean(
                    values[mask]
                )  # if there are multiple datapoints from a month, average them. Ignores nan values

def draw_data_grid(ax, data_grid, vmin, vmax, plankton_type):
    """
    Draw a heatmap of a 2D data grid with custom coloring and hatching for missing values.

    The heatmap uses a linear colormap ranging from off-white to the color
    associated with the given plankton type (defined in the global COLORS
    dictionary). Any NaN values in the data are overlaid with hatched rectangles
    to visually indicate missing data.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The matplotlib axes object on which to draw the heatmap.
    data_grid : numpy.ndarray
        A 2D array of numeric values to visualize. NaN entries are treated
        as missing data and displayed with hatching.
    vmin : float
        The minimum data value for colormap normalization.
    vmax : float
        The maximum data value for colormap normalization.
    plankton_type : str
        Key used to select the corresponding color from the global COLORS
        dictionary.

    Returns
    -------
    matplotlib.image.AxesImage
        The image object created by `ax.imshow`, which can be used for
        adding a colorbar or further customization.
    """
    # range of colors of heatmap is from off-white to color defined in COLORS dictionary
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "custom", ["#f7f7f7", COLORS[plankton_type]]
    )

    # hatching for missing data
    nan_mask = np.isnan(data_grid)
    patches = [Rectangle((j - 0.5, i - 0.5), 1, 1) for i, j in zip(*np.where(nan_mask))]
    collection = PatchCollection(
        patches,
        facecolor="none",
        edgecolor="0.9",
        linewidth=0.8,
        hatch="/",
    )
    ax.add_collection(collection)
    
    im = ax.imshow(
        data_grid,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        vmin=vmin,
        vmax=vmax
    )
    return im


def label_graph(ax, data_grid, fig, im, years, outliers_mask,region,plankton_type):
    """
    Annotates a heatmap of monthly chlorophyll-a data with values, outliers, and axis labels.

    Each cell of the heatmap is labeled with its data value. Outliers are highlighted in red,
    and text color for other cells is automatically chosen based on luminance for readability.
    The function also sets axis ticks, labels, title, a colorbar, and minor gridlines.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object on which the heatmap and labels are drawn.
    data_grid : numpy.ndarray
        2D array of chlorophyll-a values with shape (n_years, 12),
        where each row corresponds to a year
        and each column corresponds to a month.
    fig : matplotlib.figure.Figure
        The figure object containing the axes; used for adding the colorbar.
    im : matplotlib.image.AxesImage
        The image object returned by imshow() that represents the heatmap.
    years : list or numpy.ndarray
        List of years corresponding to the rows of `data_grid`.
    outliers_mask : numpy.ndarray
        Boolean 2D array with the same shape as `data_grid`;
        True where the data point is an outlier.

    Returns
    -------
    None
        The function modifies the axes and figure in place and does not return a value.
    """
    # plot the numerical value in the center of each rectangle in the heatmap
    for i in range(len(years)):
        for j in range(12):
            if not np.isnan(data_grid[i, j]):
                if outliers_mask[i, j]:
                    # mark outliers in red
                    #(needed because the color of the graph becomes fully saturated past VMAX)
                    text_color = "red"
                else:
                    text_color = (
                        "black"
                        if luminance(data_grid[i, j], im.cmap, im.norm) > 0.5
                        else "white"
                    )

                ax.text(
                    j,
                    i,
                    f"{data_grid[i, j]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                    fontweight="bold",
                )

    # label y-axis with years
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels(years)
    # label x-axis with months
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(
        [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]
    )

    ax.set_title(
        f"Monthly Mean Chlorophyll-a (mg m⁻³)\n"
        f"Region {region} – {NAMES[plankton_type]}",
        fontsize=16,
        pad=20,
    )
    ax.set_ylabel("Year", fontsize=14)
    ax.set_xlabel("Month", fontsize=14)

    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label(
        "Chlorophyll-a Concentration (mg m⁻³)", rotation=270, labelpad=20, fontsize=12
    )
    cbar.ax.tick_params(labelsize=9)

    ax.set_xticks(np.arange(-0.5, 12.5), minor=True)
    ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

    ax.grid(which="minor", color="#f0f0f0", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", size=0)


def create_legend(ax):
    """Adds a custom legend to the given Matplotlib Axes.

    The legend includes:
      - A hatched patch representing missing data (e.g., due to cloud cover)
      - A red patch indicating outliers
    """
    nan_legend = Patch(
        facecolor="none", hatch="//", edgecolor="0.6", label="No data (cloud cover)"
    )
    outlier_legend = Patch(
        facecolor="red", edgecolor="red", label="Red numbers = Outliers"
    )
    ax.legend(
        handles=[nan_legend, outlier_legend],
        loc="upper right",
        bbox_to_anchor=(1.1, 1.1),
        frameon=True,
        fontsize=10,
    )


def create_heat_map(region, plankton_type, all_data, dates_dict, save):
    """Creates a heat map of chlorophyll-a concentration attributed
    to a specific type of plankton in a specific region, over time"""
    fig, ax = plt.subplots(figsize=(14, 8))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    plt.tight_layout()

    years = sorted(dates_dict.keys())
    data_grid = np.full((len(years), 12), np.nan)
    fill_data_grid(data_grid, years, dates_dict, all_data,region,plankton_type)

    outliers_mask = find_outliers(data_grid, 3)
    vmin = np.nanpercentile(data_grid, 0) * 0.8
    vmax = np.nanmax(data_grid[~outliers_mask])

    im = draw_data_grid(ax, data_grid, vmin, vmax,plankton_type)

    label_graph(ax, data_grid, fig, im, years, outliers_mask,region,plankton_type)
    create_legend(ax)

    if save:
        print(f"Saved heatmap-{region}-{plankton_type}.png")
        fig.savefig(
            OUTPUT_PATH + f"heatmap-{region}-{plankton_type}.png",
            dpi=300,
            bbox_inches="tight",
        )


if __name__ == "__main__":
    if not os.path.exists(CSV_PATH):
        print("No CSV files found in the folder.")
    else:
        df = read_csv(CSV_PATH)  # read in csv to pandas dataframe object

        dates = [
            datetime.strptime(d, "%Y-%m-%d") for d in df["date"].unique()
        ]  # create list of unique dates

        years = extract_years(dates)  # obtain list of unique years
        dates_dict = create_dates_dict(
            dates, years
        )  # sort each month/day date object into its corresponding year in a dictionary

        # create empty dictionary, where keys are the numbers 1-4, representing the 4 regions
        # each region corresponds to a dictionary sorted by plankton type, and then year
        all_data = {}
        for i in range(1, 5):
            all_data[i] = {
                PlanktonType.DIAT: {year: [] for year in years},
                PlanktonType.DINO: {year: [] for year in years},
                PlanktonType.GREEN: {year: [] for year in years},
                PlanktonType.PRYM: {year: [] for year in years},
            }

        extract_data(
            df, all_data, dates_dict
        )  # populate allData with the information from the pandas dataframe

        for region in range(1, 5):
            for plankton_type in PlanktonType:
                create_heat_map(region, plankton_type, all_data, dates_dict, save=True)
