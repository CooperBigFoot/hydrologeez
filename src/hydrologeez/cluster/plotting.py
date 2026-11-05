"""Publication-style maps and seasonal profiles in original physical units."""

import re
from colorsys import hsv_to_rgb
from pathlib import Path
from typing import cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from cartopy.mpl.geoaxes import GeoAxes
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_BASE_COLORS = ("#287A78", "#C17A32", "#766498", "#B85362", "#648447", "#397FA4")
_VARIABLE_NAMES = {
    "P": "Precipitation",
    "T": "Temperature",
    "PET": "Potential evapotranspiration",
    "Q": "Streamflow",
    "precipitation": "Precipitation",
    "mean_temperature": "Temperature",
    "oudin_pet": "Potential evapotranspiration",
}


def _cluster_colors(count: int) -> list[str]:
    # Preserve the approved six colors. Additional clusters use evenly advanced
    # golden-angle hues with alternating brightness, rather than cycling colors.
    colors = list(_BASE_COLORS[:count])
    for index in range(count - len(colors)):
        colors.append(mcolors.to_hex(hsv_to_rgb((0.04 + index * 0.61803398875) % 1, 0.65, 0.68 + 0.18 * (index % 2))))
    return colors


def _save(figure: Figure, directory: Path, name: str) -> None:
    FigureCanvasAgg(figure)
    for extension in ("png", "svg"):
        figure.savefig(directory / f"{name}.{extension}", dpi=200, facecolor="white", bbox_inches="tight")
    figure.clear()


def _profile(
    axis: Axes,
    members: FloatArray,
    centroid: FloatArray,
    color: str,
    cluster: int,
    variable: str,
    unit: str,
    limits: tuple[float, float],
) -> None:
    weeks = np.arange(1, 53)
    axis.plot(weeks, members.T, color="0.8", linewidth=0.6, alpha=0.65)
    axis.plot(weeks, centroid, color=color, linewidth=2)
    axis.text(0.03, 0.94, f"Cluster {cluster} · n={members.shape[0]}", transform=axis.transAxes, va="top")
    axis.set(xlim=(1, 52), ylim=limits, xticks=[1, 14, 27, 40, 52], xticklabels=["Jan", "Apr", "Jul", "Oct", "Dec"])
    display_unit = "°C" if unit == "degrees C" else unit
    axis.set_ylabel(f"{_VARIABLE_NAMES.get(variable, variable)} ({display_unit})")
    sns.despine(ax=axis)


def render_figures(
    directory: Path,
    members: FloatArray,
    centroids: FloatArray,
    labels: IntArray,
    variables: tuple[str, ...],
    units: tuple[str, ...],
    coordinates: FloatArray,
    scores: dict[int, float | None],
    search: bool,
    layout: str = "paired",
) -> None:
    """Save PNG/SVG diagnostics, a CONUS map, and one selected seasonal layout.

    Coordinates are longitude/latitude in degrees. Cluster labels are zero-based
    indices into centroids. Cartopy may download Natural Earth geographic data
    on first use; offline runs need those files in Cartopy's local data cache.
    """
    if layout not in {"paired", "separate"}:
        raise ValueError("layout must be 'paired' or 'separate'")
    directory.mkdir(parents=True, exist_ok=True)
    count = centroids.shape[0]
    colors = _cluster_colors(count)
    with (
        sns.plotting_context("paper", font_scale=1.3),
        sns.axes_style(
            "white",
            rc={"axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black", "text.color": "black"},
        ),
    ):
        if search:
            figure = Figure(figsize=(8, 6), layout="constrained")
            axis = figure.subplots()
            counts = sorted(scores)
            axis.plot(counts, [np.nan if scores[k] is None else scores[k] for k in counts], color="black", marker="o")
            axis.set(xlabel="Number of clusters", ylabel="Mean silhouette", xticks=counts)
            sns.despine(ax=axis)
            _save(figure, directory, "cluster_count_diagnostic")

        figure = Figure(figsize=(8, 6), layout="constrained")
        projection = ccrs.LambertConformal(central_longitude=-96, central_latitude=39)
        axis = cast(GeoAxes, figure.add_subplot(1, 1, 1, projection=projection))
        axis.set_extent([-125, -66, 24, 50], crs=ccrs.PlateCarree())
        axis.coastlines(resolution="110m", linewidth=0.6)
        axis.add_feature(cfeature.STATES.with_scale("110m"), edgecolor="0.65", linewidth=0.4, facecolor="none")
        for cluster, color in enumerate(colors):
            selected = labels == cluster
            axis.scatter(
                coordinates[selected, 0],
                coordinates[selected, 1],
                color=color,
                s=20,
                edgecolors="white",
                linewidths=0.25,
                transform=ccrs.PlateCarree(),
                label=f"Cluster {cluster} (n={int(selected.sum())})",
                zorder=3,
            )
        axis.legend(loc="lower left", fontsize="small", frameon=False)
        _save(figure, directory, "conus_cluster_map")

        limits: list[tuple[float, float]] = []
        for channel in range(len(variables)):
            low = float(min(members[:, :, channel].min(), centroids[:, :, channel].min()))
            high = float(max(members[:, :, channel].max(), centroids[:, :, channel].max()))
            margin = 0.08 * (high - low) if high > low else max(abs(low) * 0.08, 0.5)
            limits.append((low - margin, high + margin))
        if layout == "paired":
            figure = Figure(figsize=(8 * len(variables), 4 * count), layout="constrained")
            axes = figure.subplots(count, len(variables), squeeze=False, sharex=True, sharey="col")
            for cluster in range(count):
                for channel, variable in enumerate(variables):
                    _profile(
                        axes[cluster, channel],
                        members[labels == cluster, :, channel],
                        centroids[cluster, :, channel],
                        colors[cluster],
                        cluster,
                        variable,
                        units[channel],
                        limits[channel],
                    )
            _save(figure, directory, "seasonal_profiles_by_cluster")
        else:
            columns = min(2, count)
            rows = (count + columns - 1) // columns
            for channel, variable in enumerate(variables):
                figure = Figure(figsize=(8 * columns, 4 * rows), layout="constrained")
                axes = figure.subplots(rows, columns, squeeze=False, sharex=True, sharey=True)
                for cluster, axis in enumerate(axes.flat):
                    if cluster >= count:
                        axis.set_visible(False)
                        continue
                    _profile(
                        axis,
                        members[labels == cluster, :, channel],
                        centroids[cluster, :, channel],
                        colors[cluster],
                        cluster,
                        variable,
                        units[channel],
                        limits[channel],
                    )
                name = re.sub(r"[^a-z0-9]+", "_", _VARIABLE_NAMES.get(variable, variable).lower()).strip("_")
                _save(figure, directory, f"{name}_profiles")
