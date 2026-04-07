import ast
import json
import pathlib
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **_kwargs):
        _ = _kwargs
        return iterable


import libs.GeneralCameraClass as CamForm
import libs.GeneralStageClass as StageForm

LINE_DIRECTION_OPTIONS = {"+x", "-x", "+y", "-y"}


def _as_float_pair(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        array = value.astype(float).reshape(-1)
    elif isinstance(value, (list, tuple)):
        array = np.asarray(value, dtype=float).reshape(-1)
    elif isinstance(value, str):
        text = value.strip()
        if text.lower() == "current":
            raise ValueError("current is not a coordinate pair")
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            if "," in text:
                parsed = [part.strip() for part in text.split(",")]
            else:
                raise ValueError(
                    f"Could not parse coordinate pair from {value!r}"
                ) from None
        return _as_float_pair(parsed)
    else:
        raise ValueError(f"Unsupported coordinate value {value!r}")

    if array.size != 2:
        raise ValueError(f"Expected two coordinates, got {array!r}")
    return array.astype(float)


def _build_offsets(search_width: float, step_size: float) -> np.ndarray:
    if search_width <= 0:
        raise ValueError("search-width must be positive")
    if step_size <= 0:
        raise ValueError("step-size must be positive")

    count = max(3, int(round(search_width / step_size)) + 1)
    if count % 2 == 0:
        count += 1
    half_width = search_width / 2.0
    return np.linspace(-half_width, half_width, count)


@dataclass(frozen=True)
class LineModel:
    line_direction: str

    def __post_init__(self) -> None:
        if self.line_direction not in LINE_DIRECTION_OPTIONS:
            raise ValueError(
                f"line-direction must be one of {sorted(LINE_DIRECTION_OPTIONS)}, got {self.line_direction!r}"
            )

    def axis_and_sign(self) -> tuple[str, int]:
        if self.line_direction == "+x":
            return "x", 1
        if self.line_direction == "-x":
            return "x", -1
        if self.line_direction == "+y":
            return "y", 1
        return "y", -1

    def fit(self, points: list[np.ndarray]) -> tuple[float, float]:
        axis, _ = self.axis_and_sign()
        coordinates = np.asarray(points, dtype=float)
        if coordinates.shape[0] < 2:
            raise ValueError("Need at least two points to fit a line")

        try:
            if axis == "x":
                x_values = coordinates[:, 0]
                y_values = coordinates[:, 1]
                if np.unique(x_values).size < 2:
                    return 0.0, float(np.mean(y_values))
                slope, intercept = np.polyfit(x_values, y_values, 1)
            else:
                y_values = coordinates[:, 1]
                x_values = coordinates[:, 0]
                if np.unique(y_values).size < 2:
                    return 0.0, float(np.mean(x_values))
                slope, intercept = np.polyfit(y_values, x_values, 1)
        except np.linalg.LinAlgError:
            if axis == "x":
                return 0.0, float(np.mean(coordinates[:, 1]))
            return 0.0, float(np.mean(coordinates[:, 0]))

        return float(slope), float(intercept)

    def predict_next_point(
        self, points: list[np.ndarray], dot_spacing: float
    ) -> np.ndarray:
        axis, sign = self.axis_and_sign()
        last_point = np.asarray(points[-1], dtype=float)

        if len(points) == 1:
            step = np.array([sign * dot_spacing, 0.0], dtype=float)
            if axis == "y":
                step = np.array([0.0, sign * dot_spacing], dtype=float)
            return last_point + step

        slope, intercept = self.fit(points)
        if axis == "x":
            next_x = float(last_point[0] + sign * dot_spacing)
            next_y = slope * next_x + intercept
            return np.array([next_x, next_y], dtype=float)

        next_y = float(last_point[1] + sign * dot_spacing)
        next_x = slope * next_y + intercept
        return np.array([next_x, next_y], dtype=float)


def _ensure_directory(path: pathlib.Path) -> pathlib.Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _plot_scan_heatmap(
    score_matrix: np.ndarray,
    grid_points: np.ndarray,
    best_index: tuple[int, int],
    title: str,
    save_path: pathlib.Path,
    found_points: list[np.ndarray] | None = None,
) -> None:
    fig, (ax_map, ax_path) = plt.subplots(
        1, 2, figsize=(11, 5), constrained_layout=True
    )

    image = ax_map.imshow(
        score_matrix,
        origin="lower",
        interpolation="nearest",
        cmap="viridis",
    )
    ax_map.scatter(best_index[1], best_index[0], c="red", s=60, marker="x")
    fig.colorbar(image, ax=ax_map, label="metric (lower is better)")
    ax_map.set_title(title)
    ax_map.set_xlabel("scan x index")
    ax_map.set_ylabel("scan y index")

    flat_points = grid_points.reshape(-1, 2)
    ax_path.plot(
        flat_points[:, 0], flat_points[:, 1], linestyle="--", color="0.7", linewidth=1
    )
    if found_points:
        found_array = np.asarray(found_points, dtype=float)
        ax_path.plot(found_array[:, 0], found_array[:, 1], marker="o", color="tab:blue")
        for idx, point in enumerate(found_array, start=1):
            ax_path.text(point[0], point[1], str(idx), fontsize=8)
    ax_path.set_title("Stage positions")
    ax_path.set_xlabel("x")
    ax_path.set_ylabel("y")
    ax_path.set_aspect("equal", adjustable="datalim")

    fig.savefig(save_path)
    plt.close(fig)


@dataclass
class ScanResult:
    center: np.ndarray
    grid_points: np.ndarray
    metric_flux: np.ndarray
    metric_corr: np.ndarray
    metric_weighted: np.ndarray
    frames: np.ndarray
    best_index: tuple[int, int]
    best_point: np.ndarray
    best_score: float


class FPM_Finder:
    def __init__(self, host: str = "mimir", port: int = 5555):
        self.CamObj = CamForm.GeneralCameraObject()
        self.StageObj = StageForm.GeneralStageObject(host=host, port=port)

    def _get_frame(self, beam: int) -> np.ndarray:
        return self.CamObj.GetFrame(ibeam=beam)

    def _get_stage_position(self, stage: str, beam: int) -> float:
        return float(self.StageObj.Get_pos(stage=stage, beam=beam))

    def get_positions(self, beam: int) -> np.ndarray:
        return np.array(
            [
                self._get_stage_position("BMX", beam),
                self._get_stage_position("BMY", beam),
            ],
            dtype=float,
        )

    def set_positions(self, beam: int, position: np.ndarray) -> None:
        self._set_stage_position("BMX", beam, float(position[0]))
        self._set_stage_position("BMY", beam, float(position[1]))

    def _set_stage_position(self, stage: str, beam: int, pos: float) -> None:
        try:
            self.StageObj.Set_pos(stage=stage, beam=beam, pos=float(pos))
        except TypeError:
            self.StageObj.Set_pos(stage, beam, float(pos))

    def _frame_score(self, frame: np.ndarray, beam_center: np.ndarray) -> float:
        centre = [float(beam_center[0]), float(beam_center[1])]
        return float(
            self.CamObj.GetRelativePower(
                frame=frame,
                centre=centre,
                x_half_width=32,
                y_half_width=32,
                show_plot=False,
            )
        )

    def _build_reference_metrics(
        self, beam: int
    ) -> tuple[np.ndarray, float, np.ndarray]:
        self._set_stage_position("BMY", beam, self.no_feature_pos[1])
        self._set_stage_position("BMX", beam, self.no_feature_pos[0])

        ref_frame = self._get_frame(beam)
        ref_center = np.asarray(self.CamObj.FindMaxValueOnFrame(ref_frame), dtype=float)
        ref_flux = float(
            self.CamObj.GetRelativePower(
                frame=ref_frame,
                centre=[float(ref_center[0]), float(ref_center[1])],
                x_half_width=5,
                y_half_width=5,
                show_plot=False,
            )
        )

        ref_corr_temp = ref_frame.astype(float).ravel()
        ref_corr_temp = ref_corr_temp - np.mean(ref_corr_temp)
        return ref_center, ref_flux, ref_corr_temp

    def _resolve_start_center(self, start_center: str, beam: int) -> np.ndarray:
        if start_center.strip().lower() == "current":
            return np.array(
                [
                    self._get_stage_position("BMX", beam),
                    self._get_stage_position("BMY", beam),
                ],
                dtype=float,
            )
        return _as_float_pair(start_center)

    def _scan_center(
        self,
        beam: int,
        center: np.ndarray,
        search_width: float,
        step_size: float,
        ref_center: np.ndarray,
        ref_flux: float,
        ref_corr_temp: np.ndarray,
    ) -> ScanResult:

        offsets = _build_offsets(search_width, step_size)
        half_width = search_width / 2.0
        grid_points = np.asarray(
            self.StageObj.rasterScanSnakePattern(
                StartX=float(center[0]),
                StartY=float(center[1]),
                StepAwayFromStartX=float(half_width),
                StepAwayFromStartY=float(half_width),
                StepCountX=int(offsets.size),
                StepCountY=int(offsets.size),
            ),
            dtype=float,
        )
        count_y, count_x, _ = grid_points.shape
        metric_flux = np.zeros((count_y, count_x), dtype=float)
        metric_corr = np.zeros((count_y, count_x), dtype=float)
        metric_weighted = np.zeros((count_y, count_x), dtype=float)
        frames = None

        scan_sequence = ((iy, ix) for iy in range(count_y) for ix in range(count_x))
        progress = tqdm(
            scan_sequence,
            total=count_y * count_x,
            desc=f"Beam {beam} local scan",
            unit="pt",
        )
        for iy, scan_x_idx in progress:
            xpos = grid_points[iy, scan_x_idx, 0]
            ypos = grid_points[iy, scan_x_idx, 1]
            self._set_stage_position("BMX", beam, xpos)
            self._set_stage_position("BMY", beam, ypos)
            frame = self._get_frame(beam)
            if frames is None:
                frames = np.zeros(
                    (count_y, count_x, frame.shape[0], frame.shape[1]),
                    dtype=frame.dtype,
                )
            frames[iy, scan_x_idx] = frame

            flux = self._frame_score(frame, ref_center)
            frame_corr_temp = frame.astype(float).ravel()
            frame_corr_temp = frame_corr_temp - np.mean(frame_corr_temp)

            denom = np.sqrt(np.sum(frame_corr_temp**2) * np.sum(ref_corr_temp**2))
            if denom == 0:
                corr = 1.0
            else:
                corr = float(np.sum(frame_corr_temp * ref_corr_temp) / denom)

            flux_norm = flux / (ref_flux + 1e-12)
            flux_floor = 0.5
            lam = 5.0
            penalty = lam * max(0.0, flux_floor - flux_norm) ** 2
            weighted = corr + penalty

            metric_flux[iy, scan_x_idx] = flux
            metric_corr[iy, scan_x_idx] = corr
            metric_weighted[iy, scan_x_idx] = weighted

        best_flat = int(np.argmin(metric_corr))
        best_index: tuple[int, int] = (
            int(best_flat // metric_corr.shape[1]),
            int(best_flat % metric_corr.shape[1]),
        )
        best_point = grid_points[best_index]
        best_score = float(metric_weighted[best_index])

        if frames is None:
            frames = np.empty((0, 0, 0, 0))

        return ScanResult(
            center=center,
            grid_points=grid_points,
            metric_flux=metric_flux,
            metric_corr=metric_corr,
            metric_weighted=metric_weighted,
            frames=frames,
            best_index=best_index,
            best_point=best_point,
            best_score=best_score,
        )

    def _plot_final_summary(
        self,
        found_points: list[np.ndarray],
        line_model: LineModel,
        save_path: pathlib.Path,
    ) -> None:
        fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
        if found_points:
            points = np.asarray(found_points, dtype=float)
            ax.plot(points[:, 0], points[:, 1], marker="o", color="tab:blue")
            for idx, point in enumerate(points, start=1):
                ax.text(point[0], point[1], str(idx), fontsize=8)

            if len(found_points) >= 2:
                axis, _ = line_model.axis_and_sign()
                slope, intercept = line_model.fit(found_points)
                if axis == "x":
                    x_values = np.linspace(
                        np.min(points[:, 0]), np.max(points[:, 0]), 100
                    )
                    y_values = slope * x_values + intercept
                    ax.plot(x_values, y_values, color="tab:orange", linestyle="--")
                else:
                    y_values = np.linspace(
                        np.min(points[:, 1]), np.max(points[:, 1]), 100
                    )
                    x_values = slope * y_values + intercept
                    ax.plot(x_values, y_values, color="tab:orange", linestyle="--")

        ax.set_title("Focal mask dot positions")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal", adjustable="datalim")
        fig.savefig(save_path)
        plt.close(fig)

    def run(
        self,
        beam: int,
        line_direction: str,
        start_center: str,
        step_size: float,
        search_width: float,
        dot_spacing: float,
        n_dots: int,
        detection_threshold: float,
        save_path: pathlib.Path,
        out_file: str,
    ) -> dict:
        if beam not in {1, 2, 3, 4}:
            raise ValueError("beam must be one of 1, 2, 3, 4")

        save_path = _ensure_directory(save_path)
        start_center_xy = self._resolve_start_center(start_center, beam)
        self.no_feature_pos = np.array(
            [start_center_xy[0], start_center_xy[1]], dtype=float
        )
        line_model = LineModel(line_direction)

        found_points: list[np.ndarray] = []
        found_scores: list[float] = []
        scan_records: list[dict] = []
        corr_matrices: list[np.ndarray] = []
        flux_matrices: list[np.ndarray] = []
        weighted_matrices: list[np.ndarray] = []

        current_center = start_center_xy
        local_search_tag = "coarse"

        ref_center, ref_flux, ref_corr_temp = self._build_reference_metrics(beam)

        failure_reason = None
        failure_dot_index = None

        for dot_index in range(n_dots):
            scan_result = self._scan_center(
                beam=beam,
                center=current_center,
                search_width=search_width,
                step_size=step_size,
                ref_center=ref_center,
                ref_flux=ref_flux,
                ref_corr_temp=ref_corr_temp,
            )

            corr_matrices.append(scan_result.metric_corr)
            flux_matrices.append(scan_result.metric_flux)
            weighted_matrices.append(scan_result.metric_weighted)
            scan_records.append(
                {
                    "index": dot_index + 1,
                    "search_type": local_search_tag,
                    "center": scan_result.center.tolist(),
                    "best_index": list(scan_result.best_index),
                    "best_point": scan_result.best_point.tolist(),
                    "best_score": scan_result.best_score,
                    "best_flux": float(scan_result.metric_flux[scan_result.best_index]),
                    "best_weighted": float(
                        scan_result.metric_weighted[scan_result.best_index]
                    ),
                    "threshold": detection_threshold,
                    "grid_shape": list(scan_result.metric_corr.shape),
                }
            )

            _plot_scan_heatmap(
                score_matrix=scan_result.metric_corr,
                grid_points=scan_result.grid_points,
                best_index=scan_result.best_index,
                title=(
                    f"{local_search_tag.title()} search for dot {dot_index + 1} "
                    "(corr metric)"
                ),
                save_path=save_path
                / f"scan_{dot_index + 1:02d}_{local_search_tag}.png",
                found_points=found_points + [scan_result.best_point],
            )

            if scan_result.best_score > detection_threshold:
                failure_reason = (
                    f"No focal mask dot found below threshold {detection_threshold:.3f} "
                    f"at search {dot_index + 1}"
                )
                failure_dot_index = dot_index + 1
                if dot_index == 0:
                    break
                break

            found_points.append(np.asarray(scan_result.best_point, dtype=float))
            found_scores.append(float(scan_result.best_score))
            self._set_stage_position("BMX", beam, float(scan_result.best_point[0]))
            self._set_stage_position("BMY", beam, float(scan_result.best_point[1]))

            if len(found_points) >= n_dots:
                break

            current_center = line_model.predict_next_point(found_points, dot_spacing)
            local_search_tag = f"dot_{dot_index + 2:02d}"

        if failure_reason is not None and not found_points:
            status = "failed"
        elif len(found_points) == n_dots:
            status = "complete"
        else:
            status = "partial"
        line_fit = None
        if len(found_points) >= 2:
            slope, intercept = line_model.fit(found_points)
            axis, _ = line_model.axis_and_sign()
            line_fit = {
                "axis": axis,
                "slope": slope,
                "intercept": intercept,
            }

        output = {
            "beam": beam,
            "line_direction": line_direction,
            "start_center": start_center_xy.tolist(),
            "step_size": step_size,
            "search_width": search_width,
            "dot_spacing": dot_spacing,
            "n_dots_requested": n_dots,
            "detection_threshold": detection_threshold,
            "status": status,
            "failure_reason": failure_reason,
            "failed_dot_index": failure_dot_index,
            "found_dots": [
                {
                    "index": idx + 1,
                    "position": point.tolist(),
                    "score": score,
                }
                for idx, (point, score) in enumerate(zip(found_points, found_scores))
            ],
            "scan_records": scan_records,
            "line_fit": line_fit,
        }

        json_path = save_path / out_file
        json_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

        np.savez_compressed(
            save_path / "focal_mask_scan_data.npz",
            start_center=start_center_xy,
            found_points=(
                np.asarray(found_points, dtype=float)
                if found_points
                else np.empty((0, 2))
            ),
            found_scores=np.asarray(found_scores, dtype=float),
            coarse_and_local_metric_flux=np.asarray(flux_matrices, dtype=float),
            coarse_and_local_metric_corr=np.asarray(corr_matrices, dtype=float),
            coarse_and_local_metric_weighted=np.asarray(weighted_matrices, dtype=float),
        )

        self._plot_final_summary(
            found_points, line_model, save_path / "focal_mask_summary.png"
        )

        if failure_reason is not None and failure_dot_index == 1:
            raise RuntimeError(failure_reason)

        return output
