"""Convert between 12x12 text grids and 140-value flat-map text files.

Forward direction (default):
	Input:  12 rows of 12 space-separated numbers.
	Output: 140 values, one per line, in row-major order with the four
					corner values ((0,0), (0,11), (11,0), (11,11)) omitted.
					By default, the 12x12 grid is treated as centered on 0, so 0.5
					is added to each non-corner value in the flat-map output.
					With -c/--calibration, a 140-value per-index offset is used
					instead (added during flattening).

Reverse direction (--reverse):
	Input:  140 values, one per line.
	Output: 12x12 space-separated grid; corner positions are filled with 0.
					By default, the 140-value flat map is treated as centered on 0.5,
					so 0.5 is subtracted from each input value.
					With -c/--calibration, the corresponding 140-value per-index
					offset is subtracted instead.

Output format matches other ``*_FLAT_MAP_COMMANDS.txt`` files in this folder.
"""

from __future__ import annotations

import argparse
from pathlib import Path


GRID_SIZE = 12
FLAT_SIZE = GRID_SIZE * GRID_SIZE - 4  # 140
DEFAULT_CENTER_OFFSET = 0.5

CORNER_INDICES = {
	(0, 0),
	(0, GRID_SIZE - 1),
	(GRID_SIZE - 1, 0),
	(GRID_SIZE - 1, GRID_SIZE - 1),
}

# Linear (row-major) positions of the four corners.
_CORNER_LINEAR = {r * GRID_SIZE + c for r, c in CORNER_INDICES}


def parse_grid(path: Path) -> list[list[float]]:
	"""Read a 12x12 grid of floats from ``path``."""
	rows: list[list[float]] = []
	with path.open("r", encoding="utf-8") as handle:
		for line_number, raw_line in enumerate(handle, start=1):
			stripped = raw_line.strip()
			if not stripped:
				continue

			parts = stripped.split()
			if len(parts) != GRID_SIZE:
				raise ValueError(
					f"Line {line_number}: expected {GRID_SIZE} values, "
					f"found {len(parts)}."
				)

			try:
				row = [float(value) for value in parts]
			except ValueError as exc:
				raise ValueError(
					f"Line {line_number}: contains a non-numeric value."
				) from exc

			rows.append(row)

	if len(rows) != GRID_SIZE:
		raise ValueError(f"Expected {GRID_SIZE} non-empty rows, found {len(rows)}.")

	return rows


def flatten_without_corners(
	grid: list[list[float]], offsets: list[float] | None = None
) -> list[float]:
	"""Return row-major values excluding the four corners.

	By default, adds 0.5 to each value. If ``offsets`` is provided,
	the per-index offsets are added instead.
	"""
	flattened: list[float] = []
	offset_index = 0
	for row_idx, row in enumerate(grid):
		for col_idx, value in enumerate(row):
			if (row_idx, col_idx) in CORNER_INDICES:
				continue
			offset = DEFAULT_CENTER_OFFSET if offsets is None else offsets[offset_index]
			flattened.append(value + offset)
			offset_index += 1

	expected_count = GRID_SIZE * GRID_SIZE - len(CORNER_INDICES)
	if len(flattened) != expected_count:
		raise RuntimeError(
			f"Internal error: expected {expected_count} values, got {len(flattened)}."
		)
	return flattened


def write_flat_map(values: list[float], output_path: Path) -> None:
	"""Write one value per line."""
	with output_path.open("w", encoding="utf-8") as handle:
		for value in values:
			handle.write(f"{value}\n")


# ---------------------------------------------------------------------------
# Reverse: 140-value flat map -> 12x12 grid
# ---------------------------------------------------------------------------

def parse_flat_map(path: Path) -> list[float]:
	"""Read 140 values (one per line) from *path*."""
	values: list[float] = []
	with path.open("r", encoding="utf-8") as handle:
		for lineno, raw in enumerate(handle, start=1):
			stripped = raw.strip()
			if not stripped:
				continue
			try:
				values.append(float(stripped))
			except ValueError as exc:
				raise ValueError(f"Line {lineno}: non-numeric value.") from exc
	if len(values) != FLAT_SIZE:
		raise ValueError(f"Expected {FLAT_SIZE} values, found {len(values)}.")
	return values


def parse_calibration_offsets(path: Path) -> list[float]:
	"""Read 140 calibration offsets (one value per non-empty line)."""
	offsets: list[float] = []
	with path.open("r", encoding="utf-8") as handle:
		for lineno, raw in enumerate(handle, start=1):
			stripped = raw.strip()
			if not stripped:
				continue
			parts = stripped.split()
			if len(parts) != 1:
				raise ValueError(
					f"Calibration line {lineno}: expected exactly 1 value, found {len(parts)}."
				)
			try:
				offsets.append(float(parts[0]))
			except ValueError as exc:
				raise ValueError(
					f"Calibration line {lineno}: non-numeric value."
				) from exc

	if len(offsets) != FLAT_SIZE:
		raise ValueError(
			f"Calibration file must contain {FLAT_SIZE} values, found {len(offsets)}."
		)
	return offsets


def expand_to_grid(values: list[float], offsets: list[float] | None = None) -> list[list[float]]:
	"""Reconstruct a 12x12 grid; corner positions are set to 0.

	By default, subtracts 0.5 from each value. If ``offsets`` is provided,
	the per-index offsets are subtracted instead.
	"""
	grid: list[list[float]] = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]
	flat_iter = iter(values)
	offset_index = 0
	for linear in range(GRID_SIZE * GRID_SIZE):
		if linear in _CORNER_LINEAR:
			continue  # leave as 0.0
		r, c = divmod(linear, GRID_SIZE)
		offset = DEFAULT_CENTER_OFFSET if offsets is None else offsets[offset_index]
		grid[r][c] = next(flat_iter) - offset
		offset_index += 1
	return grid


def write_grid(grid: list[list[float]], output_path: Path) -> None:
	"""Write a 12x12 grid, one row per line, values space-separated."""
	with output_path.open("w", encoding="utf-8") as handle:
		for row in grid:
			handle.write(" ".join(str(v) for v in row) + "\n")


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Convert between a 12x12 text grid and a 140-line flat-map file.\n\n"
			"Forward (default): 12x12 grid  ->  140-value flat map (corners omitted).\n"
			"Reverse (--reverse): 140-value flat map  ->  12x12 grid (corners = 0)."
		),
		formatter_class=argparse.RawDescriptionHelpFormatter,
	)
	parser.add_argument("input", type=Path, help="Path to the input file.")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		default=None,
		help=(
			"Path to the output file. "
			"Defaults to '<stem>_FLAT_MAP_COMMANDS.txt' (forward) "
			"or '<stem>_12x12.txt' (reverse)."
		),
	)
	parser.add_argument(
		"--reverse",
		action="store_true",
		help="Reverse: expand a 140-value flat map to a 12x12 grid.",
	)
	parser.add_argument(
		"-c",
		"--calibration",
		type=Path,
		default=None,
		help=(
			"Optional calibration offset file with 140 values (one per line). "
			"When provided, these per-index offsets are used instead of the default 0.5."
		),
	)
	return parser


def _safe_output(input_path: Path, output_path: Path) -> None:
	"""Raise if output would silently overwrite the input file."""
	if output_path.resolve() == input_path.resolve():
		raise ValueError(
			f"Output path '{output_path}' is the same as the input file. "
			"Use -o/--output to specify a different destination."
		)


def main() -> int:
	args = build_parser().parse_args()
	input_path: Path = args.input

	try:
		offsets = (
			parse_calibration_offsets(args.calibration)
			if args.calibration is not None
			else None
		)
	except (OSError, ValueError) as exc:
		print(f"Error: {exc}")
		return 1

	if args.reverse:
		default_out = input_path.with_name(f"{input_path.stem}_12x12.txt")
		output_path: Path = args.output if args.output is not None else default_out
		_safe_output(input_path, output_path)
		try:
			values = parse_flat_map(input_path)
			grid = expand_to_grid(values, offsets)
			write_grid(grid, output_path)
		except (OSError, ValueError) as exc:
			print(f"Error: {exc}")
			return 1
		print(f"Wrote {GRID_SIZE}x{GRID_SIZE} grid to {output_path}")
	else:
		default_out = input_path.with_name(f"{input_path.stem}_FLAT_MAP_COMMANDS.txt")
		output_path = args.output if args.output is not None else default_out
		_safe_output(input_path, output_path)
		try:
			grid = parse_grid(input_path)
			values = flatten_without_corners(grid, offsets)
			write_flat_map(values, output_path)
		except (OSError, ValueError, RuntimeError) as exc:
			print(f"Error: {exc}")
			return 1
		print(f"Wrote {len(values)} values to {output_path}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

