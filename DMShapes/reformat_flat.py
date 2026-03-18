"""Convert 12x12 text grids into 140-value flat-map text files.

The input file must contain exactly 12 rows of 12 space-separated numbers.
The output preserves row-major order and omits the four corner values:
(row, col) = (0, 0), (0, 11), (11, 0), (11, 11).

Output format matches other ``*_FLAT_MAP_COMMANDS.txt`` files in this folder:
one number per line (140 total).
"""

from __future__ import annotations

import argparse
from pathlib import Path


GRID_SIZE = 12
CORNER_INDICES = {
	(0, 0),
	(0, GRID_SIZE - 1),
	(GRID_SIZE - 1, 0),
	(GRID_SIZE - 1, GRID_SIZE - 1),
}


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


def flatten_without_corners(grid: list[list[float]]) -> list[float]:
	"""Return row-major values excluding the four corners."""
	flattened: list[float] = []
	for row_idx, row in enumerate(grid):
		for col_idx, value in enumerate(row):
			if (row_idx, col_idx) in CORNER_INDICES:
				continue
			flattened.append(value)

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


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Convert a 12x12 space-separated text grid to a 140-line text file "
			"by omitting the four corner values."
		)
	)
	parser.add_argument("input", type=Path, help="Path to input 12x12 text file")
	parser.add_argument(
		"-o",
		"--output",
		type=Path,
		help=(
			"Path to output text file. If omitted, uses the input filename with "
			"suffix '_FLAT_MAP_COMMANDS.txt'."
		),
	)
	return parser


def default_output_path(input_path: Path) -> Path:
	stem = input_path.stem
	return input_path.with_name(f"{stem}_FLAT_MAP_COMMANDS.txt")


def main() -> int:
	args = build_parser().parse_args()

	input_path = args.input
	output_path = args.output if args.output is not None else default_output_path(input_path)

	try:
		grid = parse_grid(input_path)
		values = flatten_without_corners(grid)
		write_flat_map(values, output_path)
	except (OSError, ValueError, RuntimeError) as exc:
		print(f"Error: {exc}")
		return 1

	print(f"Wrote {len(values)} values to {output_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

