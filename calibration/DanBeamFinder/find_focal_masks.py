"""
python find_focal_mask.py

Usage: Find the focal plane mask.


--beam [int] : which beam to do the search on (1-4). must be provided.
--line-direction [str] : the direction of the line of dots on the focal plane
    mask. Can be ["+x", "-x", "+y", "-y"]. Must be provided.
--start-center [x,y] | "current" : the centre of the search area. Can be "current" to
    use the current position of the stage as the center, or a list of [x,y] coordinates.
--step-size [float] : the step size in microns for the search grid. Default is 20 microns.
--search-width [float] : the width of the search area in microns. Equal to the height. Default is 200 microns.
--dot-spacing [float] : the spacing of the dots on the focal plane mask in microns. Default is 1000 microns.
--save-path [str] : the path to save the results. Default is "Data/{date}/Scan_{beam}_{current_datetime}".
--n-dots [int] : the number of dots to search for in the focal plane mask. Default is 5.
--detection-threshold [float] : the threshold for detecting the dots in the camera images.
    No mask is ~1.0, thresholds must be <1.0. Default 0.9.
--out-file [str] : the name of the output file to save the positions of the
    found dots. Default is "focal_mask_positions.json".

This script finds a line of phase mask dots without moving through the whole focal plane. First,
it does a scan around the starting center. If a mask is not found above the detection threshold,
throw an error. If a mask is found, it then moves along the line direction and finds the next dot.
For each dot, it saves the stage position and fits a line in (x,y). The distance between the points is assumed
to be the dot spacing. Each subsequent dot is searched for using the fitted line and the dot spacing to
predict the next position (and used as the center of the search area). The search continues until the specified
number of dots is found or the search fails.
Finally, the positions of all the found dots are saved to a file.

"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib

from libs.FPM_Finder import FPM_Finder, LINE_DIRECTION_OPTIONS


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Find a line of focal-plane mask dots by repeated local raster scans."
        ),
        epilog=(
            "Behavior:\n"
            "  1) Scan around --start-center to find the first dot.\n"
            "  2) Move along --line-direction and repeat until --n-dots are found.\n"
            "  3) Save found positions to --out-file under --save-path.\n\n"
            "Notes:\n"
            "  - !! Note that the starting position must have no features !!\n"
            "  - For --detection-threshold, no-mask is typically ~1.0, so choose <1.0.\n"
            "  - --start-center accepts either 'current' or a string like '[x, y]'.\n\n"
            "Examples:\n"
            "  python find_focal_masks.py --beam 2 --line-direction=+x\n"
            "  python find_focal_masks.py --beam 3 --line-direction=-y "
            "--start-center '[1020, 3980]' --n-dots 7"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--beam",
        type=int,
        required=True,
        choices=[1, 2, 3, 4],
        help="Beam index to search (valid values: 1, 2, 3, 4).",
    )
    parser.add_argument(
        "--line-direction",
        type=str,
        required=True,
        choices=sorted(LINE_DIRECTION_OPTIONS),
        help=(
            "Direction of the line of dots on the focal-plane mask. "
            "Choices: +x, -x, +y, -y."
        ),
    )
    parser.add_argument(
        "--start-center",
        type=str,
        default="current",
        help=(
            "Center of the local search area. Use 'current' to start from the "
            "current stage position, or pass '[x, y]' in microns. "
            "Default: current. !! Note that this position must have no features !!"
        ),
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=20.0,
        help="Grid step size in microns for each local raster scan. Default: 20.0.",
    )
    parser.add_argument(
        "--search-width",
        type=float,
        default=200.0,
        help=(
            "Width of the square local search area in microns "
            "(height equals width). Default: 200.0."
        ),
    )
    parser.add_argument(
        "--dot-spacing",
        type=float,
        default=1000.0,
        help="Expected spacing between neighboring mask dots in microns. Default: 1000.0.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help=("Directory for outputs. Default: Data/Scan_{beam}_{current_datetime}."),
    )
    parser.add_argument(
        "--n-dots",
        type=int,
        default=5,
        help="Number of dots to find before stopping. Default: 5.",
    )
    parser.add_argument(
        "--detection-threshold",
        type=float,
        default=0.9,
        help=(
            "Detection threshold for mask match. Lower is stricter; "
            "use values below ~1.0. Default: 0.9."
        ),
    )
    parser.add_argument(
        "--out-file",
        type=str,
        default="focal_mask_positions.json",
        help=(
            "Output JSON filename for found dot positions and run metadata. "
            "Default: focal_mask_positions.json."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.save_path is None:
        now = datetime.datetime.now().strftime("%H%M%S")
        date_str = datetime.datetime.now().strftime("%Y%m%d")
        save_path = pathlib.Path("Data") / f"{date_str}" / f"Scan_{args.beam}_{now}"
    else:
        save_path = pathlib.Path(args.save_path).expanduser()

    finder = FPM_Finder()
    init_pos = finder.get_positions(args.beam)
    try:
        result = finder.run(
            beam=args.beam,
            line_direction=args.line_direction,
            start_center=args.start_center,
            step_size=args.step_size,
            search_width=args.search_width,
            dot_spacing=args.dot_spacing,
            n_dots=args.n_dots,
            detection_threshold=args.detection_threshold,
            save_path=save_path,
            out_file=args.out_file,
        )
    except RuntimeError as e:
        print(f"Error during focal mask finding: {e}")
        print(f"resetting back to original position ")
        finder.set_positions(args.beam, init_pos)

        return
    except KeyboardInterrupt:
        print("Process interrupted by user. Resetting to original position.")
        finder.set_positions(args.beam, init_pos)
        return

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
