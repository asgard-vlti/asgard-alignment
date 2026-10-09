import importlib.util
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import matplotlib
import numpy as onp

matplotlib.use("Agg")

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "heimdallr/m_linbo_search.py"
SPEC = importlib.util.spec_from_file_location("m_linbo_search", SCRIPT_PATH)
linbo_search = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(linbo_search)


class FakeMdsSocket:
    def __init__(
        self, events, starting_position=1000, starting_hfo=8.0, reject_position=None
    ):
        self.events = events
        self.position = starting_position
        self.hfo_position = starting_hfo
        self.reject_position = reject_position
        self.hfo_after_moves = []

    def send_string(self, command):
        self.events.append(command)
        self.command = command

    def recv_string(self):
        if self.command.startswith("read HPOL"):
            return str(self.position)
        raise AssertionError(self.command)

    def run_command(self, command, check, stdout, stderr):
        if not check or stderr != subprocess.STDOUT:
            raise AssertionError("Command output must be captured and checked")
        stdout.write("verbose child output\n")
        self.events.append(tuple(command))
        if command[0] == "move-hpol":
            target = int(command[2])
            if target == self.reject_position:
                stdout.write("move rejected\n")
                raise subprocess.CalledProcessError(1, command)
            self.hfo_position += (target - self.position) * 0.11 / 1000
            self.position = target
            self.hfo_after_moves.append(self.hfo_position)


class FakeHeimdallrSocket:
    def __init__(self, events):
        self.events = events
        self.count = 0

    def send_string(self, command):
        self.events.append(command)
        self.command = command

    def recv_string(self):
        if self.command != "status":
            raise AssertionError(self.command)
        self.count += 1
        return json.dumps(
            {
                "v2_K1": [index + self.count % 2 for index in range(6)],
                "v2_K2": [10 + index + self.count % 2 for index in range(6)],
            }
        )


class LinboSearchTests(unittest.TestCase):
    def test_scan_order_data_and_saved_files(self):
        events = []
        mds = FakeMdsSocket(events)
        heimdallr = FakeHeimdallrSocket(events)
        results = linbo_search.run_scan(mds, heimdallr, mds.run_command)
        starting_position, positions, labels, averaged = results

        self.assertEqual(starting_position, 1000)
        onp.testing.assert_array_equal(positions, onp.arange(850, 1151, 30))
        onp.testing.assert_array_equal(labels, ["1-2", "1-3", "1-4"])
        self.assertEqual(averaged.shape, (11, 2, 3))
        onp.testing.assert_array_equal(averaged[0, 0], [0.5, 1.5, 2.5])
        onp.testing.assert_array_equal(averaged[0, 1], [10.5, 11.5, 12.5])
        self.assertEqual(heimdallr.count, 11 * 50)
        self.assertEqual(mds.position, starting_position)
        self.assertAlmostEqual(mds.hfo_after_moves[0], 7.9835)
        self.assertAlmostEqual(mds.hfo_position, 8.0)
        normal_fringe_command = (
            "find-fringes",
            linbo_search.fringe_band,
            str(linbo_search.fringe_srange),
            str(linbo_search.fringe_step),
        )
        self.assertEqual(events.count(normal_fringe_command), 11)
        self.assertEqual(
            sum(event[0] == "move-hpol" for event in events if isinstance(event, tuple)),
            12,
        )
        self.assertEqual(events.count(("h-tilts",)), 11)
        self.assertEqual(
            events[:6],
            [
                "read HPOL1",
                ("move-hpol", "1", "850"),
                "read HPOL1",
                ("h-tilts",),
                normal_fringe_command,
                "status",
            ],
        )
        self.assertEqual(events[-3:], ["read HPOL1", ("move-hpol", "1", "1000"), "read HPOL1"])

        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(linbo_search.Path, "cwd", return_value=Path(directory)):
                with mock.patch.object(linbo_search.plt, "show"):
                    with mock.patch.object(linbo_search.plt, "close"):
                        linbo_search.save_results(*results)
                        figure = linbo_search.plt.gcf()
                        self.assertEqual(len(figure.axes), 2)
                        for band_index, axis in enumerate(figure.axes):
                            self.assertEqual(len(axis.lines), 3)
                            for baseline_index, line in enumerate(axis.lines):
                                onp.testing.assert_array_equal(line.get_xdata(), positions)
                                onp.testing.assert_array_equal(
                                    line.get_ydata(),
                                    averaged[:, band_index, baseline_index],
                                )
            data_path = next(Path(directory).glob("*.npz"))
            self.assertTrue(data_path.with_suffix(".png").is_file())
            with onp.load(data_path) as saved:
                onp.testing.assert_array_equal(saved["averaged_v2"], averaged)
                onp.testing.assert_array_equal(saved["hpol_positions"], positions)
                onp.testing.assert_array_equal(saved["baseline_labels"], labels)
                self.assertEqual(int(saved["starting_hpol_position"]), 1000)
                self.assertEqual(int(saved["beam_number"]), 1)
                self.assertEqual(int(saved["scan_width"]), 300)
                self.assertEqual(int(saved["scan_nsteps"]), 11)
                self.assertEqual(str(saved["fringe_band"]), "K1")
                self.assertEqual(float(saved["fringe_srange"]), 8)
                self.assertEqual(float(saved["fringe_step"]), linbo_search.fringe_step)
                self.assertEqual(int(saved["status_message_count"]), 50)

    def test_command_failure_restores_hpol(self):
        events = []
        mds = FakeMdsSocket(events)

        def fail_on_tilts(command, check, stdout, stderr):
            if command == ["h-tilts"]:
                events.append(tuple(command))
                raise RuntimeError("h-tilts failed")
            mds.run_command(command, check, stdout, stderr)

        with self.assertRaisesRegex(RuntimeError, "h-tilts failed"):
            linbo_search.run_scan(mds, FakeHeimdallrSocket(events), fail_on_tilts)

        self.assertEqual(mds.position, 1000)
        self.assertEqual(events[-3:], ["read HPOL1", ("move-hpol", "1", "1000"), "read HPOL1"])

    def test_rejected_first_move_keeps_original_error_and_position(self):
        events = []
        mds = FakeMdsSocket(events, starting_position=0, reject_position=-150)

        with self.assertRaisesRegex(RuntimeError, "move rejected"):
            linbo_search.run_scan(
                mds, FakeHeimdallrSocket(events), mds.run_command
            )

        self.assertEqual(mds.position, 0)
        self.assertEqual(events.count(("move-hpol", "1", "0")), 0)

    def test_failed_command_reports_recent_output(self):
        def fail(command, check, stdout, stderr):
            for index in range(25):
                stdout.write(f"line {index}\n")
            raise subprocess.CalledProcessError(2, command)

        with self.assertRaisesRegex(RuntimeError, "line 24") as error:
            linbo_search.run_quiet_command(["find-fringes", "K1", "8", "0.5"], fail)

        self.assertNotIn("line 0\n", str(error.exception))

    def test_other_beam_selects_its_three_baselines(self):
        events = []
        mds = FakeMdsSocket(events)
        with mock.patch.object(linbo_search, "beam_number", 2):
            results = linbo_search.run_scan(
                mds, FakeHeimdallrSocket(events), mds.run_command
            )

        onp.testing.assert_array_equal(results[2], ["1-2", "2-3", "2-4"])
        onp.testing.assert_array_equal(results[3][0, 0], [0.5, 3.5, 4.5])
        self.assertIn(("move-hpol", "2", "850"), events)


if __name__ == "__main__":
    unittest.main()
