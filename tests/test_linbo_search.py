import importlib.util
import json
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
    def __init__(self, events, starting_position=1000, reject_position=None):
        self.events = events
        self.position = starting_position
        self.reject_position = reject_position

    def send_string(self, command):
        self.events.append(command)
        self.command = command

    def recv_string(self):
        if self.command.startswith("read HPOL"):
            return str(self.position)
        if self.command.startswith("moveabs HPOL"):
            position_text = self.command.split()[-1]
            if "." not in position_text:
                return "NACK: expected a float"
            if float(position_text) == self.reject_position:
                return "NACK: move rejected"
            self.position = int(float(position_text))
            return "ACK"
        raise AssertionError(self.command)


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

        def run_command(command, check):
            self.assertTrue(check)
            events.append(tuple(command))

        results = linbo_search.run_scan(mds, heimdallr, run_command)
        starting_position, positions, labels, averaged = results

        self.assertEqual(starting_position, 1000)
        onp.testing.assert_array_equal(positions, onp.arange(850, 1151, 30))
        onp.testing.assert_array_equal(labels, ["1-2", "1-3", "1-4"])
        self.assertEqual(averaged.shape, (11, 2, 3))
        onp.testing.assert_array_equal(averaged[0, 0], [0.5, 1.5, 2.5])
        onp.testing.assert_array_equal(averaged[0, 1], [10.5, 11.5, 12.5])
        self.assertEqual(heimdallr.count, 11 * 50)
        self.assertEqual(mds.position, starting_position)
        self.assertEqual(events.count(("find-fringes", "K1", "8", "0.5")), 22)
        self.assertEqual(events.count(("h-tilts",)), 11)
        self.assertEqual(
            events[:7],
            [
                "read HPOL1",
                "moveabs HPOL1 850.0",
                "read HPOL1",
                ("find-fringes", "K1", "8", "0.5"),
                ("h-tilts",),
                ("find-fringes", "K1", "8", "0.5"),
                "status",
            ],
        )
        self.assertEqual(events[-3:], ["read HPOL1", "moveabs HPOL1 1000.0", "read HPOL1"])

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
                self.assertEqual(float(saved["fringe_step"]), 0.5)
                self.assertEqual(int(saved["status_message_count"]), 50)

    def test_command_failure_restores_hpol(self):
        events = []
        mds = FakeMdsSocket(events)

        def fail_on_tilts(command, check):
            events.append(tuple(command))
            if command == ["h-tilts"]:
                raise RuntimeError("h-tilts failed")

        with self.assertRaisesRegex(RuntimeError, "h-tilts failed"):
            linbo_search.run_scan(mds, FakeHeimdallrSocket(events), fail_on_tilts)

        self.assertEqual(mds.position, 1000)
        self.assertEqual(events[-3:], ["read HPOL1", "moveabs HPOL1 1000.0", "read HPOL1"])

    def test_rejected_first_move_keeps_original_error_and_position(self):
        events = []
        mds = FakeMdsSocket(events, starting_position=0, reject_position=-150)

        with self.assertRaisesRegex(RuntimeError, "move rejected"):
            linbo_search.run_scan(
                mds, FakeHeimdallrSocket(events), lambda *_args, **_kwargs: None
            )

        self.assertEqual(mds.position, 0)
        self.assertEqual(events.count("moveabs HPOL1 0.0"), 0)

    def test_other_beam_selects_its_three_baselines(self):
        events = []
        with mock.patch.object(linbo_search, "beam_number", 2):
            results = linbo_search.run_scan(
                FakeMdsSocket(events), FakeHeimdallrSocket(events), lambda *_args, **_kwargs: None
            )

        onp.testing.assert_array_equal(results[2], ["1-2", "2-3", "2-4"])
        onp.testing.assert_array_equal(results[3][0, 0], [0.5, 3.5, 4.5])


if __name__ == "__main__":
    unittest.main()
