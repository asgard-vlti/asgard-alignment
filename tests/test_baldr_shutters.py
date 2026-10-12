import json
import unittest
from pathlib import Path
from unittest import mock

from zaber_motion import Units

from asgard_alignment.Instrument import Instrument
from asgard_alignment.MultiDeviceServer import MultiDeviceServer
from asgard_alignment.ZaberMotor import ZaberLinearActuator


class BaldrShutterTests(unittest.TestCase):
    def setUp(self):
        self.axis = mock.Mock()
        self.axis.is_parked.return_value = False
        self.axis.is_homed.return_value = True
        self.axis.is_busy.return_value = False
        self.axis.get_position.return_value = 3009.0
        self.axis.warnings.get_flags.return_value = set()
        self.motor = ZaberLinearActuator(
            "BMY1", 26, self.axis, named_positions={"SHUTTER": 500.0}
        )
        self.instr = Instrument.__new__(Instrument)
        self.instr._devices = {
            name: mock.Mock(is_shuttered=False) for name in ("BTT1", "BTP1", "BMX1")
        }
        self.instr._devices["BMY1"] = self.motor
        self.instr.b_shutter_states = {i: "open" for i in range(1, 5)}
        self.instr.b_shutter_last_pos = {
            i: {f"BMY{i}": None} for i in range(1, 5)
        }
        self.server = MultiDeviceServer.__new__(MultiDeviceServer)
        self.server.instr = self.instr
        self.server.database_message = {"command": {"parameters": []}}
        self.server.db_update_socket = mock.Mock()

    def assert_move(self, position):
        self.axis.move_absolute.assert_called_once_with(
            position, unit=Units.LENGTH_MICROMETRES, wait_until_idle=False
        )

    def test_moveabs_accepts_integer_and_decimal_targets(self):
        for target in ("500", "500.0"):
            with self.subTest(target=target):
                self.axis.move_absolute.reset_mock()
                response = self.server.handle_message(f"moveabs BMY1 {target}")
                self.assertEqual(response, (True, "ACK"))
                self.assert_move(500.0)
                message = json.loads(
                    self.server.db_update_socket.send_string.call_args.args[0].rstrip(
                        "\0"
                    )
                )
                self.assertEqual(message["command"]["parameters"][0]["value"], 500)

    def test_moveabs_rejects_invalid_target_without_motion(self):
        with self.assertLogs(level="ERROR"):
            _, response = self.server.handle_message("moveabs BMY1 invalid")
        self.assertTrue(response.startswith("NACK:"))
        self.axis.move_absolute.assert_not_called()
        self.server.db_update_socket.send_string.assert_not_called()

    def test_asg_setup_moves_to_shutter_preset(self):
        response = self.server.handle_message("asg_setup BMY1 NAME SHUTTER")
        self.assertEqual(response, (True, "ACK"))
        self.assert_move(500.0)

    def test_asg_setup_reports_missing_preset(self):
        self.motor.named_positions.clear()
        _, response = self.server.handle_message("asg_setup BMY1 NAME SHUTTER")
        self.assertTrue(response.startswith("NACK:"))
        self.assertIn("BMY1 does not have a named position SHUTTER", response)
        self.axis.move_absolute.assert_not_called()

    def test_failed_close_does_not_mark_beam_shuttered(self):
        for positions in ({}, {"SHUTTER": -1.0}, {"SHUTTER": 10001.0}):
            with self.subTest(positions=positions):
                self.instr.b_shutter_states[1] = "open"
                for device in self.instr.devices.values():
                    device.is_shuttered = False
                self.motor.named_positions = positions
                with self.assertLogs(level="ERROR"):
                    _, response = self.server.handle_message("b_shut close 1")
                self.assertTrue(response.startswith("NACK:"))
                self.assertEqual(self.instr.b_shutter_states[1], "open")
                for device in self.instr.devices.values():
                    self.assertFalse(device.is_shuttered)
                self.axis.move_absolute.assert_not_called()

    def test_close_is_idempotent_and_open_restores_position(self):
        self.server.handle_message("b_shut close 1")
        self.assert_move(500.0)
        self.assertEqual(self.instr.b_shutter_states[1], "closed")
        for device in self.instr.devices.values():
            self.assertTrue(device.is_shuttered)

        self.axis.get_position.return_value = 500.0
        self.server.handle_message("b_shut close 1")
        self.assert_move(500.0)
        self.assertEqual(self.instr.b_shutter_last_pos[1]["BMY1"], 3009.0)

        self.axis.move_absolute.reset_mock()
        self.server.handle_message("b_shut open 1")
        self.assert_move(3009.0)
        self.assertEqual(self.instr.b_shutter_states[1], "open")
        for device in self.instr.devices.values():
            self.assertFalse(device.is_shuttered)

    def test_loading_phasemasks_preserves_configured_shutter(self):
        config_file = Path(__file__).resolve().parents[1] / "motor_info_full_system.json"
        config = json.loads(config_file.read_text())
        motor_config = next(m for m in config["motors"] if m["name"] == "BMY1")
        for key in ("named_pos", "named_positions"):
            with self.subTest(key=key):
                cfg = dict(motor_config)
                cfg[key] = cfg.pop("named_pos")
                self.instr._motor_config = {"BMY1": cfg}
                controller = mock.Mock()
                controller.get_device.return_value.get_axis.return_value = self.axis
                self.instr._controllers = {cfg["x_mcc_ip_address"]: controller}
                with (
                    mock.patch(
                        "asgard_alignment.Instrument.glob.glob",
                        return_value=["positions.json"],
                    ),
                    mock.patch(
                        "asgard_alignment.Instrument.os.path.getmtime", return_value=1
                    ),
                    mock.patch(
                        "builtins.open",
                        mock.mock_open(read_data='{"J1": [2500, 3009]}'),
                    ),
                ):
                    self.assertTrue(self.instr._attempt_to_open("BMY1"))
                self.server.handle_message("asg_setup BMY1 NAME SHUTTER")
                self.assert_move(500.0)
                self.assertEqual(self.instr.devices["BMY1"].named_positions["J1"], 3009)
                self.axis.move_absolute.reset_mock()


if __name__ == "__main__":
    unittest.main()
