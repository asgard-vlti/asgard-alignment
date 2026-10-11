import json
import unittest
from unittest import mock

try:
    from asgard_alignment.MultiDeviceServer import MultiDeviceServer
except ModuleNotFoundError:
    MultiDeviceServer = None


@unittest.skipIf(MultiDeviceServer is None, "MDS runtime dependencies are unavailable")
class MDSStandbyTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.server = MultiDeviceServer.__new__(MultiDeviceServer)
        self.server.instr = mock.Mock()
        self.server.instr._motor_config = {"M1": {}, "M2": {}}
        self.server.instr.devices = {"M1": object(), "M2": object(), "SRL": object()}
        self.server.database_message = {"command": {"parameters": []}}

        def save(subset, filename):
            self.events.append(("save", subset, filename))
            return "ACK: saved"

        def standby(axis):
            self.events.append(("standby", axis))
            self.server.instr.devices.pop(axis, None)
            return "ACK"

        self.server.instr.save.side_effect = save
        self.server.instr.standby.side_effect = standby

    def send_standby(self, device):
        message = {
            "command": {
                "name": "standby",
                "time": self.server.get_time_stamp(),
                "parameters": [{"device": device}],
            }
        }
        self.server.handle_eso_msg(json.dumps(message) + "\0")

    def test_status_distinguishes_motors_from_lamps(self):
        self.assertEqual(self.server.handle_message("standby_status"), (True, "ACTIVE"))

        del self.server.instr.devices["M1"]
        del self.server.instr.devices["M2"]

        self.assertEqual(self.server.handle_message("standby_status"), (True, "STANDBY"))
        self.assertIn("SRL", self.server.instr.devices)

    def test_full_standby_saves_before_devices_and_only_once(self):
        self.send_standby("all")

        self.assertEqual(self.events[0][:2], ("save", "all"))
        self.assertRegex(
            self.events[0][2], r"^before_standby_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}$"
        )
        self.assertEqual(
            {event[1] for event in self.events if event[0] == "standby"},
            {"M1", "M2"},
        )
        self.assertEqual(self.server.handle_message("standby_status"), (True, "STANDBY"))

        self.send_standby("all")

        self.assertEqual(sum(event[0] == "save" for event in self.events), 1)

    def test_individual_standby_does_not_save(self):
        self.send_standby("M1")

        self.assertEqual(self.events, [("standby", "M1")])
        self.assertEqual(self.server.get_standby_status(), "ACTIVE")


if __name__ == "__main__":
    unittest.main()
