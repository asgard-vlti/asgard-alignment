import contextlib
import io
import unittest
from unittest import mock

import zmq

from asgard_alignment.cmd_scripts import shutdown_instrument as shutdown_script


class FakeSocket:
    def __init__(self, port):
        self.port = port
        self.closed = False

    def close(self, linger=0):
        self.closed = True


class ShutdownInstrumentTests(unittest.TestCase):
    def setUp(self):
        self.events = []
        self.reachable_ports = {shutdown_script.DM_PORT, shutdown_script.C_RED_PORT}
        self.failed_command = None
        self.failed_response = None
        self.standby_status = "ACTIVE\n"
        self.operator_answer = "n"
        self.opened_sockets = []
        self.mds_connection = FakeSocket(shutdown_script.MDS_PORT)

        def send(connection, command):
            self.events.append(("send", connection.port, command))
            if self.failed_command == (connection.port, command):
                if self.failed_response is not None:
                    return self.failed_response
                raise zmq.Again()
            if command == "standby_status":
                return self.standby_status
            return "Exiting!" if command == "exit" else "OK"

        def open_connection(port):
            connection = FakeSocket(port)
            self.opened_sockets.append(connection)
            return connection

        def send_mds(connection, command):
            self.events.append(("send", shutdown_script.MDS_PORT, command))
            return "OK", connection

        def power_controllino(*args, **kwargs):
            device = mock.Mock()
            device.turn_off.side_effect = lambda name: self.events.append(
                ("device_off", name)
            )
            return device

        def pdu(*args, **kwargs):
            device = mock.Mock()
            device.read_power_value.return_value = "0.0"
            device.read_outlet_status.return_value = "off"
            device.switch_outlet_status.side_effect = lambda outlet, state: (
                self.events.append(("outlet", outlet, state))
            )
            return device

        patches = [
            mock.patch.object(
                shutdown_script,
                "get_mds_connection_or_recover",
                return_value=self.mds_connection,
            ),
            mock.patch.object(
                shutdown_script, "open_zmq_connection", side_effect=open_connection
            ),
            mock.patch.object(shutdown_script, "send_and_get_response", side_effect=send),
            mock.patch.object(
                shutdown_script, "send_with_mds_recovery", side_effect=send_mds
            ),
            mock.patch.object(
                shutdown_script,
                "is_tcp_port_open",
                side_effect=lambda host, port: port in self.reachable_ports,
            ),
            mock.patch.object(shutdown_script, "wait_for_tcp_port", return_value=True),
            mock.patch.object(
                shutdown_script.co, "PowerControllino", side_effect=power_controllino
            ),
            mock.patch.object(shutdown_script, "AtenEcoPDU", side_effect=pdu),
            mock.patch.object(shutdown_script.time, "sleep"),
            mock.patch.object(shutdown_script, "ping_device", return_value=False),
            mock.patch.object(
                shutdown_script,
                "input",
                create=True,
                side_effect=lambda prompt: (
                    self.events.append(("operator", prompt)),
                    self.operator_answer,
                )[1],
            ),
        ]
        for patcher in patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def run_shutdown(self, include_cred=False):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            shutdown_script.shutdown(include_cred)
        return output.getvalue()

    def save_commands(self):
        return [
            event[2]
            for event in self.events
            if event[0] == "send"
            and event[1] == shutdown_script.MDS_PORT
            and event[2].startswith("save ")
        ]

    def test_active_instrument_saves_before_shutdown(self):
        self.run_shutdown()

        self.assertEqual(len(self.save_commands()), 1)
        self.assertTrue(self.save_commands()[0].startswith("save all before_shutdown_"))
        status_event = self.events.index(
            ("send", shutdown_script.MDS_PORT, "standby_status")
        )
        save_event = self.events.index(
            ("send", shutdown_script.MDS_PORT, self.save_commands()[0])
        )
        self.assertLess(status_event, save_event)
        self.assertTrue(self.opened_sockets[0].closed)
        self.assertFalse(self.mds_connection.closed)

    def test_standby_skips_save_and_completes_shutdown(self):
        self.standby_status = "STANDBY\n"

        output = self.run_shutdown()

        self.assertEqual(self.save_commands(), [])
        self.assertIn("MDS is in standby; skipping state save", output)
        self.assertIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )

    def test_unknown_status_rejection_requires_operator_confirmation(self):
        self.standby_status = "NACK: Unknown custom command\n"

        output = self.run_shutdown()

        self.assertEqual(self.save_commands(), [])
        self.assertIn("Aborting shutdown", output)
        self.assertFalse(any(event[0] == "device_off" for event in self.events))
        self.assertFalse(any(event[0] == "outlet" for event in self.events))

    def test_unknown_status_can_continue_without_save(self):
        self.standby_status = "unexpected"
        self.operator_answer = "y"

        output = self.run_shutdown()

        self.assertEqual(self.save_commands(), [])
        self.assertIn("Proceeding with shutdown", output)
        self.assertIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )

    def test_standby_status_timeout_closes_query_socket(self):
        self.failed_command = (shutdown_script.MDS_PORT, "standby_status")

        output = self.run_shutdown()

        self.assertEqual(self.save_commands(), [])
        self.assertIn("Could not determine MDS standby status", output)
        self.assertTrue(self.opened_sockets[0].closed)
        self.assertFalse(self.mds_connection.closed)
        self.assertFalse(any(event[0] == "outlet" for event in self.events))

    def test_standby_status_timeout_can_continue_without_save(self):
        self.failed_command = (shutdown_script.MDS_PORT, "standby_status")
        self.operator_answer = "y"

        self.run_shutdown()

        self.assertEqual(self.save_commands(), [])
        self.assertIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )

    def test_active_save_timeout_keeps_existing_operator_prompt(self):
        timestamp = "2026-10-11 12:00:00"
        self.failed_command = (
            shutdown_script.MDS_PORT,
            f"save all before_shutdown_{timestamp}",
        )

        with mock.patch.object(shutdown_script.time, "strftime", return_value=timestamp):
            output = self.run_shutdown()

        self.assertIn("state is not saved", self.events[-1][1])
        self.assertIn("Aborting shutdown", output)
        self.assertFalse(any(event[0] == "outlet" for event in self.events))

    def test_panel_commands_without_cred_poweroff(self):
        output = self.run_shutdown()

        dm_exit = self.events.index(("send", shutdown_script.DM_PORT, "exit"))
        operator = next(
            i for i, event in enumerate(self.events) if event[0] == "operator"
        )
        cred_stop = self.events.index(("send", shutdown_script.C_RED_PORT, "stop"))
        cred_exit = self.events.index(("send", shutdown_script.C_RED_PORT, "exit"))
        lower_off = self.events.index(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off")
        )
        self.assertLess(dm_exit, operator)
        self.assertLess(operator, cred_stop)
        self.assertLess(cred_stop, cred_exit)
        self.assertLess(cred_exit, lower_off)
        self.assertNotIn(("outlet", shutdown_script.C_RED_OUTLET, "off"), self.events)
        self.assertIn("Sending 'exit' to DM server", output)
        self.assertIn("Sending 'stop' to C-RED server", output)
        self.assertIn("C-RED server response: Exiting!", output)

    def test_include_cred_keeps_existing_camera_shutdown(self):
        self.run_shutdown(include_cred=True)

        cred_commands = [
            event[2]
            for event in self.events
            if event[0] == "send" and event[1] == shutdown_script.C_RED_PORT
        ]
        self.assertEqual(
            cred_commands,
            ["stop", 'cli "set cooling off"', 'cli "shutdown"'],
        )
        cred_off = self.events.index(("outlet", shutdown_script.C_RED_OUTLET, "off"))
        lower_off = self.events.index(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off")
        )
        self.assertLess(cred_off, lower_off)

    def test_unreachable_panel_servers_are_reported_and_skipped(self):
        self.reachable_ports.clear()
        output = self.run_shutdown()

        panel_commands = [
            event
            for event in self.events
            if event[0] == "send"
            and event[1] in (shutdown_script.DM_PORT, shutdown_script.C_RED_PORT)
        ]
        self.assertEqual(panel_commands, [])
        self.assertIn("DM server is already unreachable", output)
        self.assertIn("C-RED server is already unreachable", output)
        self.assertIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )

    def test_dm_timeout_aborts_before_lower_box_poweroff(self):
        self.failed_command = (shutdown_script.DM_PORT, "exit")
        output = self.run_shutdown()

        self.assertIn("DM server did not acknowledge 'exit'", output)
        self.assertIn("Aborting shutdown", output)
        self.assertFalse(any(event[0] == "operator" for event in self.events))
        self.assertFalse(any(event[0] == "outlet" for event in self.events))

    def test_dm_connection_failure_aborts_before_lower_box_poweroff(self):
        def open_connection(port):
            if port == shutdown_script.DM_PORT:
                raise zmq.ZMQError("socket unavailable")
            return FakeSocket(port)

        with mock.patch.object(
            shutdown_script,
            "open_zmq_connection",
            side_effect=open_connection,
        ):
            output = self.run_shutdown()

        self.assertIn("Could not connect to DM server", output)
        self.assertFalse(any(event[0] == "operator" for event in self.events))
        self.assertFalse(any(event[0] == "outlet" for event in self.events))

    def test_cred_timeout_aborts_before_lower_box_poweroff(self):
        self.failed_command = (shutdown_script.C_RED_PORT, "exit")
        output = self.run_shutdown()

        self.assertIn("C-RED server did not acknowledge 'exit'", output)
        self.assertIn("Aborting shutdown", output)
        self.assertNotIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )

    def test_command_rejection_aborts_before_lower_box_poweroff(self):
        self.failed_command = (shutdown_script.C_RED_PORT, "stop")
        self.failed_response = "Error: command failed"
        output = self.run_shutdown()

        self.assertIn("C-RED server rejected 'stop'", output)
        self.assertNotIn(("send", shutdown_script.C_RED_PORT, "exit"), self.events)
        self.assertNotIn(
            ("outlet", shutdown_script.LOWER_BOX_OUTLET, "off"), self.events
        )


if __name__ == "__main__":
    unittest.main()
