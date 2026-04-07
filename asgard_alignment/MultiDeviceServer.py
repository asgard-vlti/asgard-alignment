import zmq
import asgard_alignment
import argparse
import sys
from parse import parse
import time

import os

import json
import datetime

# deepcopy
from copy import deepcopy

import enum
from dataclasses import dataclass
from typing import Callable

import asgard_alignment.ESOdevice
import asgard_alignment.Instrument
import asgard_alignment.MultiDeviceServer
import asgard_alignment.Engineering
import asgard_alignment.NewportMotor
import asgard_alignment.controllino
import asgard_alignment.ESOdevice

import logging


# guarantees that errors are logged
def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class MockMDS:
    def __init__(self):
        pass

    def handle_zmq(self, message):
        logging.info(f"Received message: {message}")
        return "Dummy response"


class MultiDeviceServer:
    """
    A class to run the Instrument MDS.
    """

    DATABASE_MSG_TEMPLATE = {
        "command": {
            "name": "write",
            "time": "YYYY-MM-DDThh:mm:ss",
            "parameters": [],
        }
    }

    def __init__(self, port, host, config_file):
        self.port = port
        self.host = host
        self.config_file = config_file

        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.bind(f"tcp://{self.host}:{self.port}")

        self.poller = zmq.Poller()
        self.poller.register(self.server, zmq.POLLIN)
        self.batch = 0

        self.db_update_socket = self.context.socket(zmq.PUSH)
        self.db_update_socket.connect("tcp://wag:5561")

        self._reset_setup_ls()
        self.batch = 0
        self.is_stopped = True

        self.database_message = self.DATABASE_MSG_TEMPLATE.copy()

        if config_file == "mock":
            self.instr = MockMDS()
        else:
            self.instr = asgard_alignment.Instrument.Instrument(self.config_file)

            try:
                # set all BLF to standard
                self.instr.devices["BLF1"].setup("NAME", "STANDARD")
                self.instr.devices["BLF2"].setup("NAME", "STANDARD")
                self.instr.devices["BLF3"].setup("NAME", "STANDARD")
                self.instr.devices["BLF4"].setup("NAME", "STANDARD")
            except Exception as e:
                logging.error(f"Error setting BLFs to standard: {e}")

        logging.info("Instrument all set up, ready to accept messages")

    def socket_funct(self, s):
        try:
            message = s.recv_string()
            return message
        except zmq.ZMQError as e:
            logging.error(f"ZMQ Error: {e}")
            return -1

    def log(self, message):
        logging.info(message)

    def run(self):
        running = True
        while running:
            inputready = []
            socks = dict(self.poller.poll(10))
            if self.server in socks and socks[self.server] == zmq.POLLIN:
                inputready.append(self.server)
            for s in inputready:  # loop through our array of sockets/inputs
                data = self.socket_funct(s)
                if data == -1:
                    running = False
                elif data != 0:
                    data_disp = data
                    if data_disp[0] == "{":
                        data_disp = data[:-1]

                    logging.info(f"Received message: {data_disp}")

                    is_custom_msg, response = self.handle_message(data)
                    if response == -1:
                        running = False
                        if s == sys.stdin:
                            self.log("Manually shut down. Goodbye.")
                        else:
                            self.log("Shut down by remote connection. Goodbye.")
                    else:
                        if response is None:
                            response = ""
                        # if is_custom_msg:
                        s.send_string(response + "\n")

    @staticmethod
    def get_time_stamp():
        # time_now = datetime.datetime.now()
        # time_now = time.gmtime()
        # return time.strftime("%Y-%m-%dT%H:%M:%S", time_now)

        # Get the current UTC time
        current_utc_time = datetime.datetime.now(datetime.timezone.utc)

        # Format the UTC time
        return current_utc_time.strftime("%Y-%m-%dT%H:%M:%S")

    def _reset_setup_ls(self):
        self.setup_ls = [[], []]

    def check_if_batch_done(self):
        is_done = True
        for dev in self.setup_ls[self.batch]:
            logging.info(f"Checking if {dev.device_name} is moving... ")
            is_moving = self.instr.devices[dev.device_name].is_moving()
            logging.info(f"Value is {is_moving}")
            if is_moving == True:
                is_done = False
                break
        return is_done

    def handle_message(self, message):
        """
        Handles a recieved message. Custom messages are indicated by lowercase commands
        """

        # if "!" in message:
        if message[0].islower():
            logging.info(f"Custom command: {message}")
            return True, self._handle_custom_command(message)

        if message[0] == "!":
            logging.info("Old custom command")
            return True, "NACK: Are you using old custom commands?"

        try:
            # message = message.rstrip(message[-1])
            json_data = json.loads(message.rstrip(message[-1]))
            logging.info(f"ESO msg recv: {json_data} (type {type(json_data)})")
        except:
            logging.error("Error: Invalid JSON message")
            return False, "NACK: Invalid JSON message"
        command_name = json_data["command"]["name"]
        time_stampIn = json_data["command"]["time"]

        # Acceptable window: ±5 minutes from current UTC time
        try:
            received_time = datetime.datetime.strptime(
                time_stampIn, "%Y-%m-%dT%H:%M:%S"
            )
            now_utc = datetime.datetime.utcnow()
            delta = abs((now_utc - received_time).total_seconds())
            if delta > 300:  # 5 minutes
                logging.warning(
                    f"Received time-stamp {time_stampIn} is out of range (delta={delta}s)"
                )
                command_name = "none"
        except Exception as e:
            logging.error(f"Invalid time-stamp format: {time_stampIn} ({e})")
            command_name = "none"

        reply = {
            "reply": {
                "content": "????",
                "time": "YYYY-MM-DDThh:mm:ss",
                "parameters": [],
            }
        }

        # Verification of received time-stamp (TODO)
        # If the time_stamp is invalid, set command_name to "none",
        # so no command will be processed but a reply will be sent
        # back to the client (set reply to "ERROR")

        ################################
        # Process the received command:
        ################################

        # Case of "online" (sent by wag when bringing ICS online, to check
        # that MCUs are alive and ready)

        self.database_message["command"]["parameters"].clear()

        # Case of "setup" (sent by wag to move devices)
        if "setup" in command_name:
            self.is_stopped = False
            n_devs_commanded = len(json_data["command"]["parameters"])

            semaphore_array = [0] * 100  # TODO: implement this maximum correctly
            # Create a double-list of devices to move
            self._reset_setup_ls()
            for i in range(n_devs_commanded):
                kwd = json_data["command"]["parameters"][i]["name"]
                val = json_data["command"]["parameters"][i]["value"]
                logging.info(f"Setup: {kwd} to {val}")

                # Keywords are in the format: INS.<device>.<motion type>
                prefixes = kwd.split(".")
                dev_name = prefixes[1]
                motion_type = prefixes[2]
                logging.info(f"Device: {dev_name} - motion type: {motion_type}")

                # motion_type can be one of these words:
                # NAME   = Named position (e.g., IN, OUT, J1, H3, ...)
                # ENC    = Absolute encoder position
                # ENCREL = Relative encoder postion (can be negative)
                # ST     = State. Given value is equal to either T or F.
                #          if device is shutter: T = open, F = closed.
                #          if device is lamp: T = on, F = off.

                # Look if device exists in list
                # (something should be done if device does not exist)
                device = self.instr.devices[dev_name]
                if isinstance(device, asgard_alignment.ESOdevice.ESOdevice):
                    semaphore_id = 99
                else:
                    semaphore_id = device.semaphore_id
                if semaphore_array[semaphore_id] == 0:
                    # Semaphore is free =>
                    # Device can be moved now
                    self.setup_ls[0].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )
                    semaphore_array[semaphore_id] = 1
                else:
                    # Semaphore is already taken =>
                    # Device will be moved in a second batch
                    self.setup_ls[1].append(
                        asgard_alignment.ESOdevice.SetupCommand(
                            dev_name, motion_type, val
                        )
                    )

            # Move devices (if two batches, move the first one)
            self.batch = 0
            if len(self.setup_ls[self.batch]) > 0:
                logging.info(f"batch {self.batch} of devices to move:")
                reply["reply"]["parameters"].clear()
                for s in self.setup_ls[self.batch]:
                    logging.info(
                        f"Moving: {s.device_name} to: {s.value} ( setting {s.motion_type} )"
                    )

                    self.instr.devices[s.device_name].setup(s.motion_type, s.value)

                    # Inform wag ICS that the device is moving
                    attribute = "<alias>" + s.device_name + ":DATA.status0"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": "MOVING"}
                    )
            # Once setup is forwarded to the devices, reply OK if everything is
            # normal. This means that the setup has started, no that it is done!
            reply["reply"]["content"] = "OK"

        # Case of "poll" (sent by wag to get the status of the
        # last setup sent. Normally, wag sends a "poll" every
        # second during a setup)

        elif "poll" in command_name:
            # --------------------------------------------------
            # TODO: Add here call to query the status of the batch of
            # devices that is concerned by the last setup command
            # If they all reach the target position or if
            # a STOP command occured, set is_batch_done to 1
            #
            # In this example of back-end server, we simulate
            # that by checking the cntdwnSetup variable
            # --------------------------------------------------
            is_batch_done = self.check_if_batch_done()

            reply["reply"]["parameters"].clear()
            if len(self.setup_ls[self.batch]) > 0:
                for s in self.setup_ls[self.batch]:
                    attribute = "<alias>" + s.device_name + ":DATA.status0"
                    # Case of motor with named position requested
                    if s.motion_type == "NAME":
                        # If motor reached the position, we set the
                        # attribute to the target named position
                        # (given in the setup) otherwise we set it
                        # to MOVING
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": s.value}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: normally the encoder position shall be
                        # reported along with the named position
                        # ...............................................
                        # => Call function to read the encoder position
                        #    store it in a variable "posEnc" and execute:
                        #
                        # attribute = "<alias>" + s.device_name +":DATA.posEnc"
                        # dbMsg['command']['parameters'].\
                        # append({"attribute":attribute, "value":posEnc})
                        attribute = "<alias>" + s.device_name + ":DATA.posEnc"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": s.value}
                        )

                    # Case of shutter or lamp
                    if s.motion_type == "ST":
                        # Here the device can be either a lamp or a shutter
                        # Add here code to find out the type of s.device_name

                        if isinstance(
                            self.instr.devices[s.device_name],
                            asgard_alignment.ESOdevice.Lamp,
                        ):
                            value_map = {"T": "ON", "F": "OFF"}
                        elif isinstance(
                            self.instr.devices[s.device_name],
                            asgard_alignment.ESOdevice.Motor,
                        ):
                            value_map = {"T": "OPEN", "F": "CLOSED"}
                        else:
                            logging.error(
                                f"Device {s.device_name} is not lamp or shutter"
                            )
                            continue

                        # If it is a shutter do:
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": value_map[s.value]}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                    # Case of motor with absolute encoder position requested
                    if s.motion_type == "ENC":
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: if motor is at limit, do:
                        # dbMsg['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                        # Report the absolute encoder position
                        # Here (simulation), we simply use the target
                        # position (even if the motor is supposed to move)
                        pos_enc = self.instr.devices[s.device_name].read_position()
                        attribute = "<alias>" + s.device_name + ":DATA.posEnc"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": pos_enc}
                        )
                        # Case of motor with relative encoder position
                        # not considered yet
                        # The simplest would be to read the encoder position
                        # and to update the database as for the previous case

                    # Case of motor with
                    if s.motion_type == "ENCREL":
                        if is_batch_done:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": ""}
                            )
                        else:
                            reply["reply"]["parameters"].append(
                                {"attribute": attribute, "value": "MOVING"}
                            )

                        # Note: if motor is at limit, do:
                        # dbMsg['command']['parameters'].append({"attribute":attribute, "value":"LIMIT"})
                        # Report the absolute encoder position
                        # Here (simulation), we simply use the target
                        # position (even if the motor is supposed to move)
                        pos_enc = self.instr.devices[s.device_name].read_position()
                        attribute = "<alias>" + s.device_name + ":DATA.posEnc"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": pos_enc}
                        )

            # Check if second batch remains to setup
            # (if no STOP command has been sent)
            if is_batch_done:
                if (
                    (self.batch == 0)
                    and (len(self.setup_ls[1]) > 0)
                    and (not self.is_stopped)
                ):
                    self.batch = 1
                    logging.info(f"batch {self.batch} of devices to move:")
                    for s in self.setup_ls[self.batch]:
                        logging.info(
                            f"Moving: {s.device_name} to: {s.value} ( setting {s.motion_type} )"
                        )
                        self.instr.devices[s.device_name].setup(s.motion_type, s.value)

                        # Inform wag ICS that the device is moving
                        attribute = "<alias>" + s.device_name + ":DATA.status0"
                        reply["reply"]["parameters"].append(
                            {"attribute": attribute, "value": "MOVING"}
                        )

                    reply["reply"]["content"] = "PENDING"
                else:
                    # All batches of setup are done
                    reply["reply"]["content"] = "DONE"
            else:
                reply["reply"]["content"] = "PENDING"

        # Case of sensor reading request
        elif "read" in command_name:
            reply["reply"]["parameters"].clear()
            temps = self.instr.temp_summary.get_temp_status(
                probes_only=True, raw_temps=False
            )

            for t in temps:
                reply["reply"]["parameters"].append({"value": round(t, 2)})

            reply["reply"]["content"] = "OK"

        # Case of other commands. The parameters are either a list
        # of devices, or "all" to apply the command to all the devices
        else:
            reply["reply"]["parameters"].clear()
            n_devs_commanded = len(json_data["command"]["parameters"])
            is_all_devs = False
            # Check if command applies to all the existing devices
            if (n_devs_commanded == 1) and (
                json_data["command"]["parameters"][0]["device"] == "all"
            ):
                n_devs_commanded = len(self.instr.devices)  # total number of devices
                is_all_devs = True
                dev_names = list(self.instr._motor_config.keys())
            else:
                dev_names = [
                    json_data["command"]["parameters"][i]["device"]
                    for i in range(n_devs_commanded)
                ]

            # if online, it is more efficient for instrument to do it in a batch call
            if command_name == "online":
                self.instr.online(dev_names)

                for dev_name in dev_names:
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 3}
                    )

            # standby is also a weird case, as standing by some devices shuts off others - need to iterate
            if command_name == "standby":
                for dev_name in dev_names:
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 2}
                    )

                devs_to_standby = dev_names.copy()
                while len(devs_to_standby) > 0:
                    logging.info(f"Standing by device: {devs_to_standby[0]}")
                    self.instr.standby(devs_to_standby[0])

                    devs_to_standby = list(
                        set(self.instr.devices.keys()).intersection(devs_to_standby)
                    )

            # for all other commands, do them one device at a time...
            for i in range(n_devs_commanded):
                if is_all_devs:
                    try:
                        dev_name = dev_names[i]
                    except Exception as e:
                        logging.error(f"Error {e}")
                        break
                else:
                    dev_name = json_data["command"]["parameters"][i]["device"].upper()

                if command_name == "disable":
                    logging.info(f"Power off device: {dev_name}")

                    self.instr.devices[dev_name].disable()

                elif command_name == "enable":
                    logging.info(f"Power on device: {dev_name}")

                    self.instr.devices[dev_name].enable()

                elif command_name == "off":
                    logging.info(f"Turning off device: {dev_name}")
                    # .........................................................
                    # If needed, call controller-specific functions to power
                    # down the device. It may require initialization
                    # after a power up
                    # .........................................................

                    # Update the wagics database to show that the device is
                    # in LOADED state (value of "state" attribute has to be
                    # set to 3)

                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif command_name == "simulat":
                    logging.info(f"Simulation of device {dev_name}")
                    # Set the simulation flag of dev_name to 1
                    # TODO: add code here that changes the device to simulation mode
                    # for devIdx in range(nbCtrlDevs):
                    #     if d[devIdx].name == dev_name:
                    #         break
                    # d[devIdx].simulated = 1

                    # Update the wagics database  to show that the device
                    # is in simulation and is in LOADED state

                    attribute = "<alias>" + dev_name + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

                elif command_name == "stop":
                    logging.info(f"Stop device: {dev_name}")
                    logging.info("ignored")
                    # self.instr.devices[dev_name].stop()

                    # If setup is in progress, consider it done

                    # Update of the device status (positions, etc...) will be
                    # done by the next "poll" command sent by wag

                elif command_name == "stopsim":
                    logging.info(f"Normal mode for device {dev_name}")
                    # Set the simulation flag of dev_name to 0
                    # TODO: add code here that changes the device to normal mode

                    # Update the wagics database  to show that the device
                    # is not in simulation and is in LOADED state
                    # (it may require an initialization when going to
                    # ONLINE state)

                    attribute = "<alias>" + dev_name + ".simulation"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 0}
                    )
                    attribute = "<alias>" + dev_name + ".state"
                    reply["reply"]["parameters"].append(
                        {"attribute": attribute, "value": 1}
                    )

            if command_name == "stop":
                self.is_stopped = True

            reply["reply"]["content"] = "OK"

        # Send back reply to ic0fb process (wag)
        reply["reply"]["time"] = self.get_time_stamp()

        # Convert reply JSON structure into a character string
        # terminated with null character (because ic0fb process on wag
        # in coded in C++ and needs null character to mark end of the string)

        repMsg = json.dumps(reply) + "\0"
        logging.info(json.dumps(reply))
        # self.server.send_string(repMsg)

        return False, repMsg

    def _handle_custom_command(self, message):
        # this is a custom command, acutally do useful things here lol
        def read_msg(axis):
            return str(self.instr.devices[axis].read_position())

        def stop_msg(axis):
            return str(self.instr.devices[axis].stop())

        def moveabs_msg(axis, position):
            self.instr.devices[axis].move_abs(float(position))
            return "ACK"

        def connected_msg(axis):
            return "connected" if axis in self.instr.devices else "not connected"

        def connect_msg(axis):
            # this is a connection open request
            logging.info(f"attempting open connection to {axis}")
            res = self.instr._attempt_to_open(axis, recheck_ports=True)
            logging.info(f"attempted to open {axis} with result {res}")

            return "connected" if axis in self.instr.devices else "not connected"

        def home_steppers_msg(motor):
            if motor == "all":
                motor = list(asgard_alignment.controllino.STEPPER_NAME_TO_NUM.keys())
            else:
                motor = [motor]

            self.instr.home_steppers(motor)

        def init_msg(axis):
            self.instr.devices[axis].init()
            return "ACK"

        def tt_step_msg(axis, n_steps):
            """
            Move the tip-tilt stage by n_steps.
            """
            if "HT" not in axis:
                raise ValueError(f"{axis} is not a valid tip-tilt stage")

            if axis not in self.instr.devices:
                raise ValueError(f"{axis} not found in instrument")

            n_steps = int(n_steps)

            self.instr.devices[axis].move_stepping(n_steps)

        def tt_config_step_msg(axis, step_size):
            if "HT" not in axis:
                raise ValueError(f"{axis} is not a valid tip-tilt stage")
            self.instr.devices[axis].config_step_size(int(step_size))

        def moverel_msg(axis, position):
            logging.info(f"moverel {axis} {position}")
            self.instr.devices[axis].move_relative(float(position))
            return "ACK"

        def state_msg(axis):
            return self.instr.devices[axis].read_state()

        def save_msg(subset, fname):
            if subset.lower() not in ["heimdallr", "baldr", "solarstein", "all"]:
                return "NACK: Invalid subset, must be 'heimdallr', 'baldr', 'solarstein' or 'all'"

            return self.instr.save(subset.lower(), fname)

        def ping_msg(axis):
            res = self.instr.ping_connection(axis)

            if res:
                return "ACK: connected"
            else:
                return "NACK: not connected"

        def health_msg():
            """
            check the health of the whole instrument, and return a json list of dicts
            to make a table, with columns
            - axis name,
            - motor type,
            - connected,
            - state,
            """

            health = self.instr.health()

            # convert to string
            health_str = json.dumps(health)

            return health_str

        def mv_img_msg(config, beam_number, x, y):
            try:
                res = self.instr.move_image(config, int(beam_number), x, y)
            except ValueError as e:
                return f"NACK: {e}"

            if res:
                return "ACK: moved"
            else:
                return "NACK: not moved"

        def mv_pup_msg(config, beam_number, x, y):
            logging.info(f"{beam_number} {type(beam_number)}")
            try:
                res = self.instr.move_pupil(config, int(beam_number), x, y)
            except ValueError as e:
                return f"NACK: {e}"

            if res:
                return "ACK: moved"
            else:
                return "NACK: not moved"

        def on_msg(lamp_name):
            self.instr.devices[lamp_name].turn_on()
            return "ACK"

        def off_msg(lamp_name):
            self.instr.devices[lamp_name].turn_off()
            return "ACK"

        def is_on_msg(lamp_name):
            return str(self.instr.devices[lamp_name].is_on())

        def reset_msg(axis):
            try:
                self.instr.devices[axis].reset()
                return "ACK"
            except Exception as e:
                return f"NACK: {e}"

        def asg_setup_msg(axis, mtype, value):
            try:
                # if it is a float, convert it to a python float
                try:
                    value = float(value)
                except ValueError:
                    pass
                self.instr.devices[axis].setup(mtype, value)
                return "ACK"
            except Exception as e:
                return f"NACK: {e}"

        def apply_flat_msg(dm_name):
            if dm_name not in self.instr.devices:
                return f"NACK: DM {dm_name} not found"

            # Retrieve the DM instance and its flat map
            dm_device = self.instr.devices[dm_name]
            # dm = dm_device["dm"]
            # flat_map = dm_device["flat_map"]

            # Apply the flat map to the DM
            dm_device["dm"].send_data(dm_device["flat_map"])

            logging.info(f"Flat map applied to {dm_name}")
            return f"ACK: Flat map applied to {dm_name}"

        def apply_cross_msg(dm_name):
            if dm_name not in self.instr.devices:
                return f"NACK: DM {dm_name} not found"

            # Retrieve the DM instance and its flat map
            dm_device = self.instr.devices[dm_name]
            # dm = dm_device["dm"]
            # flat_map = dm_device["flat_map"]

            # Apply the flat map to the DM
            dm_device["dm"].send_data(
                dm_device["flat_map"] + 0.3 * dm_device["cross_map"]
            )

            logging.info(f"Cross map applied to {dm_name}")
            return f"ACK: Cross map applied to  {dm_name}"

        def fpm_get_savepath_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.savepath
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].savepath

        def fpm_mask_positions_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.mask_positions
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].mask_positions

        def fpm_update_position_file_msg(axis, filename):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_position_file(filename)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].update_position_file(filename)
            #     return "ACK"

        def fpm_move_to_phasemask_msg(axis, maskname):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_to_mask(maskname)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_to_mask(maskname)
            #     return "ACK"

        def fpm_move_relative_msg(axis, new_pos):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_relative(new_pos)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_relative(new_pos)
            #     return "ACK"

        def fpm_move_absolute_msg(axis, new_pos):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.move_absolute(new_pos)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].move_absolute(new_pos)
            #     return "ACK"

        def fpm_read_position_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                return device.read_position()
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     return self.instr.devices[axis].read_position()

        def fpm_update_mask_position_msg(axis, mask_name):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_mask_position(mask_name)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Mask {mask_name} not found"
            # else:
            #     self.instr.devices[axis].update_mask_position(mask_name)
            #     return "ACK"

        def fpm_offset_all_mask_positions_msg(axis, rel_offset_x, rel_offset_y):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.offset_all_mask_positions(rel_offset_x, rel_offset_y)
                return "ACK"
            # if axis not in self.instr.devices:
            #     return f"NACK: Axis {axis} not found"
            # else:
            #     self.instr.devices[axis].offset_all_mask_positions(
            #         rel_offset_x, rel_offset_y
            #     )
            #     return "ACK"

        def fpm_write_mask_positions_msg(axis):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.write_current_mask_positions()
                return "ACK"

        def fpm_update_all_mask_positions_relative_to_current_msg(
            axis, current_mask_name, reference_mask_position_file
        ):
            device = self.instr.all_devices[axis]
            if device is None:
                return f"NACK: Axis {axis} not found"
            else:
                device.update_all_mask_positions_relative_to_current(
                    current_mask_name, reference_mask_position_file, write_file=False
                )
                return "ACK"

        def standby_msg(axis):
            return self.instr.standby(axis)

        def online_msg(axes):
            # parse axes into list
            axis_list = axes.split(",")
            return self.instr.online(axis_list)

        def h_shut_msg(state, beam_numbers):
            if beam_numbers == "all":
                beam_numbers = list(range(1, 5))
            else:
                beam_numbers = [int(b) for b in beam_numbers.split(",")]

            if state not in ["open", "close"]:
                return "NACK: Invalid state for h_shut, must be 'open' or 'close'"

            return self.instr.h_shut(state, beam_numbers)

        def h_splay_msg(state):
            return self.instr.h_splay(state)

        def temp_status_msg(mode):
            """
            Get the temperature status of the instrument.
            Returns a list of values in order, see instrument documentation.
            """
            if mode == "now":
                return str(self.instr.temp_summary.get_temp_status())
            if mode == "keys":
                keys = self.instr.temp_summary.get_temp_keys()
                return f"[{','.join(keys)}]"
            return "NACK: Invalid mode for temp_status, must be 'now' or 'keys'"

        def set_kaya_msg(state):
            if state not in ["on", "off"]:
                return "NACK: Invalid state for set_kaya, must be 'on' or 'off'"

            self.instr.set_kaya(state)
            return "ACK"

        def home_rotm(idxes: list = [-1]):
            """
            Home the baldr ADCs and/or the HPOLs

            """
            # TODO clever homing that knows which way to go and/or doesnt spin around the full way...

        def rotm_disable():
            """
            Disable all rotation stage motors
            """
            self.instr._controllers["rotm_teensy"].disable_all()
            return "ACK"

        def rotm_slew(adc_set, reltarget):
            """
            Enable motor set U or L, then move them relative
            """
            reltarget = int(reltarget)

            if adc_set not in asgard_alignment.controllino.STEPPER_GROUPS.keys():
                return f"NACK: Invalid ADC set {adc_set}, must be one of {list(asgard_alignment.controllino.STEPPER_GROUPS.keys())}"

            self.instr._controllers["rotm_teensy"].disable_all()

            self.instr._controllers["rotm_teensy"].enable_subset(adc_set)
            self.instr._controllers["rotm_teensy"].move_enabled_relative(reltarget)

            return "ACK"

        def status():
            return "ACK"

        @dataclass
        class Command:
            """Metadata for a command"""

            info: str
            format_str: str
            func: Callable

        commands = {
            "read": Command(
                info="read {axis} - read the position of the given axis",
                format_str="read {}",
                func=read_msg,
            ),
            "stop": Command(
                info="stop {axis} - stop movement of the given axis",
                format_str="stop {}",
                func=stop_msg,
            ),
            "moveabs": Command(
                info="moveabs {axis} {position} - move axis to absolute position",
                format_str="moveabs {} {:f}",
                func=moveabs_msg,
            ),
            "connected?": Command(
                info="connected? {axis} - check if axis is connected",
                format_str="connected? {}",
                func=connected_msg,
            ),
            "connect": Command(
                info="connect {axis} - attempt to open connection to axis",
                format_str="connect {}",
                func=connect_msg,
            ),
            "init": Command(
                info="init {axis} - initialize the given axis",
                format_str="init {}",
                func=init_msg,
            ),
            "tt_step": Command(
                info="tt_step {axis} {n_steps} - move tip-tilt stage by n_steps",
                format_str="tt_step {} {}",
                func=tt_step_msg,
            ),
            "tt_config_step": Command(
                info="tt_config_step {axis} {step_size} - configure tip-tilt step size",
                format_str="tt_config_step {} {}",
                func=tt_config_step_msg,
            ),
            "moverel": Command(
                info="moverel {axis} {position} - move axis by relative position",
                format_str="moverel {} {:f}",
                func=moverel_msg,
            ),
            "state": Command(
                info="state {axis} - read the state of the given axis",
                format_str="state {}",
                func=state_msg,
            ),
            "save": Command(
                info="save {subset} {filename} - save instrument state to file (subset: heimdallr, baldr, solarstein, or all)",
                format_str="save {} {}",
                func=save_msg,
            ),
            "dmapplyflat": Command(
                info="dmapplyflat {dm_name} - apply flat map to deformable mirror",
                format_str="dmapplyflat {}",
                func=apply_flat_msg,
            ),
            "dmapplycross": Command(
                info="dmapplycross {dm_name} - apply cross map to deformable mirror",
                format_str="dmapplycross {}",
                func=apply_cross_msg,
            ),
            "fpm_getsavepath": Command(
                info="fpm_getsavepath {axis} - get save path for focal plane mask",
                format_str="fpm_getsavepath {}",
                func=fpm_get_savepath_msg,
            ),
            "fpm_maskpositions": Command(
                info="fpm_maskpositions {axis} - get focal plane mask positions",
                format_str="fpm_maskpositions {}",
                func=fpm_mask_positions_msg,
            ),
            "fpm_movetomask": Command(
                info="fpm_movetomask {axis} {maskname} - move focal plane mask to named position",
                format_str="fpm_movetomask {} {}",
                func=fpm_move_to_phasemask_msg,
            ),
            "fpm_moverel": Command(
                info="fpm_moverel {axis} {new_pos} - move focal plane mask by relative position",
                format_str="fpm_moverel {} {}",
                func=fpm_move_relative_msg,
            ),
            "fpm_moveabs": Command(
                info="fpm_moveabs {axis} {new_pos} - move focal plane mask to absolute position",
                format_str="fpm_moveabs {} {}",
                func=fpm_move_absolute_msg,
            ),
            "fpm_readpos": Command(
                info="fpm_readpos {axis} - read focal plane mask position",
                format_str="fpm_readpos {}",
                func=fpm_read_position_msg,
            ),
            "fpm_update_position_file": Command(
                info="fpm_update_position_file {axis} {filename} - update focal plane mask position file",
                format_str="fpm_update_position_file {} {}",
                func=fpm_update_position_file_msg,
            ),
            "fpm_updatemaskpos": Command(
                info="fpm_updatemaskpos {axis} {mask_name} - update focal plane mask position",
                format_str="fpm_updatemaskpos {} {}",
                func=fpm_update_mask_position_msg,
            ),
            "fpm_offsetallmaskpositions": Command(
                info="fpm_offsetallmaskpositions {axis} {rel_offset_x} {rel_offset_y} - offset all focal plane mask positions",
                format_str="fpm_offsetallmaskpositions {} {} {}",
                func=fpm_offset_all_mask_positions_msg,
            ),
            "fpm_writemaskpos": Command(
                info="fpm_writemaskpos {axis} - write focal plane mask positions to file",
                format_str="fpm_writemaskpos {}",
                func=fpm_write_mask_positions_msg,
            ),
            "fpm_updateallmaskpos": Command(
                info="fpm_updateallmaskpos {axis} {current_mask_name} {reference_mask_position_file} - update all focal plane mask positions relative to current",
                format_str="fpm_updateallmaskpos {} {} {}",
                func=fpm_update_all_mask_positions_relative_to_current_msg,
            ),
            "ping": Command(
                info="ping {axis} - ping connection to axis",
                format_str="ping {}",
                func=ping_msg,
            ),
            "health": Command(
                info="health - check health of the whole instrument",
                format_str="health",
                func=health_msg,
            ),
            "on": Command(
                info="on {lamp_name} - turn on lamp",
                format_str="on {}",
                func=on_msg,
            ),
            "off": Command(
                info="off {lamp_name} - turn off lamp",
                format_str="off {}",
                func=off_msg,
            ),
            "is_on": Command(
                info="is_on {lamp_name} - check if lamp is on",
                format_str="is_on {}",
                func=is_on_msg,
            ),
            "reset": Command(
                info="reset {axis} - reset the given axis",
                format_str="reset {}",
                func=reset_msg,
            ),
            "mv_img": Command(
                info="mv_img {config} {beam_number} {x} {y} - move image for given config and beam",
                format_str="mv_img {} {} {:f} {:f}",
                func=mv_img_msg,
            ),
            "mv_pup": Command(
                info="mv_pup {config} {beam_number} {x} {y} - move pupil for given config and beam",
                format_str="mv_pup {} {} {:f} {:f}",
                func=mv_pup_msg,
            ),
            "asg_setup": Command(
                info="asg_setup {axis} {mtype} {value} - setup axis with motion type and value",
                format_str="asg_setup {} {} {}",
                func=asg_setup_msg,
            ),
            "home_steppers": Command(
                info="home_steppers {motor} - home stepper motors (motor name or 'all')",
                format_str="home_steppers {}",
                func=home_steppers_msg,
            ),
            "standby": Command(
                info="standby {axis} - put axis into standby mode",
                format_str="standby {}",
                func=standby_msg,
            ),
            "online": Command(
                info="online {axes} - bring axes online (comma-separated list)",
                format_str="online {}",
                func=online_msg,
            ),
            "h_shut": Command(
                info="h_shut {state} {beam_numbers} - control heimdallr shutter (state: open/close, beam_numbers: comma-separated or 'all')",
                format_str="h_shut {} {}",
                func=h_shut_msg,
            ),
            "h_splay": Command(
                info="h_splay {state} - control heimdallr splay",
                format_str="h_splay {}",
                func=h_splay_msg,
            ),
            "temp_status": Command(
                info="temp_status {mode} - get temperature status (mode: 'now' or 'keys')",
                format_str="temp_status {}",
                func=temp_status_msg,
            ),
            "set_kaya": Command(
                info="set_kaya {state} - set kaya state (state: on/off)",
                format_str="set_kaya {}",
                func=set_kaya_msg,
            ),
            "rotm_disable": Command(
                info="rotm_disable - disable all rotation stage motors",
                format_str="rotm_disable",
                func=rotm_disable,
            ),
            "rotm_slew": Command(
                info="rotm_slew {adc_set} {reltarget} - enable and move rotation motor set (adc_set: U or L)",
                format_str="rotm_slew {} {}",
                func=rotm_slew,
            ),
            "status": Command(
                info="status - get system status",
                format_str="status",
                func=status,
            ),
            "command_names": Command(
                info="command_names - list all available commands",
                format_str="command_names",
                func=lambda: f'"{list(commands.keys())}"',
            ),
        }

        try:
            first_word = message.split(" ")[0]
            if first_word in commands:
                cmd = commands[first_word]
                result = parse(cmd.format_str, message)
                return cmd.func(*result)
            else:
                return "NACK: Unkown custom command"

            # old
            # for pattern, func in patterns.items():
            #     result = parse(pattern, message)
            #     if result:
            #         return func(*result)
        except Exception as e:
            first_word = message.split(" ")[0]
            cmd = commands.get(first_word)
            help_text = cmd.info if cmd else "Unknown command"
            logging.error(f"Custom command error: {e}")
            return f"NACK: command usage is: {help_text} \n {e}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the MDS server.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--host", type=str, default="192.168.100.2", help="Host address"
    )
    parser.add_argument(
        "--log-location",
        type=str,
        default="~/logs/mds/",
        help="Path to the log directory",
    )
    parser.add_argument("-p", "--port", type=int, default=5555, help="Port number")

    args = parser.parse_args()

    # logname from the current time
    log_fname = (
        datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d_%H-%M-%S")
        + ".log"
    )
    log_path = os.path.join(os.path.expanduser(args.log_location), log_fname)

    # Remove all handlers associated with the root logger object (if any)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler with ms precision
    file_handler = logging.FileHandler(log_path)
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Console handler with same formatter
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    serv = MultiDeviceServer(args.port, args.host, args.config)
    serv.run()
