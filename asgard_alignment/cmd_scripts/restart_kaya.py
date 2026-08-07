import asgard_alignment.controllino as co
import time


def main():
    """Power-cycle the Kaya device through the Controllino."""
    cc = co.PowerControllino("192.168.100.10", init_motors=False)

    cc.turn_off("Kaya")
    time.sleep(2)
    cc.turn_on("Kaya")

    print("Kaya restarted")
