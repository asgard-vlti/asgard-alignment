import asgard_alignment.controllino as co
import time


def main():
    """Power-cycle the Kaya device through the Controllino."""
    cc = co.PowerControllino("192.168.100.10", init_motors=False)

    res = cc.turn_off("Kaya")
    if not res:
        print("Kaya was already off")

    time.sleep(2)
    res = cc.turn_on("Kaya")
    if not res:
        print("Failed to turn on Kaya")

    print("Kaya restarted")
