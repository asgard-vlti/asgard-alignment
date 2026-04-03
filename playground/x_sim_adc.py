# %%
import asgard_alignment.adc_fns as af
from astropy.utils import iers
import datetime
import astropy
import matplotlib.pyplot as plt
import numpy as np

iers.conf.auto_download = False
print(iers.conf.iers_degraded_accuracy)
iers.conf.iers_degraded_accuracy = "warn"

# %%
config_file = "/home/taras/Documents/usyd/asgard/asgard-alignment/config_files/adc_slew_config.toml"

consts = af.load_config(config_file)

# %%
sim_time = datetime.datetime(2026, 4, 2, 2, 40, 43, tzinfo=datetime.timezone.utc)
# 	10 10 53.14,	-80 39 20.07
ra = "101053.14"
dec = "-803920.07"

# target 2:
sim_time = datetime.datetime(2026, 4, 2, 3, 16, 43, tzinfo=datetime.timezone.utc)
# 	14 57 33.25	-00 10 03.40
ra = "145733.25"
dec = "-001003.40"

# target 3:
# sim_time = datetime.datetime(2026, 4, 2, 3, 36, 43, tzinfo=datetime.timezone.utc)
# # 11 27 53.18	35 22 14.03
# ra = "112753.18"
# dec = "+352214.03"


targets = [
    {
        "ra": "101053.14",
        "dec": "-803920.07",
        "time": datetime.datetime(2026, 4, 2, 2, 40, 43, tzinfo=datetime.timezone.utc),
    },
    {
        "ra": "145733.25",
        "dec": "-001003.40",
        "time": datetime.datetime(2026, 4, 2, 3, 16, 43, tzinfo=datetime.timezone.utc),
    },
    {
        "ra": "112753.18",
        "dec": "+352214.03",
        "time": datetime.datetime(2026, 4, 2, 3, 36, 43, tzinfo=datetime.timezone.utc),
    },
]

for target in targets:
    plt.figure()
    ra = target["ra"]
    dec = target["dec"]
    sim_time = target["time"]

    ra, dec = af.parse_ra_dec(ra, dec)
    alt, az = af.ra_dec_to_altaz(ra, dec, consts, sim_time)

    # # high target:
    # alt = 89.9
    # az = 0.0

    adc_upper, adc_lower = af.calculate_adc_targets(alt, az, consts)

    print(adc_upper, adc_lower)
    print((adc_upper + 18000) % 36000 - 18000, (adc_lower + 18000) % 36000 - 18000)

    vec_lower = af.angle_to_vector(adc_lower)
    vec_upper = af.angle_to_vector(18000 - adc_upper)

    resultant_vec = vec_lower + vec_upper

    adc_vec_color = "C0"
    resultant_color = "C1"

    # plot lower adc vector
    plt.arrow(
        0,
        0,
        vec_lower[0],
        vec_lower[1],
        head_width=0.1,
        head_length=0.1,
        fc=adc_vec_color,
        ec=adc_vec_color,
        label="ADC Lower",
    )
    # plot upper adc vector starting from tip of lower adc vector
    plt.arrow(
        vec_lower[0],
        vec_lower[1],
        vec_upper[0],
        vec_upper[1],
        head_width=0.1,
        head_length=0.1,
        fc=adc_vec_color,
        ec=adc_vec_color,
        label="ADC Upper",
    )
    # plot resultant vector from origin
    plt.arrow(
        0,
        0,
        resultant_vec[0],
        resultant_vec[1],
        head_width=0.1,
        head_length=0.1,
        fc=resultant_color,
        ec=resultant_color,
        label="Resultant",
    )

    plt.grid()
    plt.axis("equal")

    plt.title(
        f"ADC positions are adc_upper={adc_upper} centidegrees, adc_lower={adc_lower} centidegrees"
    )

    print(
        f"Angle: {np.degrees(np.arctan(resultant_vec[1] / resultant_vec[0]))} degrees"
    )

# %%
