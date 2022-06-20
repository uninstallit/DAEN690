# import sys
# sys.path.append('./')
import pandas as pd

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
src = os.path.dirname(current)
DAEN690 = os.path.dirname(src)
sys.path.append(DAEN690)

from libs.PyNotam.notam import Notam

def parse_raw_notam():
    s = """(A1912/15 NOTAMN
            Q) LOVV/QWPLW/IV/BO/W/000/130/4809N01610E001
            A) LOVV B) 1509261100 C) 1509261230
            E) PJE WILL TAKE PLACE AT AREA LAAB IN WALDE
            PSN:N480930 E0161028 RADIUS - 1NM
            F) GND G) FL130)"""
    n = Notam.from_str(s)
    print(f'valid:{n.valid_from}, \narea:{n.area}, \nplain_language:{n.decoded()}')


def main():
    parse_raw_notam()
    

if __name__ == "__main__":
    main()
