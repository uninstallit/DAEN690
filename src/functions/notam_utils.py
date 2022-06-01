import sys
sys.path.append('./')

import pandas as pd
from libs.PyNotam.notam import Notam

def notam_msg_parser(msg):
    msg = "/".join([parts.strip() for parts in f"({msg})".split("/")])
    notam = Notam.from_str(msg)
    return notam


def main():

    launch_df = pd.read_csv("./data/launch.csv")
    print(launch_df.head())

    notam_msg = launch_df["NOTAM Condition/LTA subject/Construction graphic title"][0]
    notam = notam_msg_parser(notam_msg)

    for v in vars(notam):
        print(v)


if __name__ == "__main__":
    main()
