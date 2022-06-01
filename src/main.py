import pandas as pd
import functions.notam_utils as notam

# info: https://www.theairlinepilots.com/flightplanningforairlinepilots/notamdecode.php

def main():

    launch_df = pd.read_csv("./data/launch.csv")
    print(launch_df.head())

    notam_msg = launch_df["NOTAM Condition/LTA subject/Construction graphic title"][0]
    notam_obj = notam.notam_msg_parser(notam_msg)

    # for v in vars(notam_obj):
    #     print(v)

    # print(notam_obj.purpose)
    # print(notam_obj.scope)
    # print(notam_obj.full_text)
    print(notam_obj.body)


    


if __name__ == "__main__":
    main()
