import sys
sys.path.insert(0,'../../')
from libs.PyNotam._abbr import ICAO_abbr
from libs.faa_notam_abbr import ICAO_faa_abbr

import pandas as pd

def covert_xlsx_to_dict():
    # reading 12 sheets from the xlsx file
    file_name = '../../libs/faa_notam_abbr.xlsx'
    # xlsx to df
    tables = [f'Table {i}'  for i in range(1,13)]
    sheets = pd.read_excel(file_name, engine='openpyxl', sheet_name=tables)
    
    # df to dict
    dict ={}
    for sheet in tables:
        df = sheets.get(sheet)
        for i in range(0,len(df)):
            id = df.iloc[i][0]
            val =  df.iloc[i][1]
            dict[id]=val 
   
    # dict to file
    with open('../../libs/faa_notam_abbr.py', 'w') as f:
        f.write('ICAO_faa_abbr = {\n')
        for key, value in dict.items(): 
            f.write('\'%s\':\'%s\',\n' % (key, value))
        f.write('}')

def combine_faa_dict_to_PyNotam_dict():
    for key, val in ICAO_faa_abbr.items():
        ICAO_abbr[key] = val

    # write the combined keys back to _appr.py file
    with open('../../libs/PyNotam/_abbr.py', 'w') as f:
        f.write('ICAO_abbr = {\n')
        for key, value in ICAO_abbr.items(): 
            f.write('\'%s\':\'%s\',\n' % (key, value))
        f.write('}')
    
   
# uncomment to call the functions
#covert_xlsx_to_dict()
#combine_faa_dict_to_PyNotam_dict()




