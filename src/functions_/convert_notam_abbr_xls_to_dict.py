import sys
sys.path.insert(0,'../../')


from libs.PyNotam._abbr import ICAO_abbr
from libs.faa_7340_2K_notam_abbr import ICAO_faa_7340_2K_abbr

import pandas as pd

def covert_xlsx_to_dict(from_xlsx_file, tab_count, to_py_file, dict_name):
    # xlsx to df
    tables = [f'Table {i}'  for i in range(1,tab_count)]
    sheets = pd.read_excel(from_xlsx_file, engine='openpyxl', sheet_name=tables)
    
    # df to dict
    dict ={}
    for sheet in tables:
        df = sheets.get(sheet)
        for i in range(0,len(df)):
            id = df.iloc[i][0]
            val =  df.iloc[i][1]
            dict[id]=val 

    # dict to py file in order to verify the key value 
    with open(to_py_file, 'w') as f:
        dict_var = f'{dict_name}' + '= {\n'
        f.write(dict_var)
        for key, value in dict.items(): 
            f.write('\'%s\':\'%s\',\n' % (key, value))
        f.write('}')


def create_faa_7340_2K_abbr_py_file():
    tab_count= 54
    from_xlsx_file = '../../libs/faa_7340_2K_notam_abbr.xlsx'
    to_py_file= '../../libs/faa_7340_2K_notam_abbr.py'
    to_dict_name = 'ICAO_faa_7340_2K_abbr'
    covert_xlsx_to_dict(from_xlsx_file, tab_count, to_py_file, to_dict_name)


def write_faa_7340_2K_abbr_py_file_to_final_abbr_py ():
    print('write_new_faa_7340_2K_abbr_py_file_to_final_abbr_py')
    
    for key, val in ICAO_faa_7340_2K_abbr.items():
          ICAO_abbr[key] = val
            
    # write new dict to _abbr.py file to replace the original _abb.py in PyNotam lib
    with open('../../libs/PyNotam/_abbr.py', 'w') as f:
        f.write('ICAO_abbr = {\n')
        for key, value in ICAO_abbr.items(): 
            f.write('\'%s\':\'%s\',\n' % (key, value))
        f.write('}')
   
def main():
    # create_faa_7340_2K_abbr_py_file()
    write_faa_7340_2K_abbr_py_file_to_final_abbr_py()

if __name__ == "__main__":
    main()

