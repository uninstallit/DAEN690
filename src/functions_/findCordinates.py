import pandas as pd
import re

def clean(text):
    # text = text.replace(/[^\x20-\x7E]/gmi, ' '); //replace line breaks with spaces
    text = re.sub('/[\r\n]+/gm', ' ', text)
    print(f'text1: {text}')
    text2 = re.sub('/^\s+|\s+$|\s+(?=\s)/g', '', text)
    print(f'text2: {text2}')
    return text2
    #return text.replace(/[^\x20-\x7E]/gmi, ' ').trim().replace('  ', ' ');


def subStr(str, start, end):
    return str[start:end]

#returns a [lat,lng] from most common notam coordinate formats
def getCoords(coordinates):
    if len(coordinates) == 14:   #ex: 4025N00404W999
        lat = subStr(coordinates, 0, 2) + subStr(coordinates, 2, 4) + subStr(coordinates,4, 5)
        lng = subStr(coordinates, 5, 8) + subStr(coordinates, 8, 10) + subStr(coordinates, 10, 11)
        return [{'lat':lat, 'long':lng,  'radius': subStr(coordinates,11, 14) }]

    if len(coordinates) == 15:  # ex: 402500N 075000W           
        lat = subStr(coordinates, 0, 2) + subStr(coordinates, 2, 4) + subStr(coordinates,4, 6) + subStr(coordinates,6, 7)
        lng = subStr(coordinates,8, 10) + subStr(coordinates,10, 12) + subStr(coordinates, 12, 14) + subStr(coordinates,14, 15)
        return [{'lat':lat, 'long':lng}]

    if len(coordinates)== 16:  #ex: 280944N 0152630W  
        lat = subStr(coordinates,0, 2) + subStr(coordinates,2, 4) + subStr(coordinates,4, 5) + subStr(coordinates,5, 16)
        lng = subStr(coordinates,8, 11) + subStr(coordinates,11, 13) + subStr(coordinates,13, 15) + subStr(coordinates,15, 16)
        return [{'lat':lat, 'long':lng}]
    if len(coordinates)== 22: # ex: 411415.07N 0083700.29W    
        lat = subStr(coordinates,0, 2) + subStr(coordinates,2, 4) + subStr(coordinates,4, 9) + subStr(coordinates,9, 10)
        lng = subStr(coordinates,11, 14) + subStr(coordinates,14, 16) + subStr(coordinates,16, 21) + subStr(coordinates,21, 22)
        return [{'lat':lat, 'long':lng}]
    
    else:
        return []

def findCoordinatesFromQcode(q):
    coordinates = ''
    cols = q.split('/')
    if len(cols) == 8:
        coordinates = cols[7]
        return getCoords(coordinates)
    return []


def findCoordinatesFromAnyStr(str):
    print('findCoordinatesFromAnyStr')
    text = clean(str)
    return text
    # res = []
    # if len(text) >= 15):
    #     #ex: 402500N 075000W       
    #     #ex: 412644N 0083144W  
    #     #ex: 411415.07N 0083700.29W 

    # return res

def main():
    q ="LOVV/QWPLW/IV/BO/W/000/130/4809N01610E001"
    latlon = findCoordinatesFromQcode(q)
    print(f'{latlon}')

    anyStr = "GLIDER FLYING (PARAGLIDING WORLD CUP) WILL TAKE PLACE ON AREA (MANTEIGAS): 405500N 0073000W (VILA DA PONTE) - 405735N 0065231W (ESCALHAO) - ALONG PORTUGUESE/SPANISH BOUNDARY - 393950N 0073230W (MONTE FIDALGO) - 402500N 075000W (TRAVANCINHA) -  405500N 0073000W (VILA DA PONTE)."
    cleanText = findCoordinatesFromAnyStr(anyStr)
    print(cleanText)

    # launch_df = pd.read_csv("./data/launch.csv")
    # print(launch_df.head())

    # notam_msg = launch_df["NOTAM Condition/LTA subject/Construction graphic title"][0]
    # notam = notam_msg_parser(notam_msg)

    # for v in vars(notam):
    #     print(v)


if __name__ == "__main__":
    main()
