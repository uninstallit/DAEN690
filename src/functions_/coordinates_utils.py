from shapely.geometry import Polygon, Point, LineString
from geographiclib.geodesic import Geodesic

# def clean(text):
#     # text = text.replace(/[^\x20-\x7E]/gmi, ' '); //replace line breaks with spaces
#     text = re.sub('/[\r\n]+/gm', ' ', text)
#     print(f'text1: {text}')
#     text2 = re.sub('/^\s+|\s+$|\s+(?=\s)/g', '', text)
#     print(f'text2: {text2}')
#     return text2
#     #return text.replace(/[^\x20-\x7E]/gmi, ' ').trim().replace('  ', ' ');


def floatStr(str, start, end):
    subStr = str[start:end] # ie 2201 ==> 22.01
    return subStr[0:2] + '.' + subStr[2:4]

def subStr(str, start, end):
    return str[start:end]

# convert lat lon to Degree Decimal
def convert_DMS_to_DD(degrees, minutes, seconds, direction):    
    dd = int(degrees) + int(minutes) / 60 + float(seconds) / (60 * 60)
    if direction == "S" or direction == "W":
        dd = dd * -1
    return round(dd,6) 

#returns [lat,lng, radius] from most common notam coordinate formats
def get_DDcoords(coordinates):

    if len(coordinates) == 14:   #ie: 4025N00404W999
        lat = convert_DMS_to_DD(subStr(coordinates, 0, 2), subStr(coordinates, 2, 4), 0, subStr(coordinates,4, 5))
        lng = convert_DMS_to_DD(subStr(coordinates, 5, 8), subStr(coordinates, 8, 10), 0, subStr(coordinates, 10, 11))
        return {'lat':lat, 'lon':lng,  'radius': round((float(subStr(coordinates,11, 14))), 2) }

    if len(coordinates) == 15:  # ie: 402500N 075000W           
        lat = convert_DMS_to_DD(subStr(coordinates, 0, 2),subStr(coordinates, 2, 4), subStr(coordinates,4, 6), subStr(coordinates,6, 7))
        lng = convert_DMS_to_DD(subStr(coordinates,8, 10), subStr(coordinates,10, 12), subStr(coordinates, 12, 14), subStr(coordinates,14, 15))
        return {'lat':lat, 'lon':lng, 'radius':None}

    if len(coordinates)== 16:  #ie: 280944N 0152630W  
        lat = convert_DMS_to_DD(subStr(coordinates,0, 2), subStr(coordinates,2, 4), subStr(coordinates,4, 5), subStr(coordinates,5, 16))
        lng = convert_DMS_to_DD(subStr(coordinates,8, 11), subStr(coordinates,11, 13), subStr(coordinates,13, 15), subStr(coordinates,15, 16))
        return {'lat':lat, 'lon':lng, 'radius':None}

    if len(coordinates)== 22: # ie: 411415.07N 0083700.29W    
        lat = convert_DMS_to_DD(subStr(coordinates,0, 2), subStr(coordinates,2, 4), subStr(coordinates,4, 9), subStr(coordinates,9, 10))
        lng = convert_DMS_to_DD(subStr(coordinates,11, 14), subStr(coordinates,14, 16), subStr(coordinates,16, 21), subStr(coordinates,21, 22))
        return {'lat':lat, 'lon':lng, 'radius':None}

    if len(coordinates) == 20: # ie: 37053691N 074272201W
        lat = convert_DMS_to_DD(subStr(coordinates,0, 2), subStr(coordinates,2, 4), floatStr(coordinates,4, 8), subStr(coordinates,8, 9))
        lng = convert_DMS_to_DD(subStr(coordinates,10, 13), subStr(coordinates,13, 15), floatStr(coordinates,15, 19), subStr(coordinates,19, 20))
        return {'lat':lat, 'lon':lng, 'radius':None}

    else:
        return {}


def calculate_centroid_and_radius(vertices):
    geod = Geodesic.WGS84 

    if len(vertices) == 1:
        point = Point(vertices)
        centroid = point.centroid
        return (round(centroid.x, 6),round(centroid.y, 6), 0) # just a point so radius is 0
    
    if len(vertices) == 2:
        l1 = LineString(vertices)
        centroid = l1.centroid
        x_lat = centroid.x
        x_lon = centroid.y
        radius = 0
        for x, y in l1.coords:
            g = geod.Inverse(x_lat, x_lon, x, y ) 
            dis = g['s12']/1852.344  # 1852.344 meter = 1NM
            if dis > radius:
                radius = dis
        return (round(centroid.x, 6),round(centroid.y, 6), round(radius, 2))

    p1 = Polygon(vertices )
    centroid = p1.centroid
    x_lat = centroid.x
    x_lon = centroid.y
    radius = 0
    for x, y in p1.exterior.coords:
        g = geod.Inverse(x_lat, x_lon, x, y ) 
        dis = g['s12']/1852.344  # 1852.344 meter = 1NM
        if dis > radius:
            radius = dis

    return (round(centroid.x, 6),round(centroid.y, 6), round(radius, 2))


def main():
 
    test = '4809N01610E001'  #14
    coords = get_DDcoords(test)   
    print(f's:{test} => {coords}') 
    test= '4028N10713W025' #15
    coords = get_DDcoords(test)   
    print(f's:{test} => {coords}') 

    test ='402500N 075000W' #16
    coords = get_DDcoords(test)   
    print(f's:{test} => {coords}') 

    test ='411415.07N 0083700.29W' #22
    coords = get_DDcoords(test)   
    print(f's:{test} => {coords}') 

    test = '37053691N 074272201W' #20
    coords = get_DDcoords(test)   
    print(f's:{test} => {coords}') 

    # TODO test calculate_centroid_and_radius

if __name__ == "__main__":
    main()
