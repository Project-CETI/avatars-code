import numpy as np
from geopy import Point, distance                                                                                                                                                                    
from geopy.distance import geodesic 
from geographiclib.geodesic import Geodesic
import pyproj, math, datetime
import typing as t
import collections
Boat_Control = collections.namedtuple('Boat_Control', ['b_theta', 'b_v'])
Agent_Control = collections.namedtuple('Agent_Control', ['agent_theta', 'agent_v', 'agent_VHF'])


ObservationClass_whale = collections.namedtuple('ObservationClass_whale', ['current_whale_up', 'current_receiver_error', \
    'receiver_current_loc', 'current_observed_AOA', 'current_observed_xy'])
ObservationClass_whale_TA = collections.namedtuple('ObservationClass_whale_TA', ['current_whale_up', 'current_receiver_error', \
    'receiver_current_loc', 'current_observed_AOA_candidate1', 'current_observed_AOA_candidate2', 'current_observed_xy'])

ObservationClass_whale_v2 = collections.namedtuple('ObservationClass_whale_v2', ['current_whale_up', 'current_receiver_error', \
    'receiver_current_loc', 'current_observed_AOA'])

Loc_xy = collections.namedtuple('Loc_xy', ['x', 'y'])

Radian_to_degree = 180 / np.pi
Deg_to_radian = np.pi / 180
Earth_radius_in_m = 6371 * 1000
Earth_radius_in_km = 6371
Roseau_long = -61.3794
Roseau_lat = 15.3092
Buoys_lat = [15.347333, 15.402083, 15.490722]
Buoys_long = [ -61.490667, -61.564583, -61.55125]
Buoys_lat_long = [(15.347333, -61.490667), (15.402083, -61.564583), (15.490722, -61.55125)]
Min_long = -61.697258999999974
Max_long = -61.25692300000003
Min_lat = 15.104013600000004
Max_lat = 15.548686399999996
bouy1_long = -61.490667
bouy1_lat = 15.347333
bouy2_long = -61.564583
bouy2_lat = 15.402083
bouy3_long = - 61.55125
bouy3_lat = 15.490722
Whale_speed_mtpm =  160
Crs_4326 = pyproj.CRS.from_epsg(4326)
Crs = pyproj.CRS.from_dict({'proj': 'utm', 'zone': 20, 'south': False})
Crs_dom = pyproj.CRS.from_epsg(Crs.to_authority()[1])
Gps_transformer = pyproj.Transformer.from_crs(Crs_4326, Crs_dom)

whale_surface_color = np.array([65/255, 105/255, 225/255])
whale_underwater_color = np.array([0, 0, 0])
whale_assigned_color = np.array([1, 0, 0])

def angle_diff_degree(b, a):
    return np.mod(b - a + 180, 360) - 180

def angle_diff_radian(b, a):
    return np.mod(b - a + np.pi, 2 * np.pi) - np.pi

def get_distance_from_latLon_to_meter(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    distance_in_meter = distance.distance(Point(lat1, lon1), Point(lat2, lon2)).m
    return distance_in_meter


def get_bearing_from_p1_p2(p1_long, p1_lat, p2_long, p2_lat): #long: x, lat: y
    d = Geodesic.WGS84.Inverse(p1_lat, p1_long, p2_lat, p2_long)
    theta = (d['azi1']) * Deg_to_radian
    return theta

def get_gps_from_start_vel_bearing(p1_long, p1_lat, vel, theta_rad): # vel in mtpm

    p2_lat_p2_long = geodesic(meters=vel).destination((p1_lat, p1_long), \
        theta_rad * Radian_to_degree).format_decimal()  
    p2_lat,p2_long = map(float, p2_lat_p2_long.split(','))
    return (p2_long, p2_lat)

def convert_longlat_to_xy_in_meters_v2(long1, lat1):
    x, y = Gps_transformer.transform(long1, lat1)
    return (x, -y)

def convert_longlat_to_xy_in_meters(long1, lat1, min_lat = None, min_long = None):
    min_lat = (min_lat if min_lat is not None else Min_lat)
    y = get_distance_from_latLon_to_meter(lat1, long1, min_lat, long1)
    if min_lat > lat1:
        y = -y
    min_long = (min_long if min_long is not None else Min_long)
    x = get_distance_from_latLon_to_meter(lat1, long1, lat1, min_long)
    if min_long > long1:
        x = -x
    return (x,y)

def convert_xy_in_meters_to_longlat(x: float, y:float)-> t.Tuple[float, float]:
    bearing = np.arctan2(y, x)
    distance_in_meters = np.sqrt(x * x + y * y)
    distance_in_km = distance_in_meters / 1000
    lat1 = Min_lat * Deg_to_radian  # Current lat point converted to radians
    lon1 = Min_long * Deg_to_radian  # Current long point converted to radians
    lat2 = math.asin(math.sin(lat1) * math.cos(distance_in_km / Earth_radius_in_km) )
    lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_in_km / Earth_radius_in_km) * math.cos(lat1), \
        math.cos(distance_in_km / Earth_radius_in_km) - math.sin(lat1) * math.sin(lat2))
    lat2 = math.degrees(lat2)
    lon2 = math.degrees(lon2)
    return (lon2, lat2)


def get_cartesian(lat=None,lon=None):
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)
    x = Earth_radius_in_m * np.cos(lat) * np.cos(lon)
    y = Earth_radius_in_m * np.cos(lat) * np.sin(lon)
    z = Earth_radius_in_m *np.sin(lat)
    return x,y,z

def get_lat_long(x, y, z):
    latitude = np.arcsin(z/Earth_radius_in_m) * 180 / np.pi
    longitude = np.arctan2(y, x) * 180 /np.pi
    return latitude, longitude


def from_datetext_to_text(date_text):
    ymd, hms = date_text.split(' ')
    try:
        y, m, d = list(map(int, ymd.split('-')))
    except Exception as e:
        y, m, d = list(map(int, ymd.split(':')))
    h, mi, s = list(map(int, hms.split(':')))
    date_obj = datetime.datetime(y, m, d, hour=h, minute=mi, second=s) #UTC
    return date_obj