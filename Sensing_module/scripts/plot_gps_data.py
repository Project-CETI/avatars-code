#!/usr/bin/env python3

import numpy as np
import csv
import matplotlib.pyplot as plt
from math import radians, cos, sin, asin, sqrt



def distance(lat1, lat2, lon1, lon2):
     
    # The math module contains a function named
    # radians which converts from degrees to radians.
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
      
    # Haversine formula 
    dlon = lon2 - lon1 
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
 
    c = 2 * asin(sqrt(a)) 
    
    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371
      
    # calculate the result
    return(c * r)



def main():
    r = 6371
    with open('../data/field_oct_10_2023_file_3.csv', 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    data_array = np.array(data).astype(float)
    data_sliced = data_array

    print(data_sliced[0:10,1])
    print(len(data_sliced))
    x = data_sliced[:,1]
    y = data_sliced[:,2]

    dist = distance(float(x[0]), float(x[30]), float(y[0]), float(y[30]))
    print("Distance in meters, ", dist*1000)

    lat, lon = x*3.14/180, y*3.14/180
    x_m = r * np.multiply(np.cos(lat), np.cos(lon)) * 1000 #convert to meters
    y_m = r * np.multiply(np.cos(lat) , np.sin(lon)) * 1000
    z_m = r *np.sin(lat) * 1000

    x_final = x_m[0]-x_m
    y_final = y_m[0]-y_m
    z_final = z_m[0]-z_m

    fig, ax = plt.subplots()
    ax.scatter(x_final, y_final)
    ax.grid(True)

    plt.show()

if __name__ == "__main__":
    main()
