from exif import Image
import os
import csv
import PIL.Image
from os import listdir
from os.path import isfile, join
import numpy as np
import cv2
from matplotlib import pyplot as plt
from exif import Image
from geopy import Point, distance
from geopy.distance import geodesic
from src.configs.constants import convert_longlat_to_xy_in_meters

class PhotoIDGT:
    def __init__(self,dir_photoID, op_dir, start_time, end_time):
        self.dir_photoID = dir_photoID
        self.op_dir = op_dir
        self.start_time = start_time
        self.end_time = end_time
        self.output = []
        self.skip_field=["maker_note", "components_configuration", "flashpix_version", "_interoperability_ifd_Pointer"]
        
        #This is manually obtained
        self.bbox = [800,300]
        self.fluke_width = 1500 #mm
        self.fluke_height = 1000 #mm
        self.sensor_width = 22.5 #Canon EOS 70D
        self.sensor_height = 15.0 #Canon EOS 70D
        self.debug = 0
        
    def get_surfacings(self):
        w_coords = ([],[])
        w_xys = ([], [])
        c_coords = ([],[])
        c_xys = ([], [])
        angles = []
        for fn in os.listdir(self.dir_photoID):
            f = os.path.join(dir, fn)
            
            if os.path.isfile(f) and 'jpg' in f:
                fname = f.split('/')[-1]
                print(f)
                
                self.bbox = self.find_bbox(f)
                
                with open(f, 'rb') as image_file:
                    my_image = Image(image_file)                    
                    datetime = my_image.datetime 
                    time = datetime.split(" ")[1]
                    print(time)
                    focal_length = my_image.focal_length * 1.6 #300
                    pixel_width = 1 / my_image.focal_plane_x_resolution
                    pixel_height = 1 / my_image.focal_plane_y_resolution 
                    pixel_dim = (pixel_width+pixel_height)/2
                    # print(bbox[0]*pixel_dim*25.4, bbox[1]*pixel_dim*25.4)
                    
                    hori_distance = (focal_length*self.fluke_width*my_image.pixel_x_dimension)/(self.bbox[0]*self.sensor_width) / 1000
                    vert_distance = (focal_length*self.fluke_height*my_image.pixel_y_dimension)/(self.bbox[1]*self.sensor_height) / 1000
                    
                    lat = my_image.gps_latitude[0] + my_image.gps_latitude[1] /60
                    lon = - my_image.gps_longitude[0] - my_image.gps_longitude[1] /60
                    print(my_image.gps_latitude, my_image.gps_longitude, lat, lon)
                    p2_lat_p2_long = geodesic(meters=hori_distance).destination((lat, lon), my_image.gps_img_direction).format_decimal()  
                    p2_lat,p2_long = map(float, p2_lat_p2_long.split(','))

                    print([fname, time, hori_distance, vert_distance, my_image.gps_img_direction, p2_lat, p2_long])
                    self.output.append([fname, time, hori_distance, vert_distance, my_image.gps_img_direction])

                    w_coords[0].append(p2_long)
                    w_coords[1].append(p2_lat)

                    c_coords[0].append(lon)
                    c_coords[1].append(lat)

                    xy = convert_longlat_to_xy_in_meters(p2_long, p2_lat)
                    w_xys[0].append(xy[0])
                    w_xys[1].append(xy[1])

                    xy = convert_longlat_to_xy_in_meters(lon, lat)
                    c_xys[0].append(xy[0])
                    c_xys[1].append(xy[1])

                    angles.append(my_image.gps_img_direction)
            print("----------------------")

        print(self.output)
        plt.cla()
        for i in range(len(w_xys[0])):
            plt.text(w_xys[0][i], w_xys[1][i] ,s = str(round(w_coords[0][i],2)) +','+ str(round(w_coords[1][i],2)), rotation = 45)
            plt.text(c_xys[0][i], c_xys[1][i] ,s = str(round(c_coords[0][i],2)) +','+ str(round(c_coords[1][i],2)) + ','+ str(round(angles[i])) + ',' +str(i), rotation = 45)
        plt.scatter(w_xys[0], w_xys[1], label = 'whale ground truth estimates')
        plt.scatter(c_xys[0], c_xys[1], label = 'Camera GPS')
        plt.legend()
        plt.show()

    def find_bbox(self,fn):
        nemo = cv2.imread(fn) # img
        hsv_nemo = cv2.cvtColor(nemo, cv2.COLOR_RGB2HSV)
        light_gray = (0, 0, 0)
        dark_gray = (100,100,100)
        mask = cv2.inRange(nemo, light_gray, dark_gray)
        result = cv2.bitwise_and(nemo, nemo, mask=mask)
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                
        op = self.draw(contours, mask)
        im2 = op[0]
        w = op[1]
        h = op[2]
        
        if not self.debug:
            plt.imshow(mask, cmap="gray")
            fname = fn.split('/')[-1]
            plt.savefig(self.op_dir+fname)
        else:
            plt.subplot(2, 2, 1)
            plt.imshow(hsv_nemo, cmap="gray")
            plt.subplot(2, 2, 2)
            plt.imshow(mask, cmap="gray")
            plt.subplot(2, 2, 3)
            plt.imshow(result)
            plt.subplot(2, 2, 4)
            plt.imshow(im2)
            plt.show()
        return [w,h]


    def draw(self,contours,img):
        # contours, hierarchy = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print("Number of Contours = " ,len(contours))
        contours = sorted(contours, key = cv2.contourArea, reverse = True)
        img = cv2.drawContours(img, [contours[0]], -1, (0, 255, 0), 3)
        (x,y,w,h) = cv2.boundingRect(contours[0])
        img= cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 20)
        
        return [img,w,h]


if __name__ == "__main__":
    dir = "Science_Rebuttal/photo-id/only_flukes/"
    op_dir = "Science_Rebuttal/photo-id/only_flukes/out/"
    start_time = "11:00"
    end_time = "3:30"
    obj = PhotoIDGT(dir,op_dir,start_time,end_time)
    obj.get_surfacings()