#!/usr/bin/env python3

import asyncio
from mavsdk import System
import time
import signal
import numpy as np

class WhaleEmulatorTracker():
    def __init__(self):
        self.gps_timestamped_information = []


    async def run(self):
        # Init the drone
        drone = System()
        print("Waiting to connect to the drone")
        #await drone.connect(system_address="serial:///dev/ttyUSB0:57600")  # Replace with your drone's address
        await drone.connect(system_address="udp://:14540") 
        # await drone.connect(system_address="://:14540") 
        print("Connected to the drone. Fetching data")

        # Start the tasks
        #asyncio.ensure_future(self.print_gps_info(drone))
        asyncio.ensure_future(self.print_position(drone))


        while True:
            await asyncio.sleep(1)

    async def print_gps_info(self,drone):
        async for gps_info in drone.telemetry.gps_info():
            print(f"GPS info: {gps_info}")
            timestamp = time.time_ns()
            #self.gps_timestamped_information.append([timestamp, gps_info.num_satellites])
            #print("Latest data : ", self.gps_timestamped_information[-1])
            signal.signal(signal.SIGINT, self.handler)

    async def print_position(self,drone):
        print("in print position function")
        async for position in drone.telemetry.position():
            timestamp = time.time_ns()
            self.gps_timestamped_information.append([timestamp, position.latitude_deg, position.longitude_deg, position.absolute_altitude_m, position.relative_altitude_m])
            print("Latest data : ", self.gps_timestamped_information[-1])
            signal.signal(signal.SIGINT, self.handler)


    def handler(self, signum, frame):
        print("================== Ctrl-c was pressed =================")
        print("Saving file")
        print(self.gps_timestamped_information)
        output = np.asarray(self.gps_timestamped_information)
        current_unix_timestamp = int(time.time())
        original_string = "dummy_whale_gps_data_"
        string_with_unix_timestamp = original_string + str(current_unix_timestamp) +"_.csv"
        np.savetxt(string_with_unix_timestamp, output, delimiter=",")
        exit(1)

if __name__ == "__main__":
    # Start the main function
    obj = WhaleEmulatorTracker()
    asyncio.run(obj.run())
