import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import numpy #use numpy for buffers
import time
import copy
import matplotlib.pyplot as plt
import argparse
from sdr_class import SDRDevice
import scipy.io
import datetime

class UHDB2100Device(SDRDevice):
    def __init__(self, sdr_type, duration, iterations, tag_vhf_frequency, gain = 40, sample_rate=0.71428e5):
        super().__init__(sdr_type, duration, iterations, tag_vhf_frequency, gain, sample_rate)
        self.bandwidth = 2e5
        self.buffer_size = 256
        self.__FLAG_AGC_OFF = True
        self.__FLAG_INFO = False
        self.rx_gain = gain
        self.__FLAG_Plot = False
        self.__FLAG_SAVE_File = True


    def collect_data(self):
        #Change Gain
        print("Gain before = ", self.sdr.getGain(SOAPY_SDR_RX, 0))
        
        if(self.__FLAG_AGC_OFF):
            print("Disbling automatic gain control and seeting gain manually")
            self.sdr.setGainMode(SOAPY_SDR_RX, 0, False)
            self.sdr.setGain(SOAPY_SDR_RX, 0, self.rx_gain)
            self.sdr.setGainMode(SOAPY_SDR_RX, 1, False)
            self.sdr.setGain(SOAPY_SDR_RX, 1, self.rx_gain)

        # #apply settings
        self.sdr.setSampleRate(SOAPY_SDR_RX, 0, self.rx_sample_rate)
        self.sdr.setSampleRate(SOAPY_SDR_RX, 1, self.rx_sample_rate)
        # sdr.setSampleRate(SOAPY_SDR_RX, 0, 3e5) #For RTL SDR
        self.sdr.setFrequency(SOAPY_SDR_RX, 0, self.frequency)
        self.sdr.setFrequency(SOAPY_SDR_RX, 1, self.frequency)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 0, self.bandwidth)
        self.sdr.setBandwidth(SOAPY_SDR_RX, 1, self.bandwidth)
        
        #setup a stream (complex floats)
        rxStream = self.sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32, [0,1])
        
        if(self.__FLAG_INFO):
            print("Direct access buffers = ", self.sdr.getNumDirectAccessBuffers(rxStream))
            stream_mtu_val = self.sdr.getStreamMTU(rxStream)
            print("Stream MTU : ", stream_mtu_val)


        for runs in range(self.iterations):
            self.sdr.activateStream(rxStream) #start streaming
            final_output0 = []
            final_output1 = []
            final_output=[]
            epoch_time_ns = []
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('VHF Signal pings')

            #receive some samples
            print("Starting logging for Iteration: ", runs)
            start_time = time.time()

            # for i in range(10000):
            while True:
                buff0 = numpy.array([0]*self.buffer_size, numpy.complex64)
                buff1 = numpy.array([0]*self.buffer_size, numpy.complex64)
                nano_start_readstream = time.time_ns()
                sr = self.sdr.readStream(rxStream, [buff0, buff1], len(buff0), timeoutUs=1000)
                nano_end_readstream = time.time_ns()
                avg_nano_time = (nano_start_readstream + nano_end_readstream)/2
                ts = numpy.array([avg_nano_time]*self.buffer_size, numpy.double)
                #ts = numpy.array([time.time_ns()]*self.buffer_size, numpy.double)
                final_output0.append(buff0)
                final_output1.append(buff1)
                epoch_time_ns.append(ts)

                dur = time.time() - start_time
                if(dur > self.data_collection_duration):
                    break


            self.sdr.deactivateStream(rxStream) #stop streaming
            
            print("Done data collection. Aggregating data")
            epoch_time_ns_np_array = numpy.asarray(epoch_time_ns).flatten().reshape(len(epoch_time_ns)*len(epoch_time_ns[0]),1)
            final_output0_np_array = numpy.asarray(final_output0).flatten().reshape(len(final_output0)*len(final_output0[0]),1)
            final_output1_np_array = numpy.asarray(final_output1).flatten().reshape(len(final_output1)*len(final_output1[0]),1)
            rx_csi = numpy.concatenate((epoch_time_ns_np_array,final_output0_np_array,final_output1_np_array), axis=1)

            rx_csi_np_array_downsample = rx_csi[0::32] #Downsample to remove redundant data points
            #rx_csi_np_array_downsample = rx_csi
            X_axis_val = [i for i in range(0,len(rx_csi_np_array_downsample),1)]

            if(self.__FLAG_Plot):

                # for kk in range(10):
                    # print(rx_csi_np_array_downsample[kk])

                print("Plotting")
                plt.close("all")
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))
                fig.suptitle('VHF Signal pings')
                ax1.title.set_text('RX 1')
                ax2.title.set_text('RX 2')
                ax1.scatter(X_axis_val, abs(rx_csi_np_array_downsample[:,1]), marker='x')
                ax2.scatter(X_axis_val, abs(rx_csi_np_array_downsample[:,2]), marker='x')
                plt.show(block=False)
                plt.pause(5)

            if(self.__FLAG_SAVE_File):
                scipy.io.savemat('vhf_drone_payload_data.mat', {'rx_csi': rx_csi_np_array_downsample})
                ct = datetime.datetime.now()
                ts = ct.timestamp()
                tsop = 'vhf_drone_payload_data_'+str(ts)+'_.mat'
                scipy.io.savemat(tsop, {'rx_csi': rx_csi})
        
        print("Done data collection. Closing stream and exiting")
        self.sdr.closeStream(rxStream)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments:",
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument("-a", "--archive", action="store_true", help="archive mode")
    # parser.add_argument("-v", "--verbose", action="store_true", help="increase verbosity")
    # parser.add_argument("-B", "--block-size", help="checksum blocksize")
    # parser.add_argument("--ignore-existing", action="store_true", help="skip files that exist")
    # parser.add_argument("--exclude", help="files to exclude")
    parser.add_argument("duration", type=int, help="duration of a single data collection round (seconds)")
    parser.add_argument("rounds", type=int, help="total rounds of data collection")
    parser.add_argument("tag_vhf_frequency", type=float, help="Frequency of fish tracker")
    args = parser.parse_args()
    sdr_type = "uhd"
    obj = UHDB2100Device(sdr_type, args.duration, args.rounds, args.tag_vhf_frequency)
    #obj.get_device_info()
    obj.collect_data()
