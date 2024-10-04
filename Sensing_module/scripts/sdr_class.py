import SoapySDR
from SoapySDR import * #SOAPY_SDR_ constants
import time
import copy

#bandwidth lowerbound = #200 KHz for bladeRF,
#149.242e6

class SDRDevice:
    def __init__(self, sdr_type, duration, iterations, tag_vhf_frequency, gain = 10, sample_rate=1e6):
        self.data_collection_duration = duration #seconds
        
        if(sdr_type == "none"):
            get_connected_devices()
            print("Use this info to get device name")
            exit(1)
        
        self.iterations = iterations
        self.device_args = dict(driver=sdr_type)  #The driver name can be obtained after enumerating.
        self.sdr = SoapySDR.Device(self.device_args)
        self.__FLAG_AGC_OFF = False
        self.__FLAG_INFO = False
        self.rx_gain = gain
        self.rx_sample_rate = sample_rate
        self.frequency = tag_vhf_frequency
        self.__FLAG_Plot = True

    def get_connected_devices(self):
        results = SoapySDR.Device.enumerate()
        for result in results: print(result)

    def get_device_info(self):
        print("Num Channels = ", self.sdr.getNumChannels(SOAPY_SDR_RX))
        print("Antennas = ", self.sdr.listAntennas(SOAPY_SDR_RX, 0))
        print("Gains = ", self.sdr.listGains(SOAPY_SDR_RX, 0))
        print("Has Automatic gain control? = ", self.sdr.hasGainMode(SOAPY_SDR_RX, 0))
        print("Has hardware clock? = ", self.sdr.hasHardwareTime())
        print("Has IQ Balance? = ", self.sdr.hasIQBalance(SOAPY_SDR_RX, 0))
        
        freqs = self.sdr.getFrequencyRange(SOAPY_SDR_RX, 0)
        for freqRange in freqs: print( "Frequency range {}".format(freqRange))
        
        srate = self.sdr.getSampleRateRange(SOAPY_SDR_RX, 0)
        for srateRange in srate: print( "Sample Rate range {}".format(srateRange))

        val = self.sdr.getStreamFormats(SOAPY_SDR_RX, 0)
        print("getStreamFormats: ", val)


