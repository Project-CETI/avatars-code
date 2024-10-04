#!/bin/sh

# Use python code
itr=1
duration=40
python_itr=1
vhf_freq_a=149.546e6
vhf_freq_b=149.242e6
vhf_freq_c=150.793e6
vhf_freq_boston_harbor_vertical=150.793e6
island_pelican_case_vertical=150.754e6
island_pelican_case_horizontal=150.713e6
vhf_freq_boston_harbor_horizontal=150.772e6
vhf_freq_boston_harbor_dra_1_vertical=148.700e6
vhf_freq_boston_harbor_dra_2_vertical=147.770e6
horn_pond_fishTracker_3_vertical=150.563e6
fishtracker_1=150.983e6
fishtracker_2=150.524e6


#Run some initializations
sudo timedatectl set-timezone "America/New_York"
set_date_now=$(ssh ceti-pc-1@192.168.1.2 "date | cut -c 5-27")
echo $set_date_now
echo "============ Updating system time ============"
sudo date --set "$set_date_now"
echo "=============================================="
SoapySDRUtil --probe="driver=uhd,type=b200"
echo "***** SDR driver loaded. Ensure that Drone and Payload GPS are not impacted."
echo "***** Sleeping for 1 minute."
#sleep 60 #Time for the UAV To takeoff
echo "***** If GPS modules are working correctly start drone takeoff. Data collection will start in 1 minute."
#sleep 30
echo "Data collection starting in 30 seconds"
sleep 30
echo "===================== Starting Data collection now ==================="

while [ $itr -lt 3 ]
do

	#Collect the data
	#at 1M rate 20M data collected for each antenna => total of 40M
	echo "======= Starting Wireless signal data collection - Iteration: $itr ======="

	echo "Starting logging SDR data"
	#python3	SoapySDRCode/get_sdr_data_uhd.py $duration $python_itr $vhf_freq_c &
	python3 SoapySDRCode/get_sdr_data_uhd.py $duration $python_itr $vhf_freq_boston_harbor_dra_1_vertical &
	#python3 SoapySDRCode/get_sdr_data_uhd.py $duration $python_itr $fishtracker_1 &

	#Temp fix to handle delay between SDR and traj data collection.
	echo "Sleeping for 5 seconds"
	sleep 5
	
	echo "Starting logging orientation data"
	rostopic pub --once /ceti_sar/start_data_logging std_msgs/Bool "data: true" &
   
	echo "Waiting for 40 seconds before data collection stops automatically"
	sleep 40

	echo "Waiting for 5 sec before initiating file transferring"		
	sleep 5
	
	echo "Done VHF data collection for first half. Sending data to master computer" #Send data to thinkpad for AOA computation.
	scp ~/mocap_vrpn_displacement.csv ceti-pc-1@192.168.1.2:~/CETI_experiments/2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/ori_data/ &
	scp ~/gps_data.csv ceti-pc-1@192.168.1.2:~/CETI_experiments/2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/gps_data/ &
	scp ~/vhf_drone_payload_data.mat ceti-pc-1@192.168.1.2:~/CETI_experiments/2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/iq_data/
			
	itr=`expr $itr + 1`
	sleep 0.1

	echo "Copied all files to master computer. Starting AOA computation"
        rostopic pub --once /got_data_rover std_msgs/Bool "data: true" 

	#delete old files
	rm -rf ~/vhf_drone_payload_data.mat
	rm -rf ~/mocap_vrpn_displacement.csv
	rm -rf ~/gps_data.csv

	echo "Waiting for AOA computation to complete"
        #busy wait till aoa computed. Note this command does not read the msg, but due to error will execute the last echo and unblock.
        rostopic echo /got_aoa | { read message; if [ -z "$message" ]; then echo "NULL"; else exit; fi; } || echo "got aoa"

done

echo " =========== Completed this set of experiments ================"
