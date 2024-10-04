function config = load_config_variables()
    config.plot_figure = true;

    %% Note

    %Custom tag Matlab parameters:
    %frequency = 148.700e6 MHz
    %pulse_dur = 0.08;
    %thrshld = 0.5; %For custom tag
    %pulse_sunset = 3;

    %Off-the-shelf tag Matlab parameters : 
    %frequency = 150.563e6 MHz
    %pulse_dur = 0.02;
    %thrshld = 5; %for off-the-shelf tag since that has high SNR value
    %pulse_sunset = 6; %More correlation helps pulse detection for high SNR

    %% For AOA profile computation

    config.centerfreq = 148.700e6; %DRA module
%     config.centerfreq = 150.563e6; %Off-the-shelf-fishtag
    config.c = 3e8;
    config.subCarriersNum = 1;
    config.lambda = config.c/config.centerfreq;       
    config.phi_min = -180*(pi/180);
    config.phi_max = 180*(pi/180);
    config.nphi = 360;
    config.theta_min = 0*(pi/180);
    config.theta_max = 90*(pi/180);
    config.ntheta = 90;
    config.sub_arrays = 3;
    config.antenna_separation = 1.10; % in meters

    config.save_filtered_matched_data_in_file=false;
    
    %% For signal filtering
    config.div_val = 7.5; %FOR VHF outdoors at Horn pond
    config.sampling_rate = 71428; % Used in the SDR configuration
    config.pulse_rep = 1.07;
    config.duty_cycle_buffer = 1; 
    config.pulse_dur = 0.08; %For custom tag
    config.thrshld = 0.5; %For custom tag
    config.pulse_sunset = 3; %For custom tag
%     config.pulse_dur = 0.02; %For off-the-shelf tag
%     config.thrshld = 5; %for off-the-shelf tag
%     config.pulse_sunset = 6; %For off-the-shelf tag
    
    %If the iq data is not downsampled on the SDR side, the use n-th point
    %to subsample locally, else no need to use n_th points
    
    config.iq_down_sample_factor = 32; %NJ-this is set in the RPi python script to minimize size of data transfer.
%     config.iq_down_sample_factor = 1; %NJ-this is set in the RPi python script to minimize size of data transfer.
    config.downsample_h_list = false;
%     config.downsample_h_list = false; %NJ- means that the data is already downsampled so use the iq_downsample_factor = 32
    config.every_nth_point = 32;


    %% For comparison with in-place rotation data
    config.in_place_rotation = false;   
    

end