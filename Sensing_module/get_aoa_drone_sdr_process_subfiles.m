function [final_csi_rx1_filtered, final_csi_rx2_filtered, final_h_list, final_yawList, rho, varphiList, varthetaList, final_position_x , final_position_y, final_position_z] = get_aoa_drone_sdr_process_subfiles(fn_csi, ...
                                                                                                            fn_displacement_data, ...
                                                                                                            plot_figure, ...
                                                                                                            div_val, ...
                                                                                                            dowmsample_h_list, ...
                                                                                                            every_nth_point, ...
                                                                                                            iq_down_sample_factor, ...
                                                                                                            sampling_rate, ...
                                                                                                            pulse_rep, ...
                                                                                                            pulse_dur, ...
                                                                                                            duty_cycle_buffer, ...
                                                                                                            thrshld, ...
                                                                                                            pulse_sunset)

    final_h_list = [];
    final_yawList = [];
    rho = [];
    varphiList = [];
    varthetaList = [];
    h_list_gps = [];
    h_list_imu = [];
    gps_yawList = [];
    imu_yawList = [];
    odom_gps = true;

    %% Process IQ Data files to get signal phase without CFO
    fprintf("Reading VHF Data...\n");
    a=datetime;
    data_temp = load(fn_csi);
    data = data_temp.rx_csi;
    b=datetime;

    a=datevec(a);
    b=datevec(b);
    fprintf("Time to readfile = %f seconds\n",etime(b,a));
    
    %Discard inital datapoints as they get corrupted when SDR collection starts. 
    iq_times_temp = data(80000:end,1);
    rx1_csi = data(80000:end,2);
    rx2_csi = data(80000:end,3);        

    size_val = size(rx1_csi);
    clear data_temp;
   
    % CSI data for RX1
    if plot_figure
        figure(101);
        x = linspace(1,size_val(:,1),size_val(:,1));
        scatter(x,abs(rx1_csi),'x');
        title('Signal magnitude from IQ Data RX1 - raw');
        xlabel('Samples');
        ylabel('abs(I+jQ)');
    end
     
    % CSI data for RX2
    if plot_figure
        figure(102);
        x = linspace(1,size_val(:,1),size_val(:,1));
        scatter(x,abs(rx2_csi),'x');
        title('Signal magnitude from IQ Data RX2 - raw');
        xlabel('Samples');
        ylabel('abs(I+jQ)');
    end
    
    % Get Complex conjugate relative channel
    h_ref_temp = rx2_csi.*conj(rx1_csi);  %ant2->ant1 (boston - Oct
    
    figure(1);
    size_val_href = size(h_ref_temp);
    x_href = linspace(1,size_val_href(:,1),size_val_href(:,1));
    scatter(x_href,abs(h_ref_temp),'x');
    set(gca, 'FontSize', 16);
    title('IQ Data Test href raw');
    xlabel('Samples');
    ylabel('Signal strength');
    
    warning('off','MATLAB:colon:nonIntegerIndex');
    clear data;
    clear rx1_a;
    clear rx1_b;
    clear rx1_mag;
%     clear rx1_csi;
    clear rx2_a;
    clear rx2_b;
    clear rx2_mag;
%     clear rx2_csi;


    %% UAV-RT pulsefind (partial code)
    Fs = sampling_rate/iq_down_sample_factor; %71428 is the sampling rate set for the SDR, but when saving the data, its is downsampled.
    data_in_2 = h_ref_temp';
    t = 1/Fs * (0:1:length(data_in_2)-1)+1/Fs; %First point is at 1/Fs
    data_abs = abs(data_in_2);    
    tick = 1;
    loop_end = 1;
    duty_cycle = pulse_dur/pulse_rep*100 + duty_cycle_buffer;

    %Run this loop twice
    for loop=1:loop_end
        % CREATE A MOVING MEAN OF THE DATA TO USE AS A BASELINE FOR THE THRESHOLD DETECTOR
        %use moving mean over 1 pulse rep rate
        n_pulse_rep = round(pulse_rep*Fs);%this is the number of samples at the windowed rate that should contain a single pulse based on the rep rate of the pulses
        n_pulse_dur = round(pulse_dur*Fs);
        movemean_data = movmean(data_abs,n_pulse_rep-5*n_pulse_dur);
        
        figure(103);
        plot(x,data_abs,'x'); hold on;
        plot(x,movemean_data,'x'); hold on;
        
        % THRESHOLD TO DETECT PULSES OVER 1.5X THE MOVING MEAN
        data_abs_thresh = data_abs;
        data_abs_thresh(data_abs<(1+thrshld)*movemean_data)=0;
        plot(x,data_abs_thresh,'--'); 
        hold off;
        
        % SLIDING WINDOW CORRELATOR TO REJECT FALSE POSITIVES IN THE TIME BETWEEN PULSES
        n_window = round(Fs*pulse_rep*pulse_sunset);    %look back for pulse_sunset pulse periods
        n_pulse_rep = round(Fs*pulse_rep);              %width of a single pulse period
        t_window = pulse_rep*pulse_sunset;                %time of the window we are considering (time of the pulse_sunset window)
        
        for i = n_window+1:n_pulse_rep:length(data_abs_thresh)  %we jump forward by n_pulse_rep each time and then look back n_window
            data_abs_thresh_wind = data_abs_thresh(i-n_window:i); %this is the block of data we want to consider
            %This next line creates a square wave with 0-1 amplitude that has a
            %period of the pulse_rep and a duty cycle of 10% so that we can be off
            %by +/-5% of the actual pulse rep rate and still catch the pulses in
            %the sliding correlator. 
            y = 1/2*(1+square(2*pi*(1/pulse_rep)*t(i-n_window:i),duty_cycle));%+/- 5% on the pulse
            [xcorvals, thelags]=xcorr(data_abs_thresh_wind,y);  %Do the sliding window correlation
            shift_inds = thelags(find(xcorvals==max(xcorvals),1,'first'));  %Find the index shifting that has the highest correlation
            y_shift = circshift(y,shift_inds);  %Now shift the square wave by the shift index that maximized correlation
            data_abs_thresh_xcor(i-n_window:i) = data_abs_thresh_wind.*y_shift;   %Multiply the data and the shifted square wave to silence spurrious signals between actual pulses
            y_shift_log(i-n_window:i) = y_shift; %Take the current processed block and add it to the record
            tick = tick+1;
        end
        
        if length(data_abs_thresh_xcor)<length(t) %in most cases the previous block will have a few less points than the original vector, so we just set the last few of the xcor vector equal to the original
            data_abs_thresh_xcor(end+1:length(t)) =  data_abs_thresh(length(data_abs_thresh_xcor)+1:length(t));
            y_shift_log(end+1:length(t)) =  zeros(1,length(t)-length(y_shift_log));
        end

        figure(1041);
        plot(x,data_abs_thresh_xcor,'x'); 
%         plot(t,y_shift_log*0.00005,':k'); hold off;
        title('IQ Data Test href filtered using UAV-RT pulsefind code');
        xlabel('Samples');
        ylabel('abs(I+jQ)');

        data_abs = data_abs_thresh_xcor;
    end
 

    clean_href_with_square_wave = data_abs_thresh_xcor';
    I = find(clean_href_with_square_wave);
    filtered_ts = [];
    filtered_href = [];
    filtered_rx1_csi = []; %Saving for further analysis
    filtered_rx2_csi = []; %Saving for further analysis

    for kk=1:length(I)
        filtered_ts = [filtered_ts;iq_times_temp(I(kk),1)];
        filtered_href = [filtered_href;h_ref_temp(I(kk),1)];
        filtered_rx1_csi = [filtered_rx1_csi;rx1_csi(I(kk),1)];
        filtered_rx2_csi = [filtered_rx2_csi;rx2_csi(I(kk),1)];
    end
    
%     filtered_href_ts = [filtered_ts, filtered_href];
    filtered_href_ts = [filtered_ts, filtered_href, filtered_rx1_csi, filtered_rx2_csi];
    clear rx1_csi;
    clear rx2_csi;

    if plot_figure
        figure(104);
        size_val_href = length(filtered_href_ts);
        x_href = linspace(1,size_val_href,size_val_href);
        scatter(x_href,abs(filtered_href_ts(:,2)),'x');
        title('IQ Data Test href filtered');
        xlabel('Samples');
        ylabel('abs(I+jQ)');   
    end
  
    %% Removing noise peaks
    h_ref_max = max(abs(filtered_href_ts(:,2)));
    idx = abs(abs(filtered_href_ts(:,2)))>h_ref_max/div_val;
    h_ref_filtered_final = filtered_href_ts(idx,:);
    
    figure(2);
    size_val_href = length(h_ref_filtered_final);
    x_href = linspace(1,size_val_href,size_val_href);
    scatter(x_href,abs(h_ref_filtered_final(:,2)),'x');
    title('IQ Data Test href Final');
    xlabel('Samples');
    ylabel('abs(I+jQ)');
    
    figure(3);
    scatter(x_href,angle(h_ref_filtered_final(:,2)),'x');
    title('Relative Phase without CFO')
    xlabel('Sample');
    ylabel('Phase (radians)')
    ylim([-4 4]);  

    iq_times = h_ref_filtered_final(:,1);
    h_list = h_ref_filtered_final(:,2);
    
    %Saving for further analysis
    csi_rx1_filtered = h_ref_filtered_final(:,3);
    csi_rx2_filtered = h_ref_filtered_final(:,4);

    %Further downsample since there are many redundant points
    if(dowmsample_h_list)
        iq_times = downsample(iq_times,every_nth_point);
        h_list = downsample(h_list,every_nth_point);
        csi_rx1_filtered = downsample(csi_rx1_filtered,every_nth_point);
        csi_rx2_filtered = downsample(csi_rx2_filtered,every_nth_point);
    end

    % Clear variables to free memory when using large VHF files
    clear csi_times_temp
    clear h_ref_temp
    clear x
    clear x_href
    clear temp_href
    clear h_ref

    position_data = csvread(fn_displacement_data);
    position_times = position_data(:,1);
    constant=position_times(1,1);
    position_times = position_times - constant;
    iq_times_for_position = iq_times - constant;
    imu_angles_orig= position_data(:,3);
    angles_yaw = unwrap(imu_angles_orig); %Same as above - this will eliminate any discontinuities in the angle due to the -pi2pi notation
    angles_deg = angles_yaw*180/pi;

    % Find the first moving angles and filter data using it
    firsttheta = angles_deg(1,1);
    lasttheta = angles_deg(end,1);
    fisrtMovingIndex = find(abs(angles_deg(:,1)-firsttheta) > 2,1); %Changed from 1 to 2 degree when using with drone 
    lastMovingIndex = find(abs(angles_deg(:,1)-lasttheta)>1,1, 'last');

    angles_yaw = angles_yaw(fisrtMovingIndex:lastMovingIndex,:);
    position_times = position_times(fisrtMovingIndex:lastMovingIndex,:);
    position_x = position_data(fisrtMovingIndex:lastMovingIndex,4);
    position_y = position_data(fisrtMovingIndex:lastMovingIndex,5);
    position_z = position_data(fisrtMovingIndex:lastMovingIndex,6);

    %Interpolate
    [~,~,startIndex_csi,endIndex_csi,~] = returnClosestIndices(position_times,iq_times_for_position,0); 
    
    if isempty(startIndex_csi ) && isempty(endIndex_csi)
        final_h_list = [];
        final_yawList = [];
        rho = [];
        varphiList = [];
        varthetaList = [];
    else
        final_h_list = h_list(startIndex_csi:endIndex_csi-1,:);
        final_csi_rx1_filtered = csi_rx1_filtered(startIndex_csi:endIndex_csi-1,:);
        final_csi_rx2_filtered = csi_rx2_filtered(startIndex_csi:endIndex_csi-1,:);
        iq_times_for_position=iq_times_for_position(startIndex_csi:endIndex_csi-1,:); %discard unused CSI values
        temp_yawList =  interp1(position_times,angles_yaw(:,1),iq_times_for_position);
        temp_position_x =  interp1(position_times,position_x(:,1),iq_times_for_position);
        temp_position_y =  interp1(position_times,position_y(:,1),iq_times_for_position);
        temp_position_z =  interp1(position_times,position_z(:,1),iq_times_for_position);
       
        if(plot_figure)
            figure(108);
            scatter(cos(temp_yawList(:,1)),sin(temp_yawList(:,1)),'x');
            title('Interpolated Orientation');
            axis([-1 1 -1 1]);
            xlabel('unit x-axis')
            ylabel('unit y-axis')  
        end

        %remove any Nan values
        [row, ~] = find(isnan(temp_yawList));
        temp_yawList(row,:) = [];
        temp_position_x(row,:) = [];
        temp_position_y(row,:) = [];
        temp_position_z(row,:) = [];
        final_h_list(row,:) = [];
        
        final_yawList   = wrapToPi(temp_yawList);
        final_position_x = temp_position_x;
        final_position_y = temp_position_y;
        final_position_z = temp_position_z;
    end
    
end