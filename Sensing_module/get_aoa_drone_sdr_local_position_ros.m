function get_aoa_drone_sdr_local_position_ros(dir_iq, dir_ori_mocap, dir_gps)

%% Declare all variables
    %Location 3: Got using GPS RTK module , Dummy values, ignore.
    tx_latitude = 42.3658106;
    tx_longitude = -71.1246243;

    array_radius = 0 ; %For orbit mode, this is calculated using the position data
    antenna_offset = 0;

    config = load_config_variables();

    %% Declare all variables
    in_place_rotation = config.in_place_rotation;   
    plot_figure = config.plot_figure;
    sub_arrays = config.sub_arrays;
    antenna_separation = config.antenna_separation;
    lambda = config.lambda; 
    phi_min = config.phi_min;
    phi_max = config.phi_max;
    nphi = config.nphi;
    theta_min = config.theta_min;
    theta_max = config.theta_max;
    ntheta = config.ntheta;
    phiList = linspace(phi_min, phi_max, nphi).';
    thetaList = linspace(theta_min, theta_max, ntheta);
    phiRep = repmat(phiList, [1, ntheta]);
    thetaRep = repmat(thetaList, [nphi, 1]); 
    lambdaRep = repmat(reshape(lambda, 1, 1, []), [nphi, ntheta, 1]);    

    %Generate multiple sub-antenna arrays by subsampling data
    intermediate_aoaProfile = [];

    %% ROS Configuration
    rosinit;
    N = ros.Node('AOA_node');
    
    pubRover = ros.Publisher(N,"/got_data_rover","std_msgs/Bool");
    pub_aoa_status = ros.Publisher(N,"/got_aoa","std_msgs/Bool");
    msg = rosmessage(pubRover);
    msg.Data = false;
    msg_aoa = rosmessage(pub_aoa_status);
    msg_aoa.Data = true;

    send(pubRover, msg) %Initialize
    subRover = ros.Subscriber(N,"/got_data_rover","std_msgs/Bool");
    fprintf("Initializations done\n");

    %% Start processing the subfiles
    while true
        got_data_rover = subRover.LatestMessage;

        if(got_data_rover.Data)
            fprintf("Got data from Drone\n");
            send(pubRover, msg) %this will reset got_data_rover to false in the next loop

            iq_file_list = dir(dir_iq);
            orientation_using_imu_file_list = dir(dir_ori_mocap);
            gps_file_list = dir(dir_gps);
            
            h_list_all = [];
            yawList_all = [];
            true_azimuth_all = [];
        
            for ii=3:length(iq_file_list)
                fn_iq_data = fullfile(dir_iq, iq_file_list(ii).name);
                fn_disp_drone = fullfile(dir_ori_mocap, orientation_using_imu_file_list(ii).name);
                fn_gps_drone = fullfile(dir_gps, gps_file_list(ii).name);
                
%               %Backup iq and ori data files
                dt = datetime('now','TimeZone','local','Format','d-MMM-y HH:mm:ss');
                raw_ts = string(dt);
                temp_ts = split(raw_ts);
                ts=append("_",temp_ts(1,:),"_",temp_ts(2,:));
                temp_fn = split(iq_file_list(ii).name,'.');
            
                dest = append("2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/all_iq_data/",temp_fn(1,:),ts,".mat");
                copyfile(fn_iq_data,dest);

                temp_fn = split(orientation_using_imu_file_list(ii).name,'.');    
                dest = append("2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/all_ori_data/",temp_fn(1,:),ts,".csv");
                copyfile(fn_disp_drone,dest);

                temp_fn = split(gps_file_list(ii).name,'.');
                dest = append("2.5D_new_formulation_experiments/vhf_outdoor_drone_experiments/all_gps_data/",temp_fn(1,:),ts,".csv");
                copyfile(fn_gps_drone,dest);

                true_azimuth = get_true_AOA_using_GPS(fn_gps_drone, tx_latitude, tx_longitude);
%                 
                if in_place_rotation
                    array_radius = antenna_separation;
                else
                    array_radius = get_circular_trajectory_radius(fn_disp_drone) - (antenna_separation/2); %Since the antenna axis is centered on the drone
                end
                fprintf("Estimated array Radius: %f \n", array_radius);
                
                tau = array_radius / antenna_separation;

                [temp_h_list, temp_yawList, ~, ~, ~] = get_aoa_drone_sdr_process_subfiles(fn_iq_data, ...
                                                                                         fn_disp_drone, ...
                                                                                         config.plot_figure, ...
                                                                                         config.div_val,...
                                                                                         config.downsample_h_list, ...
                                                                                         config.every_nth_point, ...
                                                                                         config.iq_down_sample_factor, ...
                                                                                         config.sampling_rate, ...
                                                                                         config.pulse_rep, ...
                                                                                         config.pulse_dur, ...
                                                                                         config.duty_cycle_buffer, ...
                                                                                         config.thrshld);
                yawList_all = [yawList_all;temp_yawList];
                
                if isempty(temp_h_list)
                    h_list_all = [];
                    break;
                else
                    h_list_all = [h_list_all;temp_h_list];
                    true_azimuth_all = [true_azimuth_all;true_azimuth];
%                     true_elevation_all = [true_elevation_all;true_elevation*180/pi];
                end
            end
         
            if isempty(h_list_all)
                fprintf("Bad data. Waiting for new one\n");
            else
                %% ---------------------------------- Compute AOA IMU (i.e MOCAP Orientation)----------------------------------------
                % imu_yawList_all = imu_yawList_all - imu_yawList_all(1); 
                figure(8);
                scatter(cos(yawList_all(:,1)),sin(yawList_all(:,1)),'x');
                title('Interpolated Orientation Combined - IMU')
                set(gca, 'FontSize', 12);
                xlabel('unit X-axis')
                ylabel('unit Y-axis')  
                
                
                for sub_array_val=1:sub_arrays 
                    h_list_curr = downsample(h_list_all,sub_arrays,sub_array_val-1);
                    yawList_curr = downsample(yawList_all,sub_arrays,sub_array_val-1);
                        
                    len = length(yawList_curr);           
                    phiRep2 = repmat(phiRep, [1, 1, 1, len]);
                    yawList2 = repmat(reshape(yawList_curr, [1 , 1, 1, len]), [nphi ntheta 1, 1]);
                    lambdaRep2 = repmat(lambdaRep, [1, 1, 1, len]);
                    thetaRep2 = repmat(thetaRep, [1, 1, 1, len]);
                    
                    % Steering vector for new 2 antenna configuration.
                    e_term = exp(1i*2*pi*array_radius*(cos(phiRep2 - yawList2).*sin(thetaRep2))./lambdaRep2);
                    leftMat= double(squeeze(e_term(:, :, 1, :))); % 360 * 90 * 135
                    rightMat = h_list_curr; % 135 * 134 
                    resultMat = double(prodND_fast(leftMat, rightMat));
                    aoaProfile = permute((double(abs(resultMat)).^2), [3, 1, 2]);        
                    
    
                    if(plot_figure)
                        figure(60+sub_array_val);
                        subplot(2,1,1)
                        surf(phiList*180/pi, thetaList*180/pi, aoaProfile.', 'EdgeColor', 'none');
                        set(gcf,'Renderer','Zbuffer')            
                        xlabel('Azimuth (Degree)');
                        ylabel('Elevation (Degree)');
                        title(sprintf('Combined AOA profile (side view)'));         
                        subplot(2,1,2)
                        surf(phiList*180/pi, thetaList*180/pi, aoaProfile.', 'EdgeColor', 'none');
                        set(gcf,'Renderer','Zbuffer');
                        view(2)
                        title('Combined AOA profile Top View');
                        xlabel('Azimuth (degree)');
                        ylabel('Elevation (degree)');
                    end
            
                    if sub_array_val == 1
                        intermediate_aoaProfile = aoaProfile;
                    else
                        intermediate_aoaProfile = intermediate_aoaProfile.*aoaProfile;
                    end
            
                    % Get top angle i.e. AOA 
                    [val, idx] = max(aoaProfile(:));
                    [idx_row, idx_col] = ind2sub(size(aoaProfile),idx);
                    azimuth_aoa = phiList(idx_row)*180/pi;
                    elevation_aoa = thetaList(idx_col)*180/pi;
                    fprintf("AOA  for Drone using sub-array profile: Azimuth = %f , Elevation = %f\n", azimuth_aoa,elevation_aoa);
                end
            
                % Get top angle i.e. AOA 
                [val, idx] = max(intermediate_aoaProfile(:));
                [idx_row, idx_col] = ind2sub(size(intermediate_aoaProfile),idx);
                azimuth_aoa = phiList(idx_row)*180/pi;
                elevation_aoa = thetaList(idx_col)*180/pi;
                
                fprintf("AOA  for Drone data using profile product: Azimuth = %f , Elevation = %f\n", azimuth_aoa,elevation_aoa);
                acutal_elevation_aoa = estimate_correct_elevation(elevation_aoa,tau);
                fprintf("AOA  for Drone data using profile product: Azimuth = %f , Correct Elevation = %f\n", azimuth_aoa,acutal_elevation_aoa);
                
                figure(9);
                subplot(2,1,1)
                surf(phiList*180/pi, thetaList*180/pi, intermediate_aoaProfile.', 'EdgeColor', 'none');
                set(gcf,'Renderer','Zbuffer') ;
                set(gca, 'FontSize', 12);
                xlabel('Azimuth (Degree)');
                ylabel('Elevation (Degree)');
                title(sprintf('Drone: Final AOA profile (side view)'));         
                subplot(2,1,2)
                h = pcolor(phiList*180/pi, thetaList*180/pi, intermediate_aoaProfile.');
                set(h, 'EdgeColor', 'none');
                view(2)

                hold on;
                ds = ceil(0.025 * length(true_azimuth_all));  %Use only 10% of points to visualize true AOA
                true_azimuth_all_final = downsample(true_azimuth_all,ds);
                elevation_aoa_list = repelem(elevation_aoa,length(true_azimuth_all_final))';
                sz = repelem(100,length(true_azimuth_all_final));
                scatter(true_azimuth_all_final,elevation_aoa_list,sz,'o','MarkerFaceColor', 'red','MarkerEdgeColor', 'red', 'MarkerFaceAlpha',.1, 'MarkerEdgeAlpha',.1);
                title('Drone: Final AOA profile (top view)');
                set(gca, 'FontSize', 12);
                xlabel('Azimuth (degree)');
                ylabel('Elevation (degree)');
                hold off;
                
            if in_place_rotation
                fprintf("True azimuth AOA %f \n", mean(true_azimuth_all(2:length(true_azimuth_all)))); %From center
                fprintf("Error in AOA estimation- using RX circular trjaectory gps center: %f degree \n", mean(true_azimuth_all(2:length(true_azimuth_all))) - azimuth_aoa);
            else
                fprintf("True azimuth AOA (from trajectory center) %f \n", true_azimuth_all(1)); %From center
                fprintf("Error in AOA estimation- using RX circular trjaectory gps center: %f degree \n", calculate_angle_difference(true_azimuth_all(1), azimuth_aoa)); %From center
            end   
                    
            end

            beep;
            pause(1);  
        
            %Start next iteraton. Not doing slidingg windown calculation
            %for now.
            for i=1:50
                send(pub_aoa_status, msg_aoa); %Signal the robots to start next round of data collection
                pause(0.1);
            end
            fprintf("Published for next iteration \n");
        end
    end
end


function angle_difference = calculate_angle_difference(angle1, angle2)
    % Ensure that angles are between 0 and 360 degrees
    angle1 = mod(angle1, 360);
    angle2 = mod(angle2, 360);

    % Calculate the absolute angular difference
    absolute_difference = abs(angle1 - angle2);

    % Take the minimum of the absolute difference and 360 minus the absolute difference
    angle_difference = min(absolute_difference, 360 - absolute_difference);
end


function real_elevation_estimate = estimate_correct_elevation(elevation_aoa_from_profile, tau)
    angle_val = 90;
    step = 0.25;
    diff = 100;
    real_elevation_estimate = 0;
    while angle_val > 0
        elevation_aoa_from_theory = asin(sin(angle_val*pi/180)/tau);
        if abs(elevation_aoa_from_theory - elevation_aoa_from_profile) < diff
            real_elevation_estimate = elevation_aoa_from_theory;
        end
        angle_val = angle_val - step;
    end
    fprintf("Actual elevation angle = %f \n", real_elevation_estimate);
end
