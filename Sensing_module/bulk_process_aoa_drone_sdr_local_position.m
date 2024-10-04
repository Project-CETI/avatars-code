function bulk_process_aoa_drone_sdr_local_position(dir_iq, dir_ori_mocap, dir_gps, dir_gps_tx)
    
    config = load_config_variables();

    %% Declare all variables
    in_place_rotation = config.in_place_rotation;   
    plot_figure = config.plot_figure;
    sub_arrays = config.sub_arrays;
    antenna_separation = config.antenna_separation;
    subCarriersNum = config.subCarriersNum;
    lambda = config.lambda; 
    phi_min = config.phi_min;
    phi_max = config.phi_max;
    nphi = config.nphi;
    theta_min = config.theta_min;
    theta_max = config.theta_max;
    ntheta = config.ntheta;

    intermediate_aoaProfile = [];
    phiList = linspace(phi_min, phi_max, nphi).';
    thetaList = linspace(theta_min, theta_max, ntheta);
    phiRep = repmat(phiList, [1, ntheta]);
    thetaRep = repmat(thetaList, [nphi, 1]); 
    lambdaRep = repmat(reshape(lambda, 1, 1, []), [nphi, ntheta, 1]);
    save_filtered_matched_data_in_file = config.save_filtered_matched_data_in_file;

    %% Start processing the subfiles
    iq_file_list = dir(dir_iq);
    orientation_using_imu_file_list = dir(dir_ori_mocap);
    gps_file_list = dir(dir_gps);
    gps_tx_file_list = dir(dir_gps_tx);
    
    for ii=3:length(iq_file_list)
        fprintf("-----------------------------------------------------\n");
        h_list_all = [];
        yawList_all = [];
        csi_rx_1 = [];
        csi_rx_2 = [];
        true_azimuth_all = [];
        x_all = [];
        y_all = [];
        z_all = [];
        fn_iq_data = fullfile(dir_iq, iq_file_list(ii).name);
        fn_disp_drone = fullfile(dir_ori_mocap, orientation_using_imu_file_list(ii).name);
        fn_gps_drone = fullfile(dir_gps, gps_file_list(ii).name);
        fn_gps_tx = fullfile(dir_gps_tx, gps_tx_file_list(3).name);
        
        [true_azimuth, ~] = get_true_AOA_to_moving_tx_using_GPS(fn_gps_drone, fn_gps_tx);
        array_radius = antenna_separation; 
        fprintf("Estimated array Radius: %f \n", array_radius);

        % Signal filtering, generating hlist and yawlist.      
        [temp_csi_rx1, temp_csi_rx_2, temp_h_list, temp_yawList, ~, ~, ~,temp_x,temp_y,temp_z] = get_aoa_drone_sdr_process_subfiles(fn_iq_data, ...
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
                                                                                 config.thrshld, ...
                                                                                 config.pulse_sunset);

        yawList_all = [yawList_all;temp_yawList];
        x_all  = [x_all ;temp_x];
        y_all  = [y_all ;temp_y];
        z_all  = [z_all ;temp_z];
        
        if isempty(temp_h_list)
            h_list_all = [];
            continue;
        else
            h_list_all = [h_list_all;temp_h_list];
            true_azimuth_all = [true_azimuth_all;true_azimuth];
            csi_rx_1 = [csi_rx_1;temp_csi_rx1];
            csi_rx_2 = [csi_rx_2;temp_csi_rx_2];
        end

        if isempty(h_list_all)
            fprintf("Bad data. Waiting for new one\n");
        else            
            
            %%Save the filtered data files:
            if save_filtered_matched_data_in_file
                op_fn_path = split(fn_iq_data,'.');
                op_fn = op_fn_path(1)+"__h1_h2_pos.mat";
                save(op_fn,"csi_rx_1","csi_rx_2","yawList_all", "x_all", "y_all", "z_all");
            end

            list_end = length(yawList_all);
            list_mid = list_end;
            list_start = 1;
            list_iteration = 1;
            list_stop=1;
           

            while(list_iteration<=list_stop)
                    
                figure(8);
                scatter(cos(yawList_all(list_start:list_mid,1)),sin(yawList_all(list_start:list_mid,1)),52,'x');
                title('Interpolated Orientation Combined')
                set(gca, 'FontSize', 12);
                xlabel('unit X-axis');
                ylabel('unit Y-axis');  
                
                % Generate AOA profiles for subarrays
                for sub_array_val=1:sub_arrays 
                    h_list_curr = downsample(h_list_all(list_start:list_mid,:),sub_arrays,sub_array_val-1);
                    yawList_curr = downsample(yawList_all(list_start:list_mid,:),sub_arrays,sub_array_val-1);
                        
                    len = length(yawList_curr);           
                    phiRep2 = repmat(phiRep, [1, 1, subCarriersNum, len]);
                    yawList2 = repmat(reshape(yawList_curr, [1 , 1, 1, len]), [nphi ntheta subCarriersNum, 1]);
                    lambdaRep2 = repmat(lambdaRep, [1, 1, 1, len]);
                    thetaRep2 = repmat(thetaRep, [1, 1, subCarriersNum, len]);
                    
                    % Steering vector two-antenna SAR configuration.
                    e_term = exp(1i*2*pi*array_radius*(cos(phiRep2 - yawList2).*sin(thetaRep2))./lambdaRep2);
                    leftMat= double(squeeze(e_term(:, :, 1, :))); % 360 * 90 * 135
                    rightMat = h_list_curr; % 135 * 134 
                    resultMat = double(prodND_fast(leftMat, rightMat));
                    intermediate_aoaProfile = permute((double(abs(resultMat)).^2), [3, 1, 2]);

                    if(plot_figure)
                        figure(60+sub_array_val);
                        subplot(2,1,1)
                        surf(phiList*180/pi, thetaList*180/pi, intermediate_aoaProfile.', 'EdgeColor', 'none');
                        set(gcf,'Renderer','Zbuffer')            
                        xlabel('Azimuth (Degree)');
                        ylabel('Elevation (Degree)');
                        title(sprintf('Combined AOA profile (side view)'));         
                        subplot(2,1,2)
                        surf(phiList*180/pi, thetaList*180/pi, intermediate_aoaProfile.', 'EdgeColor', 'none');
                        set(gcf,'Renderer','Zbuffer');
                        view(2)
                        title('Combined AOA profile Top View');
                        xlabel('Azimuth (degree)');
                        ylabel('Elevation (degree)');
                    end
            
                    if sub_array_val == 1
                        aoaProfile = intermediate_aoaProfile;
                    else
                        aoaProfile = aoaProfile.*intermediate_aoaProfile; % Final AOA profile = element-wise multiplication of sub-profiles
                    end
            
                    % Get top angle i.e. AOA 
                    [val, idx] = max(aoaProfile(:));
                    [idx_row, idx_col] = ind2sub(size(aoaProfile),idx);
                    azimuth_aoa = phiList(idx_row)*180/pi;
                    elevation_aoa = thetaList(idx_col)*180/pi; %As per the convention
                    fprintf("AOA  for Drone using sub-array profile: Azimuth = %f , Elevation = %f\n", azimuth_aoa,elevation_aoa);
                end
                
                list_start = list_mid;
                list_mid = list_end;
                list_iteration= list_iteration + 1;
        
                % Get top angle i.e. AOA 
                [val, idx] = max(aoaProfile(:));
                [idx_row, idx_col] = ind2sub(size(aoaProfile),idx);
                azimuth_aoa = phiList(idx_row)*180/pi;
                elevation_aoa = (thetaList(idx_col)*180/pi); %As per the convention
                fprintf("AOA  for Drone data using profile product: Azimuth = %f , Elevation = %f\n", azimuth_aoa,elevation_aoa);
    
                figure(9);
%                 subplot(2,1,1)
                surf(phiList*180/pi, thetaList*180/pi, aoaProfile.', 'EdgeColor', 'none');
                set(gcf,'Renderer','Zbuffer');
                set(gca, 'FontSize', 12);
                xlabel('Azimuth (Degree)');
                ylabel('Elevation (Degree)');
                title(sprintf('Final AOA profile'));
                view(0,0);
%                 subplot(2,1,2)
%                 h = pcolor(phiList*180/pi, thetaList*180/pi, aoaProfile.');
%                 set(h, 'EdgeColor', 'none');
%                 view(2)
       
%                 hold on;
%                 ds = ceil(0.1 * length(true_azimuth_all));  %Use only 10% of points to visualize true AOA
%                 true_azimuth_all_final = downsample(true_azimuth_all,ds);
%                 elevation_aoa_list = repelem(elevation_aoa,length(true_azimuth_all_final))';
%                 sz = repelem(100,length(true_azimuth_all_final));
%                 scatter(true_azimuth_all_final,elevation_aoa_list,sz,'o','MarkerFaceColor', 'red','MarkerEdgeColor', 'red', 'MarkerFaceAlpha',.1, 'MarkerEdgeAlpha',.1);
%                 title('Final AOA profile (top view)');
%                 set(gca, 'FontSize', 12);
%                 xlabel('Azimuth (degree)');
%                 ylabel('Elevation (degree)');
%                 hold off;

                hold on;
                zl = zlim; 
                true_azimuth_all_final = repelem(mean(true_azimuth_all),length(zl))';
                p = plot3(true_azimuth_all_final, [0,0], zl, 'Color','r');
                p.LineWidth = 2;
                hold off;

                if in_place_rotation
                    fprintf("True azimuth AOA %f \n", mean(true_azimuth_all(2:length(true_azimuth_all)))); %From center
                    fprintf("Error in AOA estimation- using RX circular trjaectory gps center: %f degree \n", mean(true_azimuth_all(2:length(true_azimuth_all))) - azimuth_aoa);
                else
                    avg_diff = 0;
                    min_diff=1000;
                    closest_true_aoa=0;
                    for kk=1:length(true_azimuth_all)
                        temp = calculate_angle_difference(true_azimuth_all(kk), azimuth_aoa);
                        avg_diff = avg_diff + temp;
                        if temp < min_diff
                            min_diff = temp;
                            closest_true_aoa = true_azimuth_all(kk);
                        end
                    end
                    avg_diff = avg_diff/length(true_azimuth_all);
    %                 fprintf("True azimuth AOA (from trajectory center) %f deg\n", true_azimuth_all(1)); %From center
                    fprintf("Error in AOA estimation- using RX circular trjaectory gps center: %f degree \n", calculate_angle_difference(true_azimuth_all(1), azimuth_aoa)); %From center
                    fprintf("Error in AOA estimation- using RX circular trjaectory gps center (no abs): %f degree \n", calculate_angle_difference_non_abs(true_azimuth_all(1), azimuth_aoa)); %From center
                    fprintf("Mean Error in AOA estimation: %f degree \n", avg_diff); %From center
                    fprintf("Closest true AOA %f deg \n", closest_true_aoa); %From center
                    fprintf("Error in AOA estimation (using closest true AOA): %f degree \n", min_diff); %From center
                end 
                
                format long;    
                fprintf("%s \n",fn_iq_data)
    
                clear e_term;
                clear phiRep2;
                clear lambdaRep2;
                clear thetaRep2;
            end
        end
    end
end

% Helper functions

function angle_difference = calculate_angle_difference_non_abs(angle1, angle2)
    fprintf("%f, %f \n", angle1, angle2);
    difference = angle2 - angle1;
    angle_difference = difference;
end

function angle_difference = calculate_angle_difference(angle1, angle2)
    angle1 = mod(angle1, 360);
    angle2 = mod(angle2, 360);
    absolute_difference = abs(angle1 - angle2);
    angle_difference = min(absolute_difference, 360 - absolute_difference);
end

function real_elevation_estimate = estimate_correct_elevation(elevation_aoa_from_profile, tau)
    angle_val = 90;
    step = 0.25;
    diff = 100;
    real_elevation_estimate = 0;
    while angle_val > 0
        elevation_aoa_from_theory = asin(sin(angle_val*pi/180)/tau) * 180/pi;
        if abs(elevation_aoa_from_theory - elevation_aoa_from_profile) < diff
            real_elevation_estimate = angle_val;
            diff = abs(elevation_aoa_from_theory - elevation_aoa_from_profile);
        end
        angle_val = angle_val - step;
    end
    fprintf("Actual elevation angle = %f degree\n", real_elevation_estimate);
end

function val = diff_360(a,b)
    tmp = a - b;
    if (tmp > 180)
        tmp = tmp- 360;
    elseif(tmp < -180)
        tmp = tmp+360;
    end
    val = tmp;
end