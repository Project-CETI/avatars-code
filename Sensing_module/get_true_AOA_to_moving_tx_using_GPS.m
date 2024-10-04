function [true_azimuth, true_distance] = get_true_AOA_to_moving_tx_using_GPS(fn_gps_drone, fn_gps_drone_tx)
    true_azimuth= [];
    [lat, long] = get_gps_center(fn_gps_drone);
    
    [tx_latitude_decimal, tx_longitude_decimal] = get_average_gps_of_moving_tx(fn_gps_drone, fn_gps_drone_tx);
    
    heading = get_gps_heading(lat,long, tx_latitude_decimal, tx_longitude_decimal);
    fprintf("Groundtruth AOA w.r.t true north from center : %f degrees\n", heading);
    
    true_distance = haversine(lat,long, tx_latitude_decimal, tx_longitude_decimal);
    fprintf("Distance between RX and TX = %f meters\n", true_distance);
    
    true_azimuth = [true_azimuth; heading];
    data_temp = load(fn_gps_drone);

    for i=1:length(data_temp)
        heading = get_gps_heading(data_temp(i,4), data_temp(i,5),  tx_latitude_decimal, tx_longitude_decimal);
        true_azimuth = [true_azimuth;heading];
    end
end

function [tx_latitude_decimal, tx_longitude_decimal] =  get_average_gps_of_moving_tx(fn_gps_drone, fn_gps_drone_tx)

    all_rx_gps_data = csvread(fn_gps_drone);
    all_tx_gps_data = csvread(fn_gps_drone_tx);
    
    %find the first matching and last match gps values
    first_rx_gps_ts = all_rx_gps_data(1,1);
    last_rx_gps_ts = all_rx_gps_data(end,1);
    first_tx_matching_timestamp = find(all_tx_gps_data(:,1)>=first_rx_gps_ts,1); %Changed from 1 to 2 degree when using with drone 
    last_tx_matching_timestamp = find(all_tx_gps_data(:,1)>=last_rx_gps_ts,1);
    
    distance_val = haversine(all_tx_gps_data(first_tx_matching_timestamp,2), ...
                            all_tx_gps_data(first_tx_matching_timestamp,3), ...
                            all_tx_gps_data(last_tx_matching_timestamp,2), ...
                            all_tx_gps_data(last_tx_matching_timestamp,3));
    fprintf("Displacement of TX when AOA is being calculated %f meters \n",distance_val);
    
    r=6371;
    tx_disp = [];

    lat = all_tx_gps_data(first_tx_matching_timestamp,2) * 3.14 / 180; 
    lon = all_tx_gps_data(first_tx_matching_timestamp,3) * 3.14 / 180;
    x_0_m = r * cos(lat) * cos(lon) * 1000 ; 
    y_0_m = r * cos(lat) * sin(lon) * 1000 ;

    for ii=first_tx_matching_timestamp:last_tx_matching_timestamp
        lat = all_tx_gps_data(ii,2) * 3.14 / 180; 
        lon = all_tx_gps_data(ii,3) * 3.14 / 180;
        x_m = r * cos(lat) * cos(lon) * 1000 ; 
        y_m = r * cos(lat) * sin(lon) * 1000 ;
        temp = [x_m - x_0_m,y_m-y_0_m];
        tx_disp = [tx_disp;temp];
    end

    tx_latitude_decimal = mean(all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp-1,2));
    tx_longitude_decimal = mean(all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp-1,3));

    figure(771);
    plot(all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp,2), ...
        all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp,3));
    title('Trajectory of TX');
    xlabel('Latitude');
    ylabel('Longitude') ;

    figure(772)
    plot(tx_disp(1:length(tx_disp)-1,1), ...
         tx_disp(1:length(tx_disp)-1,2));
    set(gca, 'FontSize', 16);
    title('Trajectory of TX');
    xlabel('X-axis (meters)');
    ylabel('Y-axis (meters)') ;

    figure(773);
    clf;
    plot(all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp,2), ...
        all_tx_gps_data(first_tx_matching_timestamp:last_tx_matching_timestamp,3));
    hold on;
    plot(all_tx_gps_data(first_tx_matching_timestamp,2),all_tx_gps_data(first_tx_matching_timestamp,3),'*','MarkerSize',18);
    plot(all_rx_gps_data(:,4),all_rx_gps_data(:,5));
    plot(all_rx_gps_data(1,4),all_rx_gps_data(1,5),'*','MarkerSize',18);
    hold off;
    title('Trajectory of RX and TX');
    xlabel('Latitude');
    ylabel('Longitude') ;

end


function distance = haversine(lat1, lon1, lat2, lon2)
    % Earth radius in meters
    R = 6371000;

    % Convert latitude and longitude from degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);

    % Haversine formula
    dlat = lat2 - lat1;
    dlon = lon2 - lon1;

    a = sin(dlat/2)^2 + cos(lat1) * cos(lat2) * sin(dlon/2)^2;
    c = 2 * atan2(sqrt(a), sqrt(1-a));

    % Calculate the distance
    distance = R * c;
end