function heading = get_gps_heading(coord1_lat, coord1_long, tx_latitude_decimal, tx_longitude_decimal)
    
    % Example coordinates (latitude, longitude) in decimal degrees
    coord1 = [coord1_lat, coord1_long];  % Replace with your coordinates
    coord2 = [tx_latitude_decimal, tx_longitude_decimal];  % Replace with your coordinates
 
    lat1 = coord1(1);
    lon1 = coord1(2);
    lat2 = coord2(1);
    lon2 = coord2(2);

    % Convert latitude and longitude from degrees to radians
    lat1 = deg2rad(lat1);
    lon1 = deg2rad(lon1);
    lat2 = deg2rad(lat2);
    lon2 = deg2rad(lon2);

    % Calculate the difference in longitudes
    dlon = lon2 - lon1;
    dlat = lat2 - lat1;

    y = sin(dlat);
    x = cos(lon1) * sin(lon2) - sin(lon1) * cos(lon2) * cos(dlat);

    % Calculate the initial bearing (in degrees) and convert it to the range [0, 360]
    initial_bearing = atan2d(y, x);
    initial_bearing = mod(initial_bearing + 360, 360);

    heading = initial_bearing;
    if heading > 180
        heading = heading - 360;
    end
%     fprintf("Heading. %f degrees\n", heading);
end
