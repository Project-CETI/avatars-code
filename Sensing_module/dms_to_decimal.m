function decimal_degrees = dms_to_decimal(degrees, minutes, seconds, direction)
    % Calculate decimal degrees
    decimal_degrees = degrees + (minutes / 60) + (seconds / 3600);
    
    % Adjust for direction (N, S, E, W)
    if ismember(direction, {'S', 'W'})
        decimal_degrees = -decimal_degrees;
    end
end
