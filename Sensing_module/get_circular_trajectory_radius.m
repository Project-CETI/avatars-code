function[array_radius] = get_circular_trajectory_radius(displacement_file)
    all_disp_data = csvread(displacement_file);
 
    figure(71);
    clf;
    hold on;
    plot(all_disp_data(:,4),all_disp_data(:,5));
    title('Real trajectory of robot');
    xlabel('X-axis (meters)');
    ylabel('Y-axis (meters)');

    upperBound = 1;
    lowerBound = length(all_disp_data);
    all_x_center = [];
    all_y_center = [];
    for ii=1:50
        % Three points on the circle
        a = ceil((upperBound - lowerBound) .* rand() + lowerBound);
        b = ceil((upperBound - lowerBound) .* rand() + lowerBound);
        c = ceil((upperBound - lowerBound) .* rand() + lowerBound);

        A = [all_disp_data(a,4), all_disp_data(a,5)];
        B = [all_disp_data(b,4), all_disp_data(b,5)];
        C = [all_disp_data(c,4), all_disp_data(c,5)];
        
        % Find the perpendicular bisectors
        midAB = (A + B) / 2;
        midBC = (B + C) / 2;
        midCA = (C + A) / 2;
        slopeAB = -(B(1) - A(1)) / (B(2) - A(2));
        slopeBC = -(C(1) - B(1)) / (C(2) - B(2));
        slopeCA = -(A(1) - C(1)) / (A(2) - C(2));
        interceptAB = midAB(2) - slopeAB * midAB(1);
        interceptBC = midBC(2) - slopeBC * midBC(1);
        interceptCA = midCA(2) - slopeCA * midCA(1);
        xCenter = (interceptAB - interceptBC) / (slopeBC - slopeAB);
        yCenter = slopeAB * xCenter + interceptAB;
        
        % Display the center coordinates
        if(isnan(xCenter) || isnan(yCenter))
            continue;
        else
            all_x_center = [all_x_center;xCenter];
            all_y_center = [all_y_center;yCenter];
        end
    end

    plot(all_disp_data(1,4),all_disp_data(1,5),'r*','MarkerSize',14);
%     hold off;
    xCenterFinal = mean(all_x_center);
    yCenterFinal = mean(all_y_center);
    plot(xCenterFinal,yCenterFinal,'*','MarkerSize',14);
    set(gca, 'FontSize', 12);
    radius_val = sqrt((xCenterFinal - all_disp_data(:,4)).^2 + (yCenterFinal - all_disp_data(:,5)).^2);
    array_radius = mean(radius_val);

    fprintf("Mean Center coordinates : (%f,%f) \n",xCenterFinal, yCenterFinal);
end