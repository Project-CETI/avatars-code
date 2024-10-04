function[xCenterFinal, yCenterFinal] = get_gps_center(fn)

    all_disp_data = csvread(fn);
    
    figure(77777);
    clf;
    hold on;
    plot(all_disp_data(:,4),all_disp_data(:,5));
    title('GPS trajectory of robot');
    xlabel('x-axis');
    ylabel('y-axis') ;
    
    angles_orig = [];

    %Find coodinates of the array center (assuming the first position is
    %(0,0)) for SAR data collection

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

    plot(all_disp_data(1,4),all_disp_data(1,5),'r*','MarkerSize',18);
    xCenterFinal = mean(all_x_center);
    yCenterFinal = mean(all_y_center);
    plot(xCenterFinal,yCenterFinal,'*','MarkerSize',18);
    set(gca, 'FontSize', 16);
    fprintf("Mean Center coordinates : (%f,%f) \n",xCenterFinal, yCenterFinal);
    hold off;

end