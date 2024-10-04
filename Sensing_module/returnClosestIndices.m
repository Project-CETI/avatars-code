function [newAngle_times,deltaT_all,startIndex_csi,endIndex_csi,endIndex_angles] = returnClosestIndices(angle_times,csi_times,deltaT)
%Returns indices in the angle vector and the csi vector that closely match
%each other in time.  In particular the start and the end indices.
newAngle_times=[];
deltaT_all=[];
endIndex_csi=[];
endIndex_angles=[];
startIndex_csi=[];
indicesVec=1:length(csi_times);
endIndex_angles=length(csi_times);
%Find CSI reading that is within deltaT of angle reading
for(a=1:length(angle_times))
    firstCSIs=csi_times(csi_times>=angle_times(a));
    if(isempty(firstCSIs))
        endIndex_angles=a;
        endIndex_csi=length(csi_times);
        return;
    end
    newAngle_times(a)=firstCSIs(1);
    %deltaT=csi_times(firstCSI_index)-angle_times(a); %a refined estimate of deltaT
    deltaT_all(a)=abs(angle_times(a)-newAngle_times(a));
    if(a==1)
        startIndex_csi=indicesVec(csi_times==newAngle_times(1));
    end
end
endIndex_csi=indicesVec(csi_times==newAngle_times(a));


