% This script takes the raw tiff file as input, compute the sum over x number of frames in the red channel
% while leaving the green channel unaffected. It returns the unafected
% green and sum in red alternating between each other.
 
directory = 'V:\Scope_LightBeads\LB_ExperimentalDATA\07122024_GCaMP8m\a2_r1\';  % Update with the path to your folder containing TIFF files
cd(directory);
TiffImagesList = dir('*.tif');
FileName = TiffImagesList(1).name;
iminfo = imfinfo(FileName);

NframeFile = 28;  % Number of frames per channel in each TIFF file
NumberOfChannelsSaved = 56;  % Total number of channels in each TIFF file

IntegWindSize = 3;  % Integration window size (number of frames to average)
 

Buffer=[];  
Buffer_red=[];  
Buffer_green=[];  
Buffer_temp = [];
counter=0; 
i=1;
while i<length(TiffImagesList) %Loop through the files
    disp(i);
    FileName=TiffImagesList(i).name;
    tifLink = Tiff(FileName, 'r');
    Next=0;

    for FR=1:NframeFile
        for CH=1:NumberOfChannelsSaved
            Next=Next+1;
            tifLink.setDirectory(Next);
            Buffer(:,:,FR,CH)=tifLink.read();
        end
    end
    
    %half = NumberOfChannelsSaved/2
    % Separate the buffers, one for green and one for red    
    if i == 0
        Buffer_green = Buffer(:,:,(IntegWindSize-1):end,1:(NumberOfChannelsSaved/2)); 
    else 
        Buffer_green = Buffer(:,:,:,1:(NumberOfChannelsSaved/2));
    end
    Buffer_red = cat(3,Buffer_temp, Buffer(:,:, :,((NumberOfChannelsSaved/2)+1):end));

    % for FR=1:NframeFile
    %     for CH=1:NumberOfChannelsSaved
    %         Next=Next+1;
    %         tifLink.setDirectory(Next);
    %         if CH<(NumberOfChannelsSaved/2)+1
    %             Buffer_green(:,:,FR,CH)=tifLink.read();
    %         else
    %             Buffer_red(:,:,FR,CH)=tifLink.read();
    %         end   
    %     end   
    % end

    while size(Buffer_red,3)>=IntegWindSize %if the buffer has enough frames
        % Grab the right buffer
        Extracted_green = zeros(512,226,1,(NumberOfChannelsSaved/2)); % 512,226
        Extracted_red = zeros(512,226,IntegWindSize,NumberOfChannelsSaved/2);  %512,226
        for CH=1:NumberOfChannelsSaved
            if CH<(NumberOfChannelsSaved/2)+1             % then only one green frame at a time
                Extracted_green(:,:,1,CH) = Buffer_green(:,:,1,CH);          
            else
                Extracted_red(:,:,1:IntegWindSize,CH) = Buffer_red(:,:,1:IntegWindSize,(CH-NumberOfChannelsSaved/2));
            end    
        end

        Buffer=Buffer(:,:,2:end,:);
        Buffer_green=Buffer_green(:,:,2:end,:);
        Buffer_red=Buffer_red(:,:,2:end,:);   

        % Take the sum
        Integrated=sum(Extracted_red,3);
        for l=1:(NumberOfChannelsSaved/2)
            Integrated(:,:,:,l) = Extracted_green(:,:,:,l);
        end



        SIntegrated=uint16(squeeze(Integrated));
        
        %then save them 
        counter=counter+1;
        COUNT=num2str(counter);
        while length(COUNT)<4
            COUNT=['0',COUNT];
        end
        
        % Reorder the channels so that the green and red alternate
        channel_order = zeros(1,NumberOfChannelsSaved);
        for k=1:(NumberOfChannelsSaved/2)
            % I want 1, 15, 2, 16, ...14, 28  
            % odds (green)
            channel_order(2*k-1) = k;
            % evens (red)
            channel_order(2*k) = k + NumberOfChannelsSaved/2;
        end
    
        for j=1:length(channel_order)
                if j==1
                    imwrite(SIntegrated(:,:,channel_order(j)),['SUM_files_3frames_alternate\SUM_',COUNT,'.tif'],'tif');
                else
                    imwrite(SIntegrated(:,:,channel_order(j)),['SUM_files_3frames_alternate\SUM_',COUNT,'.tif'],'tif','WriteMode','append');
                end
        end
    end
    i=i+1;
    Buffer_temp = Buffer_red;
    
    tifLink.close();
end



