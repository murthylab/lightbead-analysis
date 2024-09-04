
%directory='D:\WayanData\test I2C';
%directory='D:\WayanData\test I2C\a1_r11';
directory='D:\AlbertData\I2C';
cd(directory);
TiffImagesList= dir('*.tif');

Time = [];
Frames = [];

i=1;
while i<length(TiffImagesList) % Loop through the files
    disp(i)
    FileName=TiffImagesList(i).name;
    tifLink = Tiff(FileName, 'r');

    [header, ~, imgInfo] = scanimage.util.opentif(FileName,  'frame', 1 );
    
    for h = 1:size(header.I2CData,2) % Loop through all the cells in the header
        for k = 1:size(header.I2CData{1,h},2) % loop though the subcells
        
   
        if ~isempty(header.I2CData{1,h})
        
            TP=header.I2CData{1,h}(1,k);
            FrameNumberASCIICode=cell2mat(TP{1,1}(2));
            FramenumberStrg=char(nonzeros(FrameNumberASCIICode)');
        
            timestamp=cell2mat(TP{1,1}(1));
            Framenumber=str2double(FramenumberStrg);
        else
            timestamp=0;
            Framenumber=0;    
        end
        Time(end+1) = timestamp;
        Frames(end+1) = Framenumber;
        end
    end
    i=i+1;
end   

% T = table(Time,Frames);
% filename = 'Y:\Wayan\Lightbead\Flies\Method paper\04192024\I2C_frames\a1_r2\a1_r2.xlsx';
% writetable(T,filename)

%plot(Time,Frames)

% directory = '\\cup\murthy\Wayan\Lightbead\Flies\Method paper\04192024\I2C_frames\a1_r6\'
% cd(directory) 
%fileID = fopen('\\cup\murthy\Wayan\Lightbead\Flies\Method paper\06192024\I2C_frames\a1_r1\a1_r1.txt','w');
fileID = fopen('\\cup\murthy\Albert\Multisensory Data\082824_audioLR_a2\I2C\082824_a2_r4.txt','w');
fprintf(fileID,'%6.4f %12.2f\n',[Time;Frames]);
fclose(fileID);
