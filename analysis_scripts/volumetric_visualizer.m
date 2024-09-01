data_dir = 'C:/Users/alber/Documents/LightBead Data/Visualization/';                           
fname = '04032024_GCamp6f_a2_r1_w3_n200_labels.h5';
%h5disp(strcat(data_dir,fname));
labels = h5read(strcat(data_dir,fname),"/labels");
plot_dir = 'C:/Users/alber/Documents/LightBead Data/Visualization/plots';

%%
roi_pos_raw = reshape(labels,[226,512,27]);

% FLIP if necessary
%# rotate image, flip so top of the brain is first
%# flip: np.flip(np.rot90(roi_pos_raw,k=2,axes=(0,1)),2)
%roi_pos = np.flip(np.rot90(roi_pos_raw,k=2,axes=(0,1)),2)

%%
stack_file = '04192024_GCamp6f_a1_r2_w3_mean_G.nii';

volume_img_raw = niftiread(strcat(data_dir,stack_file));

% rotate
volume_img_raw=rot90(volume_img_raw,2);
%flip so that first slice is highest (for visualization)
volume_img_raw = flip(volume_img_raw,3);


% min(min(min(volume_img_raw)))
%max(max(max(volume_img_raw)))
%%
imshow(volume_img_raw(:,:,12),[])

%% reconfigure stack for offsets
first_slice_center_idx = 7; %: 0-12, 13 slices
second_slice_start = 14;
second_slice_center_idx = 21; %: 13-27, 15 slices

step_size = 5; %pixels
stack_offset = 10; %pixels
stack_yoffset = 25; %pixels

volume_shifted = zeros(size(volume_img_raw)) + min(min(min(volume_img_raw)));

for idx = 1:27
 if idx <= first_slice_center_idx
    xoffset = (first_slice_center_idx-idx)*5;
    volume_temp = padarray(volume_img_raw(:,:,idx),[xoffset 0], 'post');
    volume_shifted(:,:,idx) = volume_temp(1+xoffset:(226+xoffset),:);
    %
 elseif idx > first_slice_center_idx && idx < second_slice_start
    xoffset = (idx-first_slice_center_idx)*5;
    volume_temp = padarray(volume_img_raw(:,:,idx),[xoffset 0], 'pre');
    volume_shifted(:,:,idx) = volume_temp(1:226,:);
 elseif idx >= second_slice_start && idx <= second_slice_center_idx
    xoffset = (second_slice_center_idx-idx)*5 + stack_offset;
    volume_temp = padarray(volume_img_raw(:,:,idx),[xoffset stack_yoffset], 'post');
    volume_shifted(:,:,idx) = volume_temp(1+xoffset:(226+xoffset),(1+stack_yoffset):(512+stack_yoffset));
 elseif idx > second_slice_center_idx
    xoffset = (idx-second_slice_center_idx)*5; %+ stack_offset;
    volume_temp = padarray(volume_img_raw(:,:,idx),[xoffset 0], 'pre');
    volume_temp = padarray(volume_temp,[0 stack_yoffset], 'post');
    volume_shifted(:,:,idx) = volume_temp(1:226,(1+stack_yoffset):(512+stack_yoffset));
 end
    
end

% after padding, delete slices 9 is about the same as 13? let's delete slices 11-14 for now
volume_shifted_cut = volume_shifted;
volume_shifted_cut(:,:,11:14) = [];

%% test
imshow(volume_shifted_cut(:,:,22),[])

%% set values below threshold to NaN
min_thresh = min(min(min(volume_shifted_cut(volume_shifted_cut>0))))*1.16;
volume_img_filtered = volume_shifted_cut;
volume_img_filtered(volume_shifted_cut < min_thresh) = NaN;

% 3d render
h = slice(volume_img_filtered, [], [], 1:size(volume_shifted_cut,3));
daspect([1 1 0.2])
set(h, 'EdgeColor','none'); %, 'FaceColor','interp'
alpha(.3);
xlim([90,460])
ylim([20,220])

%% 3d render raw
h = slice(volume_shifted_cut, [], [], 1:size(volume_shifted_cut,3));
set(h, 'EdgeColor','none');
daspect([1 1 0.2])
alpha(.15);
caxis([75 90]);
xlim([90,460])
ylim([20,220])
colormap("gray")
axis off

%% 3d render gauss blur
volume_shifted_blur = imgaussfilt(volume_shifted_cut,0.8);
h = slice(volume_shifted_blur, [], [], 1:size(volume_shifted_cut,3));
set(h, 'EdgeColor','none');
daspect([1 1 0.2])
alpha(.12);
caxis([75 90]);
xlim([90,460])
ylim([20,220])
colormap("gray")
axis off