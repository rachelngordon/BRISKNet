
% Data collection

clc;clear;
addpath('./data');


%% read val_dro IDs from JSON
split_path = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data', 'split', 'data_split.json');
SPLIT = jsondecode(fileread(split_path));
% val_ids = SPLIT.val_dro; 
val_ids = SPLIT.test_dro; 

%% spokes-per-frame values to loop over
% spokes_list = [2 4 8 16 24 36];
frames_list = [144 72 36 18 12 8];

%% loop over the 15 val_dro ids
for vid = 1:numel(val_ids)              
    this_id = val_ids{vid};                            
    tok = regexp(this_id, '_sub(\d+)$', 'tokens', 'once');      
    subnum = str2double(tok{1});                       

    %% load only the subject needed for this val_dro id
    load(['sub',num2str(subnum),'.mat']);      
    data = [];                                           
    data{1}.AIF = aif;                     
    data{1}.mask = mask;                             
    data{1}.S0 = S0;                                    
    data{1}.smap = smap;                                   


    %% Generate and Save DRO
    for f = 1:numel(frames_list)
        n_frames = frames_list(f);

        option = [];
        option.n_frames = n_frames; % Optional: increase temporal frames (default follows AIF length, typically 22).
        [simImg,mask,parMap,smap,S0] = gen_DRO(data,option);
        
        save(fullfile('data', sprintf('%s_dro_%dframes.mat', this_id, n_frames)), ...
                'this_id', 'n_frames', 'simImg','mask','parMap','smap','S0', ...
                '-v7.3');
        
        close all;
        % figure(100)
        % for i = 1:size(simImg,3)
        %     imshow(simImg(:,:,i),[0,max(simImg(:))],'InitialMagnification',1600);
        %     frame = getframe(100);
        %     img = frame2im(frame);
        %     [imind cm] = rgb2ind(img,256);
        % end
        
        %% Display masks
        % 
        % disp_mask(mask,S0,1);

        % Radial k-space data generation
        spokes_per_frame = 288 / n_frames;

        n_lvl = 0.05;
        [kspace,traj] = gen_kspace_data(simImg,smap,spokes_per_frame,n_lvl);

        % save k-space separately (depends on spokes)
        save(fullfile('data', sprintf('%s_kspace_%dspf_%dframes.mat', this_id, spokes_per_frame, n_frames)), ...
             'this_id','spokes_per_frame','n_frames','n_lvl','kspace','traj','-v7.3');



    end
    
    % % Radial k-space data generation
    % 
    % % loop through different spokes-per-frame values
    % for sp = 1:numel(spokes_list)
    %     spokes_per_frame = spokes_list(sp);
    %     n_lvl = 0.05;
    %     [kspace,traj] = gen_kspace_data(simImg,smap,spokes_per_frame,n_lvl);
    % 
    % 
    %     %% Radial reconstruction using BART
    %     % NEED BART INSTALLATION TO RUN!
    % 
    %     nt = floor(size(kspace,2)/spokes_per_frame);
    %     kspace_trim = kspace(:,1:spokes_per_frame*nt,:);
    %     traj_trim = traj(:,1:spokes_per_frame*nt);
    % 
    %     [nx,ntview,ncoil] = size(kspace_trim);
    %     kspace_dim = reshape(kspace_trim,[1,nx,spokes_per_frame,nt,ncoil]);
    %     kspace_dim = permute(kspace_dim,[1,2,3,5,4]);
    %     kspace_dim = reshape(kspace_dim,[1,nx,spokes_per_frame,ncoil,1,1,1,1,1,1,nt]);
    % 
    %     clear traj_dim
    %     traj_dim(1,:,:) = real(traj_trim);
    %     traj_dim(2,:,:) = imag(traj_trim);
    %     traj_dim = cat(1,traj_dim,zeros(1,nx,ntview));
    %     traj_dim = traj_dim * (nx/2);
    %     traj_dim = reshape(traj_dim,[3,nx,spokes_per_frame,ones(1,7),nt]);
    % 
    %     smap_dim = reshape(smap,[size(smap,1),size(smap,2),1,ncoil]);
    % 
    %     % TV Regularization, lamb=0.01
    %     reco = bart('pics -S -RT:1024:0:0.01 -i100 -t',traj_dim,kspace_dim,smap_dim);
    % 
    %     grasp_bart = (squeeze(abs(reco)));
    % 
    % 
    %     % save k-space separately (depends on spokes)
    %     save(fullfile('data', sprintf('%s_kspace_%sspf.mat', this_id, spokes_per_frame)), ...
    %          'this_id','spokes_per_frame','n_lvl','kspace','traj','-v7.3');
    % 
    %     % save reconstruction separately (depends on spokes)
    %     save(fullfile('data', sprintf('%s_recon_%sspf.mat', this_id, spokes_per_frame)), ...
    %          'this_id','spokes_per_frame','reco','grasp_bart','-v7.3'); 
    % 
    % end


end

