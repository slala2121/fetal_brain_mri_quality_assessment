%% loads HASTE data
% note the slices in the volume are in reverse order compared to the HOROS
% loader

% organizes dicoms by volumes
% orients the images so that phase encode direction is laong the vertical
% axis
% records the dicom filenames


clear all;
close all;


in_plane_res=[];
slice_thickness=[];
te=[];
tr=[];
num_slices=[];
num_stacks=[];
scan_time=[];
dicom_list=[];

%%
cd d:/
source_prefix='datasets_for_iqa/original_iqa_dataset/dicoms/combined_dataset_dicoms/';
source_prefix='datasets_for_iqa/'
source='complete_haste_repo';
caseNames=dir(fullfile(source_prefix,source));

cases_to_exclude=[2;4;6;8;11;17;21;25;28];
%% record distribution of sequence parameters across the stacks


% iterate over all the cases

for i=1:size(caseNames,1)
    
    if isstruct(caseNames(i))
        caseName=caseNames(i).name;
    else
        caseName=caseNames{i};
    end
    
    if strcmp(caseName,'.') || strcmp(caseName,'..') || strcmp(caseName,'other') || strcmp(caseName,'.docx')
        continue;
    end

    % specific to the labeling in the complete_haste_repo
    if isequal(source,'complete_haste_repo')
        case_index=str2num(caseName(end-3:end));
        if ismember(case_index,cases_to_exclude)
            continue
        end
    end
    
    dicom_source=fullfile(source_prefix,source,caseName);
    dicomData=dicomCollection(dicom_source);
    num_vols=size(dicomData,1);

   
    % iterate over the HASTE stacks and note stack params
    
    loaded_vols=0;
    
    for vol=1:num_vols
        
        vol_data=dicomData(vol,:);
        
        % get stack file from each vol to read the orientation
        sample_files=vol_data(1,end);
        sample_file=sample_files.Filenames{1}(1);
        sample_dicom_data=dicominfo(sample_file);

        % special case for case 4
        if (~any(strcmp(fieldnames(sample_dicom_data),'SequenceName')))
            continue;
        end
        
        % keep only HASTE 256x256
        if ~strcmp(sample_dicom_data.SequenceName,'*h2d1_256')
            vol
            continue;
        end
        
        dicom_list=[dicom_list;sample_file];
        % slice params are the same in a single stack
        slice_thickness=[slice_thickness; sample_dicom_data.SliceThickness];
        te=[te; sample_dicom_data.EchoTime];
        tr=[tr; sample_dicom_data.RepetitionTime];
        in_plane_res=[in_plane_res; sample_dicom_data.PixelSpacing(1)]; % assuming isotropic res along the plane
        [V,spatial,dim]=dicomreadVolume(vol_data);
        V=squeeze(V);
        num_slices_stack=size(V,3);
        num_slices=[num_slices;num_slices_stack];
        slice_time_in_sec=sample_dicom_data.RepetitionTime/1000;
        scan_time=[scan_time;num_slices_stack*slice_time_in_sec];
        
        loaded_vols=loaded_vols+1;
        num_stacks=[num_stacks;loaded_vols];


    end
    
end



%% distribution


stats_folder=fullfile(source_prefix,source,'stats');
if exist(stats_folder,'dir')==7
    rmdir(stats_folder)
end
mkdir(stats_folder)


boxplot(slice_thickness);
title('Slice Thickness Distribution across HASTE stacks')
xticks('')
ylabel('Slice Thickness (mm)')
savefig(fullfile(stats_folder,'slice_thickness'));
close

boxplot(in_plane_res);
title('In plane resolution Distribution across HASTE stacks')
xticks('')
ylabel('In plane resolution (mm)')
savefig(fullfile(stats_folder,'in_plane_res'))
close

boxplot(te);
title('Echo Time Distribution across HASTE stacks')
xticks('')
ylabel('Echo Time (ms)')
savefig(fullfile(stats_folder,'te'))
close

boxplot(tr);
title('Repetition Time Distribution across HASTE stacks')
xticks('')
ylabel('Repetition Time (ms)')
savefig(fullfile(stats_folder,'tr'))
close

boxplot(num_slices);
title('Number of Slices/Stack Distribution across HASTE stacks')
xticks('')
ylabel('Number of Slices/Stack')
savefig(fullfile(stats_folder,'num_slices'))
close

boxplot(scan_time);
title('Scan Time Distribution across HASTE stacks')
xticks('')
ylabel('Stack scan time (s)')
savefig(fullfile(stats_folder,'scan_time'))
close

boxplot(num_stacks);
title('Number of Stacks Distribution across subjects')
xticks('')
ylabel('number of stacks')
savefig(fullfile(stats_folder,'num_stacks'))
close

save(fullfile(stats_folder,'stats'),'slice_thickness','in_plane_res','num_slices','scan_time','te','tr','dicom_list','num_stacks')

%% box plot comparing stats across the populations

% aggregate the data
te_over_pop=[];
tr_over_pop=[];
scan_time_over_pop=[];
slice_thickness_over_pop=[];
in_plane_res_over_pop=[];
num_slices_over_pop=[];
num_stacks_over_pop=[];
population_labels={};

%% 
cd d:/
source_prefix='datasets_for_iqa/original_iqa_dataset/dicoms/combined_dataset_dicoms';
% source_prefix='datasets_for_iqa';
source='singleton-control-ps';
% load the stats
load(fullfile(source_prefix,source,'stats','stats.mat'))

te_over_pop=[te_over_pop;te];
tr_over_pop=[tr_over_pop;tr];
scan_time_over_pop=[scan_time_over_pop;scan_time];
slice_thickness_over_pop=[slice_thickness_over_pop;slice_thickness];
in_plane_res_over_pop=[in_plane_res_over_pop;in_plane_res];
num_slices_over_pop=[num_slices_over_pop;num_slices];
num_stacks_over_pop=[num_stacks_over_pop;num_stacks];

for i=1:size(num_stacks,1)
    population_labels{end+1}=source;
end

%%

compare_stats_folder=fullfile(source_prefix,'aggr');
if exist(compare_stats_folder,'dir')==7
    rmdir(compare_stats_folder)
end
mkdir(compare_stats_folder)

boxplot(num_slices_over_pop);
title('Number of slices Distribution across HASTE stacks')
xlabel('Population')
ylabel('Number of slices')

savefig(fullfile(compare_stats_folder,'num_slices'));
close

fig=openfig(fullfile(compare_stats_folder,'num_slices'));
saveas(fig,fullfile(compare_stats_folder,'num_slices.png'));

boxplot(slice_thickness_over_pop);
title('Slice Thickness Distribution across HASTE stacks')
xlabel('Population')
ylabel('Slice Thickness (mm)')
savefig(fullfile(compare_stats_folder,'slice_thickness'));
close

fig=openfig(fullfile(compare_stats_folder,'slice_thickness'));
saveas(fig,fullfile(compare_stats_folder,'slice_thickness.png'));

boxplot(te_over_pop);
title('Echo Time Distribution across HASTE stacks')
xlabel('Population')
ylabel('Echo Time (ms)')
savefig(fullfile(compare_stats_folder,'te'));
close

fig=openfig(fullfile(compare_stats_folder,'te'));
saveas(fig,fullfile(compare_stats_folder,'te.png'));

boxplot(tr_over_pop);
title('Repetition Time Distribution across HASTE stacks')
xlabel('Population')
ylabel('Repetition Time (ms)')
savefig(fullfile(compare_stats_folder,'tr'));
close

fig=openfig(fullfile(compare_stats_folder,'tr'));
saveas(fig,fullfile(compare_stats_folder,'tr.png'));

boxplot(in_plane_res_over_pop);
title('In plane resolution Distribution across HASTE stacks')
xlabel('Population')
ylabel('in plane res (mm)')
savefig(fullfile(compare_stats_folder,'in_plane_res'));
close

fig=openfig(fullfile(compare_stats_folder,'in_plane_res'));
saveas(fig,fullfile(compare_stats_folder,'in_plane_res.png'));

boxplot(scan_time_over_pop);
title('Scan Time Distribution across HASTE stacks')
xlabel('Population')
ylabel('Scan time (s)')
savefig(fullfile(compare_stats_folder,'scan_time'));
close

fig=openfig(fullfile(compare_stats_folder,'scan_time'));
saveas(fig,fullfile(compare_stats_folder,'scan_time.png'));

boxplot(num_stacks_over_pop);
title('Number of Stacks Per Subject Distribution across HASTE stacks')
xlabel('Population')
ylabel('Number of Stacks')
savefig(fullfile(compare_stats_folder,'num_stacks'));
close

fig=openfig(fullfile(compare_stats_folder,'num_stacks'));
saveas(fig,fullfile(compare_stats_folder,'num_stacks.png'));