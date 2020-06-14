%% Assumes the input data is all HASTE brain data (note there could be body scans, etc.) 
% prepares and reorganizes HASTE data into another folder
% the data is organized in the following manner: subject > stack id 
% the stack id folder has a folder for dicoms, jpg, and a labels file which
% contains the quality labels

% manually tested and inspected




clear all;
close all;

cd d:/ % navigate to the correct directory
% TO DO: edit the source_prefix,dataSource,caseNames


source_prefix='datasets_for_iqa/original_iqa_dataset/dicoms/';
dataSource=fullfile(source_prefix,'combined_dataset');
caseNames=dir(dataSource);

% organize the data into new folder
reorganizedDataFolder= fullfile(source_prefix,'reorganized_combined_dataset');
if exist(reorganizedDataFolder,'dir')
    rmdir(reorganizedDataFolder,'s')
end
mkdir(reorganizedDataFolder)

TE_UPPERBOUND=150; % SET BASED ON THE MEAN, STD
TE_LOWERBOUND=90;

%% iterate over all the cases

for i=1:size(caseNames,1)
    
    if isstruct(caseNames(i))
        
        caseName=caseNames(i).name;
    else
        caseName=caseNames{i};
    end
    

    % ignore extraneous folders
    if strcmp(caseName,'.') || strcmp(caseName,'..') || strcmp(caseName,'other')
        continue;
    end
    
    source=fullfile(dataSource,caseName);
    fn_case= fullfile(reorganizedDataFolder,caseName);
    if exist(fn_case,'dir')
    else
        mkdir(fn_case)
    end
    
    complete_subject_dicom_data=dicomCollection(source);
    num_stacks=size(complete_subject_dicom_data,1);


    disp('Number of stacks for subject: ')
    num_stacks
    

    num_haste_stacks=0;

    for stack_idx=1:num_stacks

        stack_data=complete_subject_dicom_data(stack_idx,:);
        % get sample file from each vol to read the sequence parameters for
        % the slice 
        sample_files=stack_data(1,end);
        sample_file=sample_files.Filenames{1}(1);
        sample_dicom_data=dicominfo(sample_file);

        % special case
        if (~any(strcmp(fieldnames(sample_dicom_data),'SequenceName')))
            continue;
        end

        % keep only HASTE 256x256
        if ~strcmp(sample_dicom_data.SequenceName,'*h2d1_256')
            stack_idx
            continue;
        end
        
        % exclude body scans, location scans
        if contains(sample_dicom_data.StudyDescription,'Body')
            continue
        end
        
        if ~contains(sample_dicom_data.ProtocolName,'T2 HASTE') || 
            contains(sample_dicom_data.ProtocolName,'CERVIX')
            continue;
        end
        
        
        % filter any outlier HASTE scans e.g., TE
        echo_time=str2double(sample_dicom_data.EchoTime)
        if echo_time > TE_UPPERBOUND ||
            echo_time < TE_LOWERBOUND
            continue;
        end 
        
        

        num_haste_stacks=num_haste_stacks+1;
        [V,spatial,dim]=dicomreadVolume(complete_subject_dicom_data(stack_idx,:));
        V=squeeze(V);
        
        % create a new folder for the stack
        fold_name=fullfile(fn_case,strcat('stack_',num2str(num_haste_stacks)));
        if exist(fold_name,'dir')
        else
            mkdir(fold_name);
        end
        
        % export dicoms and jpegs
        dicoms_folder=fullfile(fold_name,'dicoms');
        jpegs_folder=fullfile(fold_name,'jpegs');
        mkdir(dicoms_folder)
        mkdir(jpegs_folder)


        % save images in the order of 
       
        for j=1:size(V,3)
            dicom_path=sample_files.Filenames{1}(j);
            [full_dicom_source_path,dicom_name,ext]=fileparts(dicom_path);
            tmp=im2double(squeeze(V(:,:,j)));
            tmp=tmp-min(tmp(:));
            tmp=tmp/max(tmp(:));
            imwrite(tmp,fullfile(jpegs_folder,char(strcat(dicom_name,'.png'))));
            copyfile(char(dicom_path),fullfile(dicoms_folder))
            
        end
        

    end

    disp('Number of loaded vols for subject:')
    num_haste_stacks
    disp('completed saving the data for case: ')
    caseName
    
end 