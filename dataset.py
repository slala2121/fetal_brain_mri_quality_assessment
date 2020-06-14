import scipy.io
import numpy as np
import os
import re
import csv
import os
import json
import glob
import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img
import tensorflow as tf
import PIL
from PIL import Image
import shutil
import sys
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pydicom 
from PIL import Image



"""
confidence_thresholds: 1d array float 
                        each element in the array is the lower bound of a bin
                        The upperbound is 1.

                        Each interval follows the following format:
                        (a,b) where indices with prediction values >= a and < b are stored.
                        Exception: when b==1, any index with prediction value 1 is stored. 


preds: 1d array (#,)

Outputs a dictionary where keys are tuple of floats representing the bin
                            values are 1d arrays containing the indices rel. to the preds 
"""
def bin_by_confidence(confidence_thresholds,preds):
    ind_by_confidence={}

    num_confidence_bins=len(confidence_thresholds)
    for index_confidence_lb in range(num_confidence_bins):

        confidence_bin_lb=confidence_thresholds[index_confidence_lb]
        confidence_bin_ub=1 if index_confidence_lb==num_confidence_bins-1 \
                            else confidence_thresholds[index_confidence_lb+1]

        # indices are relative to the order of preds
        ind_preds_lb=np.where(preds>=confidence_bin_lb)
        if int(confidence_bin_ub)==1:
            ind_preds_ub=np.where(preds<=confidence_bin_ub)
        else:
            ind_preds_ub=np.where(preds<confidence_bin_ub)


        ind_with_desired_confidence= np.intersect1d(ind_preds_lb,ind_preds_ub)


        ind_by_confidence[(confidence_bin_lb,confidence_bin_ub)]=ind_with_desired_confidence

        
        if len(ind_with_desired_confidence)>0:
            assert np.amin(preds[ind_with_desired_confidence]) >= confidence_bin_lb 
            assert np.amax(preds[ind_with_desired_confidence]) <= confidence_bin_ub

    return ind_by_confidence


"""
dest_folder: str folder path to export data to
data_partition_label: str train/val/test
all_subject_data: 1d array of SubjectData
is_task_iqa: bool 

Saves the data to a numpy array of the following shape: (# slices, 5) with a ***STRING*** data type
For each slice, the attributes are recorded i.e., roi_label,quality_label,data_path...


"""
def export_data_partition(dest_folder,data_partition_label,all_slice_data):

    all_slice_data_to_array=map(lambda x: x.get_attributes(),all_slice_data)
    all_slice_data_to_array=np.array(all_slice_data_to_array)
    np.save(os.path.join(dest_folder,data_partition_label+'.npy'),all_slice_data_to_array)

    # save labels to CSV for computing further metrics
    all_slice_quality_labels=all_slice_data_to_array[:,1].astype('int')
    np.savetxt(os.path.join(dest_folder,data_partition_label+'_labels.csv'),
                all_slice_quality_labels)

    return 

"""
Stores the meta data associated with 2d slice in a volume
"""
class SliceData(object):


    """
    roi_label: int 0: no roi, 1: roi present
    quality_label: int 0: good, 1: bad, -1: uncertain
    subject_folder_name: str subject
    stack_folder_name: str stack
    dicom_path: str filename of the dicom/png with ext e.g., dcm/IMA 
                rel. to the folder that contains all the data  

    """
    def __init__(self,roi_label,quality_label,
                subject_folder_name='',stack_folder_name='',dicom_path=''):
        self.roi_label=roi_label
        self.quality_label=quality_label
        self.dicom_path=dicom_path
        self.stack_folder_name=stack_folder_name
        self.subject_folder_name=subject_folder_name

    def is_valid_for_iqa(self):
        return self.roi_label==1 and self.quality_label!=-1

    def is_bad_slice(self):
        return self.roi_label==1 and self.quality_label==1

    def is_good_slice(self):
        return self.roi_label==1 and self.quality_label==0

    def get_complete_path(self):
        return os.path.join(self.subject_folder_name,self.stack_folder_name,self.dicom_path)
    def get_attributes(self):
        return np.array([self.roi_label,self.quality_label,self.subject_folder_name,
                        self.stack_folder_name,self.dicom_path])


    
    def __eq__(self,another_slice):

        return self.roi_label==another_slice.roi_label and \
            self.quality_label==another_slice.quality_label and \
            self.dicom_path == another_slice.dicom_path and \
            self.stack_folder_name==another_slice.stack_folder_name and \
            self.subject_folder_name == another_slice.subject_folder_name

    def __hash__(self):
        return hash((self.roi_label,self.quality_label,self.dicom_path,self.stack_folder_name,self.subject_folder_name))



"""
Stores meta data associated with the 3d stack
"""
class StackData(object):
    """
    slice_data: 1d array of SliceData objects
    stack_folder_name: str stack
    subject_folder_name: str subject
    """
    def __init__(self,slice_data,stack_folder_name='',subject_folder_name=''):
        self.num_slices=len(slice_data)
        self.slice_data=slice_data
        self.stack_folder_name=stack_folder_name
        self.subject_folder_name=subject_folder_name

    
    def __eq__(self,another_stack):
        return self.num_slices==another_stack.num_slices and \
                set(self.slice_data)==set(another_stack.slice_data) and \
                self.stack_folder_name==another_stack.stack_folder_name and \
                self.subject_folder_name==another_stack.subject_folder_name

    def __hash__(self):
        return hash((self.num_slices,self.stack_folder_name,self.subject_folder_name))
    
    def get_number_bad_slices(self):
        bad_slices=filter(lambda x: x.is_bad_slice(), self.slice_data)
        num_bad_slices=len(bad_slices)
        return num_bad_slices

    def get_number_roi_slices(self):
        roi_slices=filter(lambda x: x.roi_label==1,self.slice_data)
        num_roi_slices=len(roi_slices)
        return num_roi_slices

    def is_contaminated_stack(self):
        is_contaminated=True if self.get_number_bad_slices()>=1 else False
        return is_contaminated

    def get_fraction_contaminated(self):
        return self.get_number_bad_slices()*1.0/self.get_number_roi_slices()


    # heuristic way of deciding whether the stack is a brain/body scan
    def is_brain_stack(self):
        threshold=0.3
        frac_with_roi=self.get_number_roi_slices()*1.0/self.num_slices
        return frac_with_roi>=threshold

    """
    returns the parameters of the stack 
    """
    def get_parameters(self):
        pass


    """
    Returns a 1d array of SliceData objects where the slices are ordered
    """
    def get_ordered_slice_data(self):
        pass




class SubjectData(object):
    """
    stack_data: 1d array of StackData objects
    subject_folder_name: str ID of the subject
    """
    def __init__(self,stack_data,subject_folder_name=''):
        self.num_stacks=len(stack_data)
        self.stack_data=stack_data
        self.subject_folder_name=subject_folder_name


    
    def __eq__(self,another_subject):
        return self.num_stacks==another_subject.num_stacks and \
                self.subject_folder_name == another_subject.subject_folder_name and \
                set(self.stack_data) == set(another_subject.stack_data)
    
    def __hash__(self):
        return hash((self.num_stacks,self.subject_folder_name))
    

    def get_all_slices(self):
        slice_data=map(lambda x: x.slice_data, self.stack_data)
        slice_data=[slice_data for stack_data in self.stack_data for slice_data in stack_data.slice_data] # flatten
        slice_data=np.array(slice_data)

        return slice_data


    def get_number_bad_slices(self):
        num_bad_slices_per_stack=map(lambda x: x.get_number_bad_slices(),
                                    self.stack_data)
        number_bad_slices=sum(num_bad_slices_per_stack)
        return number_bad_slices

    def get_number_roi_slices(self):
        num_roi_slices_per_stack=map(lambda x: x.get_number_roi_slices(),
                                    self.stack_data)
        number_roi_slices=sum(num_roi_slices_per_stack)
        return number_roi_slices

    def get_brain_stacks(self):
        brain_stacks=filter(lambda x: x.is_brain_stack(),self.stack_data)
        return brain_stacks

    def filter_stack_data(self):
        filtered_stacks=map(lambda x: x.get_roi_slices(), self.stack_data)
        return filtered_stacks

    def get_number_contaminated_stacks(self):
        num_contaminated_stacks=len(filter(lambda x: x.is_contaminated_stack(),self.stack_data))
        return num_contaminated_stacks


    def get_subject_data_with_brain_stacks(self):
        brain_stacks=self.get_brain_stacks()
        return SubjectData(brain_stacks,self.subject_folder_name)

    def get_fraction_contamination_per_stack(self):
        return map(lambda x: x.get_fraction_contaminated(),self.stack_data)

    def get_average_stack_contamination(self):
        return np.mean(self.get_fraction_contamination_per_stack())

    def get_average_number_bad_slices_across_stacks(self):
        return np.mean(self.get_number_bad_slices_across_stacks)

    def get_number_bad_slices_per_stack(self):
        return map(lambda x: x.get_number_bad_slices(),self.stack_data)

    def get_number_roi_slices_per_stack(self):
        return map(lambda x: x.get_number_roi_slices(),self.stack_data)


"""
all_subject_data: 1d array of SubjectData

returns 1d array of StackData
"""
def get_all_stack_data(all_subject_data):
    all_stack_data=map(lambda x: x.stack_data,all_subject_data)
    return flatten_nested_list(all_stack_data)

def get_all_slice_data(all_stack_data):
    all_slice_data=map(lambda x:x.slice_data,all_stack_data)
    return flatten_nested_list(all_slice_data)


def flatten_nested_list(nested_list):
    flattened_list=[data_unit for sublist in nested_list \
                                for data_unit in sublist]
    flattened_list=np.array(flattened_list)
    return flattened_list


"""
data_by_stack: 1d array of 1d arrays of SliceData objects representing the stack data 
subject_name: str 
"""
def plot_stack_distribution(data_by_stack,subject_name):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    stack_names=[]
    for stack_index,stack_data in enumerate(data_by_stack):
        # create a bar for this stack
        for slice_index,slice_data in enumerate(stack_data):
            color='gray'
            if slice_data.roi_label==1:
                color='lightcoral' if slice_data.quality_label==1 else 'mediumseagreen'
            ax.bar(stack_index,height=1,width=0.5,bottom=slice_index,color=color,align='center',
                edgecolor='black')
        stack_name=slice_data.dicom_path.split("\\")[-1]
        stack_name=stack_name.split(".")
        stack_name=str(int(stack_name[2]))
        stack_names.append(stack_name)

    ax.set_xlabel("stack ID")
    ax.set_ylabel("slice index")
    xleft=0-1
    num_stacks=len(data_by_stack)
    ax.set_xlim(left=xleft,right=num_stacks+1)
    ax.set_xticks(np.arange(num_stacks))    
    ax.set_xticklabels(stack_names,rotation='vertical')

    ylims=ax.get_ylim()
    max_num_slices=int(ylims[1])+1

    # ax.set_yticks(np.arange(max_num_slices)+0.5)
    # ax.set_yticklabels(np.arange(max_num_slices))

    slice_indices=np.arange(max_num_slices,step=5)
    ax.set_yticks(slice_indices+0.5)
    ax.set_yticklabels(slice_indices)
    ax.set_title("Stack distribution %s" %subject_name)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=4),
                Line2D([0], [0], color='lightcoral', lw=4),
                Line2D([0], [0], color='mediumseagreen', lw=4)]

    ax.legend(custom_lines, ['no brain', 'Bad', 'Good'])

    return fig
    



"""
data_by_stack: 1d array of 1d arrays of SliceData objects representing the stack data 
subject_name: str 
"""
def plot_subject_stack_distribution(subject_data):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)

    stack_names=[]
    for stack_index,stack_data in enumerate(subject_data.stack_data):
        # create a bar for this stack
        for slice_index,slice_data in enumerate(stack_data.slice_data):
            color='gray'
            if slice_data.roi_label==1:
                color='lightcoral' if slice_data.quality_label==1 else 'mediumseagreen'
            ax.bar(stack_index,height=1,width=0.5,bottom=slice_index,color=color,align='center',
                edgecolor='black')
        stack_name=slice_data.dicom_path.split("\\")[-1]
        stack_name=stack_name.split(".")
        stack_name=str(int(stack_name[2]))
        stack_names.append(stack_name)

    ax.set_xlabel("stack ID")
    ax.set_ylabel("slice index")
    xleft=0-1
    num_stacks=subject_data.num_stacks
    ax.set_xlim(left=xleft,right=num_stacks+1)
    ax.set_xticks(np.arange(num_stacks))    
    ax.set_xticklabels(stack_names,rotation='vertical')

    ylims=ax.get_ylim()
    max_num_slices=int(ylims[1])+1

    # ax.set_yticks(np.arange(max_num_slices)+0.5)
    # ax.set_yticklabels(np.arange(max_num_slices))

    slice_indices=np.arange(max_num_slices,step=5)
    ax.set_yticks(slice_indices+0.5)
    ax.set_yticklabels(slice_indices)
    ax.set_title("Stack distribution %s" %subject_data.subject_folder_name)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='gray', lw=4),
                Line2D([0], [0], color='lightcoral', lw=4),
                Line2D([0], [0], color='mediumseagreen', lw=4)]

    ax.legend(custom_lines, ['no brain', 'Bad', 'Good'])

    return fig


"""
Based on the saveHASTEImages script, assumes the same order
in data_by_stack and all_dicom_names

"""
def save_dicoms_by_stack(source):

    text_file=open(os.path.join(source,'vol_dicom_names.txt'),"r")
    all_dicom_names=text_file.readlines()
    all_dicom_names_reformatted=map(lambda x: reformat_dicom_name(x), all_dicom_names)
    text_file.close()
    data_by_stack=scipy.io.loadmat(os.path.join(source,'loaded_haste_vols_v2.mat'))

    images_by_stack=data_by_stack['subject_data_by_vol'][0]
    num_stacks=len(images_by_stack)
    curr_stack_head=0
    for i in range(num_stacks):
        num_slices=images_by_stack[i].shape[-1]
        stack_dicom_fnames=all_dicom_names_reformatted[curr_stack_head:curr_stack_head+num_slices]
        curr_stack_head+=num_slices

        np.save(os.path.join(source,'vol_%d'%(i+1),'stack_dicom_names'),
            stack_dicom_fnames)




"""
dicom_name: str path of the dicom file

outputs the part of the dicom path just containing the dicom filepath with the extension
"""
def reformat_dicom_name(dicom_name):
    reformatted_dicom_name=dicom_name[:]
    reformatted_dicom_name=reformatted_dicom_name.strip()
    # get only the dicom part of the path
    path_parts=reformatted_dicom_name.split("\\")
    dicom_path=path_parts[-1]
    return dicom_path

"""
stack_labels_dict: key: str image name value: label
dicom_names: 1d array str 
            Assumes dicom_names are of the format: XXX.IMA
            corresponds to the order of the slices i.e., 1st dicom is for "slice_1"

Changes the keys of the stack_labels_dict to the ones in dicom_names
"""
def map_keys_to_dicom(stack_labels_dict,dicom_names):
    num_slices=len(dicom_names)

    stack_labels_dict_with_dicom_keys={}

    for slice_index in range(num_slices):
        slice_dicom_name=dicom_names[slice_index]
        labeled_slice_index=slice_index+1
        stack_labels_dict_with_dicom_keys[slice_dicom_name]=stack_labels_dict['slice_%d'%labeled_slice_index]

    return stack_labels_dict_with_dicom_keys
"""
source: str path to folder for stack data
        Assumes of the following format: ../../../dicom_source_folder/subject_folder/stack_folder

Outputs a StackData
"""
def get_stack_data(source):
    stack_data=[]
    labels_file=glob.glob(os.path.join(source,'*.csv'))[0]
    
    stack_roi_labels,stack_quality_labels=load_single_stack_labels(labels_file)
    dicom_names_file_path=os.path.join(source,'stack_dicom_names.npy')
    if os.path.isfile(dicom_names_file_path):
        dicom_names=np.load(dicom_names_file_path)
        dicom_names=map(lambda x: reformat_dicom_name(x),dicom_names)
        dicom_names=np.array(dicom_names)
        stack_roi_labels=map_keys_to_dicom(stack_roi_labels,dicom_names)
        stack_quality_labels=map_keys_to_dicom(stack_quality_labels,dicom_names)

    stack_roi_labels={slice_key: reformat_roi_label(slice_label) for slice_key, slice_label in stack_roi_labels.items()}
    stack_quality_labels={slice_key: reformat_quality_label(slice_label) for slice_key, slice_label in stack_quality_labels.items()}

    slice_keys=stack_roi_labels.keys()
    stack_data=[]
    for slice_key in slice_keys:
        roi_label=stack_roi_labels[slice_key]
        quality_label=stack_quality_labels[slice_key]
        # attach only the part of the path relevant to the dicom source folder
        source_path_parts=source.split("/")
        subject_folder_name,stack_folder_name=source_path_parts[-2:]
        slice_data=SliceData(roi_label,quality_label,
                            subject_folder_name=subject_folder_name,
                            stack_folder_name=stack_folder_name,
                            dicom_path=slice_key)
        stack_data.append(slice_data)

    stack_data=np.array(stack_data)


    return StackData(stack_data,subject_folder_name=subject_folder_name,
                    stack_folder_name=stack_folder_name)




"""
source: str

Outputs a 1d array of arrays where each array indexes a stack 
"""
def get_subject_data(source):
    subject_data_by_stack=[]
    subject_folder_name=os.path.split(source)[-1]

    for stack_folder_name in os.listdir(source):
        if not os.path.isdir(os.path.join(source,stack_folder_name)):
            continue

        stack_data=get_stack_data(os.path.join(source,stack_folder_name))
        if stack_data.num_slices>0:
            subject_data_by_stack.append(stack_data)

    subject_data_by_stack=np.array(subject_data_by_stack)
    return SubjectData(subject_data_by_stack,subject_folder_name)
"""
image: 3d array (#row, #col, 1)
"""
def reshape_image_for_transfer(image):
    reshaped_image=np.concatenate((image,image,image),axis=2)
    return reshaped_image



"""
fname: str assumes of the following form ../../CASENAME/VOL

Returns the part containing the case/volume i.e., CASENAME/VOL
"""
def extract_relevant_part_filename(fname):

    path_parts=fname.split("/")
    relevant_path_parts=path_parts[-2:]

    relevant_part_fname=os.path.join(relevant_path_parts[0],relevant_path_parts[1])

    return relevant_part_fname


"""
data_partition: 4d-list
    order: images, iqa_labels, roi_labels, fnames
    fnames: each filename path will end in at least this form: caseCASE_NUM/vol_VOL_NUM_SLICE_NUM
correction_labels: (# files, correction_type), 
                correction types: index 0 corresponds to  no_roi, mislabeled

correpsonding_fnames: (# files, 1) -- order of this does not nec. corresponding to the order in data partition
                    will end in at least this form: ../caseCASE_NUM/vol_VOL_NUM_SLICE_NUM

Outputs the data_partition, corrected version

"""
def correct_dataset(data_partition,correction_labels,corresponding_fnames):
    images,iqa_labels,roi_labels,fnames=data_partition

    relevant_part_fnames=np.array(map(extract_relevant_part_filename,fnames))

    # due to the weird format from matlab need to process this differently
    relevant_part_corresponding_fnames=np.array(map(extract_relevant_part_filename,corresponding_fnames))

    corrected_iqa_labels=iqa_labels.copy()
    corrected_roi_labels=roi_labels.copy()

    ind_corrections=np.where(np.any(correction_labels,axis=1)==True)[0]
    for ind_correction in ind_corrections:
        corrected_fname=relevant_part_corresponding_fnames[ind_correction]
        index_correction_rel_dataset=np.where(relevant_part_fnames==corrected_fname)[0]

        if correction_labels[ind_correction][0]:
            corrected_roi_labels[index_correction_rel_dataset]=1-corrected_roi_labels[index_correction_rel_dataset]

        if correction_labels[ind_correction][1]:
            corrected_iqa_labels[index_correction_rel_dataset]=1-corrected_iqa_labels[index_correction_rel_dataset]


    return images,corrected_iqa_labels,corrected_roi_labels,fnames






"""
TODO TUNE noise parameters
img: 3d tensor (width,height,channels) dtype:uint16
"""
def add_noise(img):
    noise_std=1
    noise=np.random.normal(loc=0,scale=noise_std,size=img.shape)
    noise_img=img+noise

    return noise_img


"""
Generator 

skeleton code: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly


Generating augmentations source code:
-https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/image_data_generator.py
-https://github.com/keras-team/keras-preprocessing/blob/master/keras_preprocessing/image/affine_transformations.py

other options:
multi-input generator 

options:
-https://github.com/keras-team/keras/issues/8130
-https://github.com/keras-team/keras/issues/2568

Decision: implement the Sequence class 
this will make it easier to apply my own processing on the input before augmentations, etc
e.g., brain masking or apply the same augmentations to the input images?

also if I want to incorporate multiple reference inputs, my own data generator class
would support doing this although
it might be more complex logic... so might want to rethink this


"""

"""
ROTATION_RANGE=360
WIDTH_SHIFT_RANGE=20
HEIGHT_SHIFT_RANGE=20
BRIGHTNESS_RANGE= [0.25,1.25]
SHEAR_RANGE=0.0
ZOOM_RANGE=0.0
CHANNEL_SHIFT_RANGE= 20.0
HORIZONTAL_FLIP=True
VERTICAL_FLIP=True
FILL_MODE='nearest'
CVAL=0
PREPROCESSING_FUNCTION= add_noise
"""

ROTATION_RANGE=20
WIDTH_SHIFT_RANGE=20
HEIGHT_SHIFT_RANGE=20
BRIGHTNESS_RANGE= None # [0.25,1.25]
SHEAR_RANGE=0.0
ZOOM_RANGE=0.0
CHANNEL_SHIFT_RANGE= 0.0 #20.0
HORIZONTAL_FLIP=True
VERTICAL_FLIP=True
FILL_MODE='constant'
CVAL=0
PREPROCESSING_FUNCTION= None #add_noise

def generate_generator_multiple(generator,dir1, dir2, batch_size, img_height,img_width):
    genX1 = generator.flow_from_directory(dir1,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)
    
    genX2 = generator.flow_from_directory(dir2,
                                          target_size = (img_height,img_width),
                                          class_mode = 'categorical',
                                          batch_size = batch_size,
                                          shuffle=False, 
                                          seed=7)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[0]], X2i[1]  #Yield both images and their mutual label

"""
mask: 2d array binary mask
assumes mask has a circle
"""
def add_context_padding(mask):
    new_mask=np.zeros(mask.shape)
    mask_indices=np.where(mask==1)
    top_row=np.amin(mask_indices[0])
    bottom_row=np.amax(mask_indices[0])
    left_col=np.amin(mask_indices[1])
    right_col=np.amax(mask_indices[1])

    new_mask[top_row:bottom_row,left_col:right_col]=1

    return new_mask

    

"""

image_path: str corresponding to the path of the dicom ending with some dicom extension

For now, loads from the png/jpg instead of dicom

Returns ndarray representing the image
"""
def get_image_array(data_source_dir,subject_folder_path,stack_folder_path,dicom_path):
    dicom_path_sans_ext,ext=os.path.splitext(dicom_path)
    image_file=glob.glob(os.path.join(data_source_dir,
                                    subject_folder_path,stack_folder_path,
                                    'jpegs',dicom_path_sans_ext+"*"))[0] #handles any extension
    
    image_obj=Image.open(image_file)
    image=np.array(image_obj)

    return image




class DataGenerator(keras.utils.Sequence):

    """
    data_partition_path: str path to the .npy file containing the meta data
                        for the slices in this data partition
    dicom_folder_path: str path to the folder containing all the dicom/jpeg data
    batch_size
    dim: if the # channels == 3, the image is reformatted by replicating the image along the channels
    shuffle
    augmentation_flag
    save_images
    save_images_path
    save_labels: Bool should only be used when evaluating
    """
    def __init__(self, data_partition_path, data_source_dir,
                batch_size=1, dim=(256,256,1),
                shuffle=False, augmentation_flag=False,
                save_images=False,save_images_path='',save_labels=False):
        
        self.all_slice_data=np.load(data_partition_path)
        self.num_instances=len(self.all_slice_data)
        self.data_source_dir=data_source_dir
        self.batch_size = batch_size
        self.dim=dim
        self.shuffle = shuffle
        self.augmentation_flag=augmentation_flag
        if self.augmentation_flag:
            self.image_transform_gen=ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True,
                                            rotation_range=ROTATION_RANGE,
                                            width_shift_range=WIDTH_SHIFT_RANGE,
                                            height_shift_range=HEIGHT_SHIFT_RANGE,
                                            brightness_range=BRIGHTNESS_RANGE,
                                            shear_range=SHEAR_RANGE,
                                            zoom_range=ZOOM_RANGE,
                                            channel_shift_range=CHANNEL_SHIFT_RANGE,
                                            horizontal_flip=HORIZONTAL_FLIP,
                                            vertical_flip=VERTICAL_FLIP,
                                            fill_mode=FILL_MODE,
                                            cval=CVAL,
                                            preprocessing_function=PREPROCESSING_FUNCTION)
        else: # only normalize image
            self.image_transform_gen=ImageDataGenerator(samplewise_center=True,
                                            samplewise_std_normalization=True)

        self.save_images=save_images
        if self.save_images:
            self.save_images_path=os.path.join(save_images_path)
            if os.path.isdir(self.save_images_path):
                shutil.rmtree(self.save_images_path)
            os.mkdir(self.save_images_path)

        # useful for the test dataset
        self.save_labels=save_labels
        if self.save_labels:
            self.labels=[]



        self.on_epoch_end() # reshuffles data

    """
    Number of batches per epoch
    """
    def __len__(self):
        return int(np.floor(self.num_instances / self.batch_size))

    """
    Retrieves a batch of data

    Outputs a batch of images and corresponding quality labels
    """
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the data points relative to the total set of images in this partition
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X,y=self.data_generation(indexes)

        
        if self.save_labels:
            self.labels.extend(y)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.num_instances)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    """
    Outputs the batch of images and labels

    Reads the image files rather than dicoms to potentially make data reading faster and less memory intensive
    """
    def data_generation(self,indices):
        # Initialization
        n_channels=self.dim[-1]
        X = np.empty((self.batch_size, self.dim[0],self.dim[1], n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for index_rel_batch, index_rel_to_original_data in enumerate(indices):
            # load image and label
            slice_data=self.all_slice_data[index_rel_to_original_data]
            roi_label,quality_label,subject_folder_path,stack_folder_path,dicom_path=slice_data
            roi_label=int(roi_label)
            quality_label=int(quality_label)
            # maybe reading just the raw png is faster than the dicom? less memory?
            # strip any extension in case

            # image_data=pydicom.dcmread(os.path.join(self.dicom_folder_path,dicom_path))
            # image=image_data.pixel_array
            """
            dicom_path_sans_ext,ext=os.path.splitext(dicom_path)
            image_file=glob.glob(os.path.join(self.data_source_dir,
                                            subject_folder_path,stack_folder_path,'jpegs',dicom_path_sans_ext+"*"))[0] #handles any extension
            
            image_obj=Image.open(image_file)
            image=np.array(image_obj)
            """

            image=get_image_array(self.data_source_dir,
                                    subject_folder_path,stack_folder_path,dicom_path)
            label=quality_label

            image=image.astype('float')
            # image*=255/np.amax(image) # rescale to 0-255 for data augmentations involving intensity shifts
            image=np.reshape(image,(self.dim[0],self.dim[1],1))

            transform_parameters=self.image_transform_gen.get_random_transform((self.dim[0],self.dim[1],1))
            image=self.image_transform_gen.apply_transform(image,transform_parameters)
            image=self.image_transform_gen.standardize(image)

            
            if n_channels==3:
                image=reshape_image_for_transfer(image)
            
            if self.save_images:
                converted_image=array_to_img(image)
                image_file=os.path.join(self.save_images_path,'im_id_%d_%d.jpg'
                    %(index_rel_to_original_data,np.random.randint(0,100000)))
                converted_image.save(image_file)


            X[index_rel_batch,]=image
            y[index_rel_batch]=label

        # print(X.shape,y.shape)
        return X, y
"""
vol_images: 1d array (# volumes, )
all_vol_fnames_concatenated: 1d array (# slices altogether in stack)
"""
def organize_dicom_names_by_vol(vol_images,all_vol_fnames_concatenated):
    fnames_by_vol=[]
    slice_index_start_vol=0
    num_vols=len(vol_images)
    for i in range(num_vols):
        num_slices_in_vol=vol_images[i].shape[-1]
        slice_index_end_vol=slice_index_start_vol+num_slices_in_vol
        vol_fnames=all_vol_fnames_concatenated[slice_index_start_vol:slice_index_end_vol]
        fnames_by_vol.append(vol_fnames)
        slice_index_start_vol=slice_index_end_vol

    fnames_by_vol=np.array(fnames_by_vol)

    return fnames_by_vol

"""
fname: path representing a case

Returns 5 1d arrays for each case, containing data per vol i.e., length of each array is the number of volumes in this subject


all_vol_names: 1d array length: # vols in the subject
                each array contains the folder it came from followed by the slice index
                the indexing starts at 1 

"""
def load_subject_data(fname):
    data_file_name=os.path.join(fname,'loaded_haste_vols_v2.mat')
    data=scipy.io.loadmat(data_file_name)
    vol_images=data['subject_data_by_vol']
    vol_images=np.reshape(vol_images,vol_images.shape[1]) # reshapes it into 1d array
    num_vols=len(vol_images)


    all_vol_fnames_concatenated=np.loadtxt(os.path.join(fname,'vol_dicom_names.txt'),dtype='str')

    vol_fnames=organize_dicom_names_by_vol(vol_images,all_vol_fnames_concatenated)
    """
    all_vol_fnames=[]
    for i in range(num_vols):
        vol_fnames=np.array([os.path.join(fname,'vol_%d_%d'%(i+1,j+1)) for j in range(vol_images[i].shape[-1])])
        
        all_vol_fnames.append(vol_fnames)
    """


    vol_roi_labels,vol_quality_labels=load_volumes_labels(fname)


    return vol_images, vol_roi_labels, vol_quality_labels, vol_fnames 

"""
labels_file: str csv path

Returns 2 arrays
"""
def load_single_volume_labels(labels_file):
    # for each vol agregate the roi, quality labels

    labels_by_slice={}

    with open(labels_file) as csvfile:
        reader=csv.DictReader(csvfile)
        for row in reader:
            labels=json.loads(row['Label'])
            fname=row['External ID']

            slice_key=fname.strip('.png')
            labels_by_slice[slice_key]=[]
            if 'roi' in labels:

                if labels['roi']=='no': # explicitly labeled no
                    labels_by_slice[slice_key].append('no')
                    labels_by_slice[slice_key].append('bad') # for now label these bad
                else:
                    labels_by_slice[slice_key].append('yes')

                    if 'image_quality' in labels.keys():
                        labels_by_slice[slice_key].append(labels['image_quality'])
                    elif 'good/bad/uncertain' in labels.keys():
                        labels_by_slice[slice_key].append(labels['good/bad/uncertain'])

            else: # assumes that if roi is not labeled, then it is present
                labels_by_slice[slice_key].append('yes')
                if 'image_quality' in labels.keys():
                    labels_by_slice[slice_key].append(labels["image_quality"])
                elif 'good/bad/uncertain' in labels.keys():
                        labels_by_slice[slice_key].append(labels['good/bad/uncertain'])
            
    
    
    # reorder the data in the order of slices

    roi_labels_by_slice=[]
    quality_labels_by_slice=[]
    num_slices=len(labels_by_slice.keys())
    for i in range(1,num_slices+1):
        roi_labels_by_slice.append(labels_by_slice['slice_'+str(i)][0])
        quality_labels_by_slice.append(labels_by_slice['slice_'+str(i)][1])

    return np.array(roi_labels_by_slice),np.array(quality_labels_by_slice)
    





"""
labels_file: str csv path

Returns a dictionary mapping each dicom key (str) to an array ROI, Quality label
"""
def load_single_stack_labels(labels_file):
    # for each vol agregate the roi, quality labels

    roi_labels_by_slice={}
    quality_labels_by_slice={}

    with open(labels_file) as csvfile:
        reader=csv.DictReader(csvfile)
        for row in reader:
            labels=json.loads(row['Label'])
            fname=row['External ID']

            slice_key=fname.strip('.png')
            roi_labels_by_slice[slice_key]=None
            quality_labels_by_slice[slice_key]=None
            if 'roi' in labels:

                if labels['roi']=='no': # explicitly labeled no
                    roi_labels_by_slice[slice_key]='no'
                    quality_labels_by_slice[slice_key]='bad' # for now label these bad
                else:
                    roi_labels_by_slice[slice_key]='yes'

                    # currently 2 versions of the keys that correspond to image quality
                    if 'image_quality' in labels.keys():
                        quality_labels_by_slice[slice_key]=labels['image_quality']
                    elif 'good/bad/uncertain' in labels.keys():
                        quality_labels_by_slice[slice_key]=labels['good/bad/uncertain']

            else: # assumes that if roi is not labeled, then it is present
                roi_labels_by_slice[slice_key]='yes'
                if 'image_quality' in labels.keys():
                    quality_labels_by_slice[slice_key]=labels["image_quality"]
                elif 'good/bad/uncertain' in labels.keys():
                    quality_labels_by_slice[slice_key]=labels['good/bad/uncertain']
            
    
    
    return roi_labels_by_slice,quality_labels_by_slice

"""
loads a subject's data by vol in numerical order

"""
def load_volumes_labels(subject_fname):
    num_vols=0
    for tmp_folder in os.listdir(subject_fname):
        if tmp_folder.endswith('.mat') or tmp_folder.endswith('.txt'):
            continue
        num_vols+=1


    roi_labels_by_vol=[]
    quality_labels_by_vol=[]
    for vol_index in range(1,num_vols+1):
        vol_folder_name=os.path.join(subject_fname,'vol_%d'%vol_index)
        cwd=os.getcwd()
        os.chdir(vol_folder_name)
        labels_file=glob.glob('*.csv')[0]

        vol_roi_labels,vol_quality_labels=load_single_volume_labels(labels_file)

        roi_labels_by_vol.append(vol_roi_labels)
        quality_labels_by_vol.append(vol_quality_labels)

        os.chdir(cwd)


    roi_labels_by_vol=np.array(roi_labels_by_vol)
    quality_labels_by_vol=np.array(quality_labels_by_vol)


    return roi_labels_by_vol,quality_labels_by_vol
    


"""
Loads the volumetric train and test sets and the corresponding dicom filenames.

Assumes that the images are oriented so that the phase encode is along the vertical axis.

output: 


"""
def load_all_subjects_data(data_source_dir='',case_order_path=''):

    #print(data_source_dir)
    """
    all_subject_data=np.array([load_subject_data(os.path.join(data_source_dir,fname)) \
                        for fname in os.listdir(data_source_dir) \
                        if os.path.isdir(os.path.join(data_source_dir,fname))])
    """
    case_order=np.load(case_order_path)
    # print(case_order)
    all_subject_data=np.array([load_subject_data(os.path.join(data_source_dir,fname)) \
                        for fname in case_order \
                        if os.path.isdir(os.path.join(data_source_dir,fname))])
    all_subject_images= all_subject_data[:,0]
    all_subject_roi_labels=all_subject_data[:,1]
    all_subject_quality_labels=all_subject_data[:,2]
    all_subject_fnames=all_subject_data[:,3]


    return all_subject_images,all_subject_roi_labels,all_subject_quality_labels, all_subject_fnames
    
    # return all_subject_data

def getTrainValTestBySubject(all_cases_data,all_cases_labels,all_cases_dicom_names,test_cases_indices):
    num_cases=len(all_cases_data)
    
    cases_indices=np.arange(num_cases)
    train_val_cases_indices=np.setdiff1d(cases_indices,test_cases_indices)

    # divide train into train and validation set
    num_valid_cases=1
    train_cases_indices=train_val_cases_indices[:-num_valid_cases]
    val_cases_indices=train_val_cases_indices[-num_valid_cases:]


    train_images=all_cases_data[train_cases_indices]
    train_labels=all_cases_labels[train_cases_indices]
    train_dicoms=all_cases_dicom_names[train_cases_indices]

    val_images=all_cases_data[val_cases_indices]
    val_labels=all_cases_labels[val_cases_indices]
    val_dicoms=all_cases_dicom_names[val_cases_indices]

    test_images=all_cases_data[test_cases_indices]
    test_labels=all_cases_labels[test_cases_indices]
    test_dicoms=all_cases_dicom_names[test_cases_indices]


    # flatten so data is separated by vol rather than subjects
    train_by_vol=np.concatenate(train_images)
    val_by_vol=np.concatenate(val_images)
    test_by_vol=np.concatenate(test_images)

    train_labels_by_vol=np.concatenate(train_labels)
    val_labels_by_vol=np.concatenate(val_labels)
    test_labels_by_vol=np.concatenate(train_labels)


    train_dicoms_by_vol=np.concatenate(train_dicoms)
    val_dicoms_by_vol=np.concatenate(val_dicoms)
    test_dicoms_by_vol=np.concatenate(test_dicoms)





"""
all_cases_dataset: 1d array containing the following arrays:
    -1d array: images
    -1d_array: roi labels
    -1d array: quality labels
    -1d array: fnames

cases_indices: lsit of int

Above data is organized by volume

Reorganizes the images so that the slice dimension is the first

partition_data_reformatted: 1d array (# vols, )
partition_roi_labels_reformatted: 1d array (# vols, )
partition_quality_labels_reformatted: 1d array (# vols, )
partition_fnames_reformatted: 1d array (# vols, )

"""
def getDatasetPartition(all_cases_dataset,cases_indices):
    
    all_cases_data=all_cases_dataset[0]
    all_cases_roi_labels=all_cases_dataset[1]
    all_cases_quality_labels=all_cases_dataset[2]
    all_cases_fnames=all_cases_dataset[3]

    partition_data=all_cases_data[cases_indices]
    partition_roi_labels=all_cases_roi_labels[cases_indices]
    partition_quality_labels=all_cases_quality_labels[cases_indices]
    partition_fnames=all_cases_fnames[cases_indices]

    # flatten this so it's all volumes
    partition_data=np.concatenate(partition_data)
    partition_roi_labels=np.concatenate(partition_roi_labels)
    partition_quality_labels=np.concatenate(partition_quality_labels)
    partition_fnames=np.concatenate(partition_fnames)


    # reformat the labels 
    num_vols=len(partition_data)
    partition_roi_labels_reformatted=[]
    partition_quality_labels_reformatted=[]
    partition_data_reformatted=[]
    partition_fnames_reformatted=[]
    for i in range(num_vols):
        num_slices=partition_data[i].shape[-1]
        vol_data_reformatted=partition_data[i]
        vol_data_reformatted=np.transpose(vol_data_reformatted,[2,0,1])
        partition_data_reformatted.append(vol_data_reformatted)

        partition_roi_labels_reformatted.append(partition_roi_labels[i])
        partition_quality_labels_reformatted.append(partition_quality_labels[i])
        partition_fnames_reformatted.append(partition_fnames[i])

    partition_data_reformatted=np.array(partition_data_reformatted)
    partition_roi_labels_reformatted=np.array(partition_roi_labels_reformatted)
    partition_quality_labels_reformatted=np.array(partition_quality_labels_reformatted)
    partition_fnames_reformatted=np.array(partition_fnames_reformatted)

    return partition_data_reformatted,partition_roi_labels_reformatted,partition_quality_labels_reformatted,partition_fnames_reformatted

 




"""
data: 1d array where each element in the array is another array containing a volume data 
    (e.g., the first dimension of each subarray array X # slices)


Flattens the volume so that slice data are concatenated

Returns an array of the following dimesnions: (# slices,XXXX) 

"""
def flatten_volume_data(data):
    num_vols=len(data)

    flattened_data=[]
    for i in range(num_vols):
        vol_data=data[i]
        num_slices=len(vol_data)
        for j in range(num_slices):
            flattened_data.append(vol_data[j])

    return np.array(flattened_data)
    




"""
Removes the last few slices to adjust the dataset
size to be divisible by the batch size
"""
def trim_data(data,batch_size):
    num_slices=data.shape[0]

    remainder=num_slices%batch_size

    if remainder==0:
        trim_data=data
    else:
        trim_data=data[:-1*remainder]
    return trim_data

"""
datasetPartition:
    3 1d arrays, where the dimnension of each array is the number of volumes in the partition
"""
def flattenPartition(datasetPartition):

    flattenedDataset=map(flatten_volume_data,datasetPartition)
    """
    trimmedDataset=map(trim_data,flattenedDataset,[batch_size]*len(flattenedDataset))

    trimmedDataset[1]=np.reshape(trimmedDataset[1],len(trimmedDataset[1]))

    return trimmedDataset
    """
    return flattenedDataset




"""
roi_label: str 'yes'/'no'

Output: int 1/0
"""
def reformat_roi_label(roi_label):

    reformatted_roi_label=0
    if roi_label=='yes':
        reformatted_roi_label=1

    return reformatted_roi_label



"""
quality_label: str good/bad/uncertain

Output: int 0/1/-1
"""
def reformat_quality_label(quality_label):

    
    if quality_label=='good':
        reformatted_quality_label=0
    elif quality_label=='bad':
        reformatted_quality_label=1
    else:
        reformatted_quality_label=-1

    return reformatted_quality_label


"""
roi_labels: 1d array
    # slices: 1, >1

    # values: yes, no


Maps yes to 1
"""
def reformat_roi_labels(roi_labels):
    roi_labels_reformatted=np.zeros(roi_labels.shape)
    ind_yes=np.where(roi_labels=='yes')
    roi_labels_reformatted[ind_yes[0]]=1

    roi_labels_reformatted=roi_labels_reformatted.astype('int32')

    return roi_labels_reformatted

"""
quality_labels: 1d array

Category labels
"Bad": 1
"Good": 0 
Uncertain: -1
"""

def reformat_quality_labels(quality_labels):
    quality_labels_reformatted=np.ones(quality_labels.shape)
    ind_good=np.where(quality_labels=='good')
    quality_labels_reformatted[ind_good[0]]=0

    ind_uncertain=np.where(quality_labels=='uncertain')
    quality_labels_reformatted[ind_uncertain[0]]=-1

    quality_labels_reformatted=quality_labels_reformatted.astype('int32')

    return quality_labels_reformatted


"""
Removes images that do not have the roi
"""
def filter_non_roi(datasetPartition):
    images,roi_labels,quality_labels,fnames=datasetPartition

    roi_ind=np.where(roi_labels==1)
    images_roi=[]
    roi_labels_only=[]
    quality_labels_roi=[]
    fnames_roi=[]
    if len(roi_ind)>0:
        images_roi=images[roi_ind]
        roi_labels_only=roi_labels[roi_ind]
        quality_labels_roi=quality_labels[roi_ind]
        fnames_roi=fnames[roi_ind]
    return images_roi,roi_labels_only,quality_labels_roi,fnames_roi


"""
difference from the above is the order in the data
"""
def filter_non_roi_v2(datasetPartition):
    images,quality_labels,roi_labels,fnames=datasetPartition

    roi_ind=np.where(roi_labels==1)
    images_roi=[]
    roi_labels_only=[]
    quality_labels_roi=[]
    fnames_roi=[]
    if len(roi_ind)>0:
        images_roi=images[roi_ind]
        roi_labels_only=roi_labels[roi_ind]
        quality_labels_roi=quality_labels[roi_ind]
        fnames_roi=fnames[roi_ind]
    return images_roi,quality_labels_roi,roi_labels_only,fnames_roi




"""
datasetPartition:
    -contains 3 arrays
    -1st array: images (# slices, 256,256)
    -2nd array: roi labels (# slices,1)
    -3rd array: quality_labels (#slices, 1)
    -4th array: fnames (# slices, 1)


mean: int number to subtract from all pixels Should reflect the mean in the training set

Strips some of the data based onthe batch size
Filters out the non-roi images and reformats the labels for this.


also reconverts to the appropriate data type
"""
def resize_and_filter_partition(datasetPartition,remove_non_roi,batch_size):
    images,roi_labels,quality_labels,fnames=datasetPartition




    roi_labels_reformatted=reformat_roi_labels(roi_labels)
    quality_labels_reformatted=reformat_quality_labels(quality_labels)

    

    if remove_non_roi:
        images,roi_labels_reformatted,quality_labels_reformatted,fnames=filter_non_roi([images,roi_labels_reformatted,
            quality_labels_reformatted,fnames])

    else:
        # ensure that the non-roi labels are marked 'bad'
        ind_non_roi=np.where(roi_labels_reformatted==0)[0]
        quality_labels_reformatted[ind_non_roi]=1





    updatedDataset=[images,roi_labels_reformatted,quality_labels_reformatted,fnames]
    trimmedDataset=map(trim_data,updatedDataset,[batch_size]*4)

    trimmedDataset[1]=np.reshape(trimmedDataset[1],len(trimmedDataset[1]))


    # return images_zero_centered,roi_labels_reformatted,quality_labels_reformatted

    return trimmedDataset




"""
image: 3d array # images , 256 x 256

Downsamples to 227x227 by central cropping
"""
def crop(images,model_name):
    cropped_images=None
    if model_name=='alexnet':
        cropped_images=map(lambda x: x[14:-15,14:-15], images) #resize to 227x227
    if model_name=='resnet50':
        cropped_images=map(lambda x: x[16:-16,16:-16], images) #resize to 224x224


    return np.array(cropped_images)


"""
data augmentations
"""

"""
Rotates the image a random angle 
desired_rot: [0,360]
"""
"""
def rotate(image,desired_rot):
    rows,cols=image.shape
    M=cv2.getRotationMatrix2D((cols/2,rows/2),desired_rot,1)
    return cv2.warpAffine(image,M,(cols,rows))


def translate(image,tx,ty):
    M=np.float32([[1,0,tx],[0,1,ty]])
    rows,cols=image.shape
    trans_image=cv2.warpAffine(image,M,(cols,rows))

    return trans_image


def flip_horizontal(image):
    return cv2.flip(image,0)

def flip_vertical(image):
    return cv2.flip(image,1)

def flip_hor_ver(image):
    return cv2.flip(image,-1)

def zoom(image,zoom_factor):
    return cv2.resize(image,dsize=None,fx=zoom_factor,fy=zoom_factor,interpolation=cv2.INTER_LINEAR)
"""    


"""
image: 2d array
Applies random transformations to the image

considerations:
    -tune the parameters
    -effect of the order of aug
"""
"""
def transform(image):
    
    rand_rot=np.random.randint(0,360)
    rand_tx=np.random.randint(-30,30)
    rand_ty=np.random.randint(-30,30)
    rand_flip=np.random.randint(-1,2)

    trans_image=rotate(image,rand_rot)
    trans_image=translate(trans_image,rand_tx,rand_ty)
    trans_image=cv2.flip(trans_image,rand_flip)
    
    

    return trans_image

"""


"""
images: 3d array
roi_masks: 3d array

Given batch of images, randomly augments image with 0.5 probability
"""
def augment_data(images,roi_masks):
    resX,resY=images.shape[1],images.shape[2]

    aug_images=[]
    aug_roi_masks=[]
    for i in range(len(images)):
        aug_im=images[i]
        aug_mask=roi_masks[i]
        if np.random.binomial(1,0.5)==1:
            aug_im,aug_mask=transform(aug_im,aug_mask)
        aug_images.append(aug_im)
        aug_roi_masks.append(aug_mask)

     
    aug_images=np.array(aug_images)
    aug_roi_masks=np.array(aug_roi_masks)
    
    return aug_images,aug_roi_masks


def remove_quality_label_images(datasetPartition,num_images_to_remove,label):
    images,roi_labels,quality_labels=datasetPartition

    ind_bad_images=np.where(quality_labels==label)[0]

    ind_bad_images_to_remove=np.random.choice(ind_bad_images,num_images_to_remove,replace=False)

    ind_images_to_keep=np.setdiff1d(np.arange(len(quality_labels)),ind_bad_images_to_remove)

    images_to_keep=images[ind_images_to_keep]
    roi_labels_to_keep=roi_labels[ind_images_to_keep]
    quality_labels_to_keep=quality_labels[ind_images_to_keep]

    return images_to_keep,roi_labels_to_keep,quality_labels_to_keep


"""
images: 3d array (# images, width, height) float dtype


crop, deal with the gray scale vs RGB

"""
def resize_and_format_for_resnet(images):
    reformatted_images=map(lambda x: x[16:-16,16:-16], images) #resize to 224x224
    reformatted_images=np.array(reformatted_images)

    #reformatted_images*=1./255

    # convert to RGB
    # https://discuss.pytorch.org/t/transfer-learning-grayscale-image-size-and-activation-function/9953/2

    width=224
    height=224
    reformatted_images=reformatted_images.reshape(reformatted_images.shape[0],width,height,1)
    reformatted_images=np.concatenate((reformatted_images,reformatted_images,reformatted_images),axis=3)

    return reformatted_images

"""
images: 3d array (# images, width, height) float dtype

outputs 4d array(# images, width, height, channels), where # channels is 3

"""

def format_for_transfer_resnet(images):
    num_images,width,height=images.shape
    reformatted_images=images.copy()
    reformatted_images=reformatted_images.reshape(reformatted_images.shape[0],width,height,1)
    reformatted_images=np.concatenate((reformatted_images,reformatted_images,reformatted_images),axis=3)

    return reformatted_images



"""
data_source_dir: str path to folder containing dicoms
remove_non_roi: bool whether to exclude slices that don't have the brain

Output:

a list with arrays in the following order:
    -index 0: array with images (# images, Resolution X, Resolution Y)
    -index 1: array with labels indicating whether brain is in slice or not (# images, 1) labels: {0,1} where 0 'not present', 1 are 'bad'
    -index 2: array with quality labels , labels {0,1} where 0 is 'good', 1 is bad
    -index 3: array with filenames for hte images (# images, 1)
"""
def load_dataset(data_source_dir,remove_non_roi):
    dataset_by_subject=load_all_subjects_data(data_source_dir)
    num_subjects=len(dataset_by_subject[0])
    subject_indices=np.arange(num_subjects)
    data_partition=getDatasetPartition(dataset_by_subject,subject_indices)
    data_flattened=flattenPartition(data_partition)
    data_preprocessed=resize_and_filter_partition(data_flattened,remove_non_roi,batch_size=1) # load all images
    data_preprocessed=filter_uncertain_partition(data_preprocessed) # removes images that have quality 'uncertain'

    return data_preprocessed



"""
dataset_partitions:
    3 arrays corresponding to train, val ,test
"""
def load_datasets(dataset_by_subject,remove_non_roi,batch_size,debug,num_images_debug,dataset_partitions,problem_type='iqa'):
    train_ind=dataset_partitions[0]
    val_ind=dataset_partitions[1]
    test_ind=dataset_partitions[2]

    train_partition=getDatasetPartition(dataset_by_subject,train_ind)
    val_partition=getDatasetPartition(dataset_by_subject,val_ind)
    test_partition=getDatasetPartition(dataset_by_subject,test_ind)

    train_flattened,val_flattened,test_flattened=map(flattenPartition,
                                    [train_partition,val_partition,test_partition])

    train_partition_preprocessed=resize_and_filter_partition(train_flattened,remove_non_roi,batch_size)
    val_partition_preprocessed=resize_and_filter_partition(val_flattened,remove_non_roi,batch_size)
    test_partition_preprocessed=resize_and_filter_partition(test_flattened,remove_non_roi,batch_size)

    train_images,train_roi_labels,train_labels,train_fnames=train_partition_preprocessed
    val_images,val_roi_labels,val_labels,val_fnames=val_partition_preprocessed
    test_images,test_roi_labels,test_labels,test_fnames=test_partition_preprocessed

    width,height=train_images.shape[-1],train_images.shape[-2]


    if debug:

        fraction_neg_labels=0.7
        if problem_type=='brain_detect':
            fraction_neg_labels=0.3

        num_debug=num_images_debug
        train_images=train_images[0:num_debug]
        #train_labels=train_labels[0:num_debug]
        # simulate skewed distribution
        num_neg_labels=int(fraction_neg_labels*num_debug)
        num_pos_labels=num_debug-num_neg_labels
        train_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels)) # to ensure roc, auc calculations are valid
        #train_roi_labels=train_roi_labels[0:num_debug]
        # train_roi_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels))
        train_roi_labels=np.array([1]*(num_debug))
        train_fnames=train_fnames[0:num_debug]


        val_images=val_images[0:num_debug]
        # val_labels=train_labels[0:num_debug]
        val_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels))
        # val_roi_labels=val_roi_labels[0:num_debug]
        # val_roi_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels))
        val_roi_labels=np.array([1]*num_debug)
        val_fnames=val_fnames[0:num_debug]

        test_images=test_images[0:num_debug]
        # test_labels=test_labels[0:num_debug]
        test_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels))
        #test_roi_labels=test_roi_labels[0:num_debug]
        # test_roi_labels=np.concatenate(([0]*num_neg_labels,[1]*num_pos_labels))
        test_roi_labels=np.array([1]*num_debug)
        test_fnames=test_fnames[0:num_debug]



    return {'train':[train_images,train_labels,train_roi_labels,train_fnames], 'val':[val_images,val_labels,val_roi_labels,val_fnames],
            'test':[test_images,test_labels,test_roi_labels,test_fnames]}




"""
For cross validation

only train, test indices
"""
def load_datasets_cv(dataset_by_subject,remove_non_roi,batch_size,debug,dataset_partitions):
    
    num_subjects=len(dataset_by_subject[0])
    num_train=int(0.7*num_subjects)
    num_test=num_subjects-num_train

   
    subject_indices=np.arange(num_subjects)

    train_ind=dataset_partitions[0]
    test_ind=dataset_partitions[1]

    train_partition=getDatasetPartition(dataset_by_subject,train_ind)
    test_partition=getDatasetPartition(dataset_by_subject,test_ind)


    train_flattened,test_flattened=map(flattenPartition,
                                    [train_partition,test_partition])

    train_partition_preprocessed=resize_and_filter_partition(train_flattened,remove_non_roi,batch_size)
    test_partition_preprocessed=resize_and_filter_partition(test_flattened,remove_non_roi,batch_size)

    train_images,train_roi_labels,train_labels,train_fnames=train_partition_preprocessed
    test_images,test_roi_labels,test_labels,test_fnames=test_partition_preprocessed

    if debug:

        train_images=train_images[0:100]
        train_labels=np.concatenate(([0]*50,[1]*50)) # to ensure roc, auc calculations are valid
        train_roi_labels=train_roi_labels[0:100]
        train_fnames=train_fnames[0:100]


        test_images=test_images[0:100]
        test_labels=np.concatenate(([0]*50,[1]*50))
        test_roi_labels=test_roi_labels[0:100]
        test_fnames=test_fnames[0:100]

    return {'train':[train_images,train_labels,train_roi_labels,train_fnames],
            'test':[test_images,test_labels,test_roi_labels,test_fnames]}



"""
pred: tensor corresponding to the output of the Unet model
image_data:  3d tesnor (# slices, 256,256)
"""
def get_roi(sess,pred,image_data):
    # rehspae the data so the last axis corresponds to # images
    image_data=np.transpose(image_data,(1,2,0))
    imageDim = np.shape(image_data) 
    image_data = np.moveaxis(image_data, -1, 0)  
    input_data = image_data[..., np.newaxis] # Add one axis to the end

    out = sess.run(tf.nn.softmax(pred), feed_dict={x: input_data}) # Find probabilities
    _out = np.reshape(out, (imageDim[2], imageDim[0], imageDim[1], n_classes)) # Reshape to input shape
    
    mask = 1 - np.argmax(np.asarray(_out), axis=3).astype(float) # Find mask
    roi_masks = np.moveaxis(mask, 0, -1) # Bring the first dim to the last


    roi_images=image_data*roi_masks


    return roi_images




"""
image: 2d array
"""
def standardize_image(image):

    standardized_image=1.0*(image-np.mean(image))/np.std(image)

    return standardized_image


def normalize_image(image):
    normalized_image=1.0*image # convert to float
    normalized_image=normalized_image/np.amax(normalized_image)

    return normalized_image

"""
masked_image: image where the non-ROI has been zeroed out
    (height,width)
    Assumes that each image along the channel is the same
    each pixel has non-negative values

roi_mask: 2d array (height, width)

        corresponds to the roi region in the masked image

 standardizes only the pixels falling inside ROI region (this includes any 0 pixels inside that region)

"""
def standardize_only_over_roi(masked_image,roi_mask):

    height,width=masked_image.shape
    total_num_pixels=height*width
    roi_ind=np.where(roi_mask>0) # assumes images have only positive values

    if len(roi_ind[0])==0:
        return masked_image.copy()

    # i.e., full image
    if len(roi_ind[0])==total_num_pixels:
        mean_pix=np.mean(masked_image)
        std_pix=np.std(masked_image)
    else:
        mean_pix=np.mean(masked_image[roi_ind[0],roi_ind[1]])
        std_pix=np.std(masked_image[roi_ind[0],roi_ind[1]])

    standardized_image=masked_image.copy()*1.0 # float operations
    standardized_image[roi_ind[0],roi_ind[1]]-=mean_pix
    standardized_image[roi_ind[0],roi_ind[1]]*=1.0/std_pix


    return standardized_image


"""
image: 2d array some non-zero data
"""
def standardize_image(image):
    standardized_image=image*1.0
    standardized_image-=np.mean(standardized_image)
    standardized_image*=(1.0/np.std(image))

    return standardized_image

"""
image has been duplicated over the channel dimension

masked_image_replicated: (height,width,channels)
"""
def standardize_only_over_roi_channel_replicated_image(masked_image_replicated):
    mask_image=masked_image_replicated[:,:,0]


    return standardized_image




"""

images: 3d array (# images, width, height)
        corresponds to masked images i.e., only ROI is present
labels: 1d array (# images)

Computes something analogous to confusion matrix

2d array: size 2x2
Row: brain present or not
    index 0: corresponds to images with no brain
    index 1: "" with brain

Column: quality label
    index 0: good
    index 1: bad
"""
def get_statistics_on_masked_images(images,labels):
    stats=np.zeros((2,2))

    for image_index in range(len(images)):

        num_roi_pix=np.where(images[image_index]>0)
        if len(num_roi_pix[0])==0:
            stats[0][labels[image_index]]+=1
        else:
            stats[1][labels[image_index]]+=1
    return stats


"""
imges: 3d (# slices, height, width)
"""
def get_max_dims(images):
    max_height=-1
    max_width=-1

    for image in images:

        roi_ind=np.where(image>0)

        if len(roi_ind[0])==0:
            continue



        im_width=np.amax(roi_ind[0])-np.amin(roi_ind[0])
        im_height=np.amax(roi_ind[1])-np.amin(roi_ind[1])

        max_height=max(max_height,im_height)
        max_width=max(max_width,im_width)

    return max_height, max_width



def is_non_empty_image(image):
    non_zero_ind=np.where(image>0)
    if len(non_zero_ind[0])==0:
        return False
    return True

"""
bb_images: corrected segmentations 

Filters corresponding images and labels based on bb_images.
"""
def get_non_empty_dataset(original_images,images_seg,images_bb,masks_bb,labels,fnames):

    flag_non_empty_images=np.array(map(lambda x: np.any(x!=0),images_bb))
    ind_non_empty=np.where(flag_non_empty_images==True)[0]

    if len(ind_non_empty)==0:
        return np.array([]),np.array([]),np.array([]),np.array([])

    non_empty_original_images=original_images[ind_non_empty]
    non_empty_images_seg=images_seg[ind_non_empty]
    non_empty_images_bb=images_bb[ind_non_empty]
    non_empty_masks_bb=masks_bb[ind_non_empty]
    non_empty_labels=labels[ind_non_empty]
    non_empty_fnames=fnames[ind_non_empty]

    return non_empty_original_images,non_empty_images_seg,non_empty_images_bb,non_empty_masks_bb,non_empty_labels,non_empty_fnames


"""
image: 2d array (width, height) 
    assumes that there is at least 1 roi pixel
    pixel values are nonnegative
center_row: int center row coordinate of the roi in the image
center_col: int "" col ""
"""
def center_roi(image,center_row,center_col):
    height,width=image.shape
    centered_image=np.zeros((height,width))

    image_center_row=int(round(height*1.0)/2)
    image_center_col=int(round(width*1.0)/2)

    # offsets bet. the roi center and the center of the image, used to translate the roi
    center_row_offset=center_row-image_center_row
    center_col_offset=center_col-image_center_col

    # translate the roi indices based on the above offset
    roi_ind=np.where(image>0)
    row_centered_roi_ind=roi_ind[0]-center_row_offset
    col_centered_roi_ind=roi_ind[1]-center_col_offset

    # generate the centered image
    num_roi_pix=len(roi_ind[0])
    for i in range(num_roi_pix):
        centered_image[row_centered_roi_ind[i]]=image[roi_ind[0][i],roi_ind[1][i]]

    return centered_image

"""
image: contains at least 1 roi pixel
        pixel values are nonnegative

outputs a 2d array corresponding to the roi cropped out of the image with the tightest bounding box
"""
def crop_roi(image):

    width,height=image.shape
    num_pix=width*height

    roi_ind_row,roi_ind_col=np.where(image>0)

    # deal with memory issues
    if len(roi_ind_row)==num_pix:
        return image.copy()

    min_row=np.amin(roi_ind_row)
    max_row=np.amax(roi_ind_row)
    min_col=np.amin(roi_ind_col)
    max_col=np.amax(roi_ind_col)

    cropped_roi_image=image[min_row:max_row+1,min_col:max_col+1]

    return cropped_roi_image

"""
image: assumes at least 1 pix is ROI
height: int to resize to >0
width: int to resize to >0
"""
def rescale_roi(image,height,width):
    cropped_image=crop_roi(image)
    pil_img=Image.fromarray(cropped_image)
    rescaled_img=pil_img.resize((width,height),resample=PIL.Image.BILINEAR)
    rescaled_img=np.array(rescaled_img,dtype='float64')

    return rescaled_img


"""
image: 2d array 
new_height: int to resize to >0
new_width: int to resize to >0

In cases where the crop is asymmetric i.e., there is no centered rectangular crop, 
returns the lower right crop rel. to center pixel.
"""
def crop_image(image,new_height,new_width):
    #pil_img=Image.fromarray(image)

    # coordinates corresponding to the new upper left, bottom right corners of the cropped image
    orig_height=image.shape[0]
    orig_width=image.shape[1]

    # round up -- this informs the decision on the parity to subtract instead of add 
    # in asymmetric cases
    delta_row=int((new_height*1.0/2))
    delta_col=int((new_width*1.0/2))

    center_row=int((orig_height*1.0/2))
    center_col=int((orig_width*1.0/2))
    
    new_upper_left_row=center_row-delta_row
    new_upper_left_col=center_col-delta_col

    new_lower_right_row=center_row+delta_row
    new_lower_right_col=center_col+delta_col

    # deal with asymmetric cases
    orig_height_parity=orig_height%2
    new_height_parity=new_height%2

    orig_width_parity=orig_width%2
    new_width_parity=new_width%2

    """
    if orig_height_parity==0:
        if orig_height_parity==new_height_parity: # asymmetry
            new_lower_right_row-=1

    if orig_width_parity==0:
        if orig_width_parity==new_width_parity: # asymmetry
            new_lower_right_col-=1
    """


    if new_height_parity==0:
        new_lower_right_row-=1
    if new_width_parity==0:
        new_lower_right_col-=1
    crop_corner_coords=(new_upper_left_row,new_upper_left_col,new_lower_right_row,new_lower_right_col)
    cropped_image=image[new_upper_left_row:new_lower_right_row+1,new_upper_left_col:new_lower_right_col+1]
    #cropped_image=pil_img.crop(crop_corner_coords)

    #cropped_image=np.array(cropped_image)
    return cropped_image






def get_aug_image(image):
    img_gen=keras.preprocessing.image.ImageDataGenerator(fill_mode='constant',cval=0)
    transform_parameters={}
    transform_parameters['theta']=np.random.uniform(low=0,high=360)
    transform_parameters['tx']=np.random.uniform(low=-20,high=20)
    transform_parameters['ty']=np.random.uniform(low=-20,high=20)
    transform_parameters['flip_horizontal']=0
    if np.random.uniform()>0.5:
        transform_parameters['flip_horizontal']=1
    transform_parameters['flip_vertical']=0
    if np.random.uniform>0.5:
        transform_parameters['flip_vertical']=1
    """
    transform_parameters['channel_shift_intensity']=0
    transform_parameters['brightness']=0
    """
    aug_image=img_gen.apply_transform(image,transform_parameters)

    return aug_image


"""
Assumes at least 1 certain

datasetPartition: 1d array
    -images
    -quality_labels {-1,0,1}
    -roi_labels
    -fnames

Outputs the correspond 1d array where the uncertain are filtered
"""
def filter_uncertain_partition(datasetPartition):
    images,quality_labels,roi_labels,fnames=datasetPartition

    ind_certain=np.where(quality_labels!=-1)[0]

  
    filteredPartition=[images[ind_certain],quality_labels[ind_certain],roi_labels[ind_certain],fnames[ind_certain]]

    return filteredPartition



"""
num_bad: int
num_total: int
"""
def compute_class_weights(num_bad,num_total):
    num_good=num_total-num_bad

    freq_bad=num_bad*1.0/num_total
    freq_good=num_good*1.0/num_total

    weight_bad=0.5/freq_bad
    weight_good=0.5/freq_good

    normalized_weight_bad=weight_bad/weight_good
    normalized_weight_good=1

    return normalized_weight_bad,normalized_weight_good


"""
"""
def get_stats(labels):
    num_images=len(labels)
    num_bad=len(np.where(labels==1)[0])
    frac_bad=num_bad*1.0/len(labels)

    num_uncertain=len(np.where(labels==-1)[0])
    frac_uncertain=num_uncertain*1.0/len(labels)
    return num_images,frac_bad, frac_uncertain




"""
dataset_size: int >0

fraction: float (0.0, 1.0]

Returns at least 1 index for sample point
"""
def get_sample_indices(dataset_size,fraction):
    sample_size=int(fraction*dataset_size)
    if sample_size==0:
        sample_size=1 # return at least 1 sample point
    sample_ind=np.random.choice(np.arange(dataset_size),size=sample_size,replace=False)

    return sample_ind