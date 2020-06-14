import os
import dataset as ds
import numpy as np
import shutil
import nibabel as nib
import analyze_data as ad

# set the source of the dataset
source_path=os.path.join('../../../../../d/datasets_for_iqa',
						'iqa_data_source/') # directory containing all the subjects

dataset_path=os.path.join(source_path,'reorganized_combined_dataset')

"""
source_path=os.path.join('test','test_data')
dataset_path=os.path.join(source_path,'reorganized_sample_data')
"""

all_subject_data=[]
all_subject_source=[]
for subject_source_path in os.listdir(dataset_path):
	complete_subject_source_path=os.path.join(dataset_path,subject_source_path)
	all_subject_data.append(ds.get_subject_data(complete_subject_source_path))
	all_subject_source.append(subject_source_path)

all_subject_data=np.array(all_subject_data)
all_subject_source=np.array(all_subject_source)


# allocate train/val/test

dataset_mode='complete' # debug, tune_lr, complete
generate_new_partitions=True # use this for reallocating the subjects on the train/val/test


# randomly assign subjects to train/val/test
if generate_new_partitions:
	num_subjects=len(all_subject_data)
	subject_indices=np.arange(num_subjects)
	np.random.shuffle(subject_indices)
	num_train=int(0.6*num_subjects)
	num_val=max(1,int(0.2*num_subjects))
	num_test=num_subjects-num_train-num_val
	train_ind=subject_indices[:num_train]
	val_ind=subject_indices[num_train:num_train+num_val]
	test_ind=subject_indices[num_train+num_val:]
else:
	train_ind,val_ind,test_ind=np.load(os.path.join(source_path,'dataset_partition',
									dataset_mode,'subject_partition.npy'))






# store train/val/test partition arrays here
data_partition_directory=os.path.join(source_path,'dataset_partition')
if not os.path.isdir(data_partition_directory):
	os.mkdir(data_partition_directory)

data_partition_directory=os.path.join(data_partition_directory,dataset_mode)
if os.path.isdir(data_partition_directory):
	shutil.rmtree(data_partition_directory)
os.mkdir(data_partition_directory)


data_partition_images_directory=os.mkdir(os.path.join(data_partition_directory,'images'))
data_partition_analysis_dir=os.mkdir(os.path.join(data_partition_directory,'analysis'))

# export data
is_task_iqa=True
data_partitions={'train':train_ind,'val':val_ind,'test':test_ind}
np.save(os.path.join(data_partition_directory,'subject_partition.npy'),
		np.array([train_ind,val_ind,test_ind]))

for data_partition_key,data_partition_ind, in data_partitions.items():
	print("preparing and analyzing partition %s"%data_partition_key)
	subject_data_partition=all_subject_data[data_partition_ind]
	subject_source_partition=all_subject_source[data_partition_ind]

	# aggregate and prepare data 

	# filtering
	data_partition_stacks=ds.get_all_stack_data(subject_data_partition)
	data_partition_brain_stacks=filter(lambda x: x.is_brain_stack(),data_partition_stacks)

	data_partition_slices=ds.get_all_slice_data(data_partition_brain_stacks)
	if is_task_iqa:
		num_slices_prior_filter=len(data_partition_slices)
		data_partition_slices=filter(lambda x: x.is_valid_for_iqa(),data_partition_slices)

	good_slices=filter(lambda x: x.is_good_slice(), data_partition_slices)
	bad_slices=filter(lambda x: x.is_bad_slice(), data_partition_slices)

	good_slice_images=map(lambda x: ds.get_image_array(dataset_path,
									x.subject_folder_name,x.stack_folder_name,x.dicom_path),good_slices)
	bad_slice_images=map(lambda x: ds.get_image_array(dataset_path,
									x.subject_folder_name,x.stack_folder_name,x.dicom_path),bad_slices)
	good_slice_images=np.array(good_slice_images)
	bad_slice_images=np.array(bad_slice_images)


	# visualize the slices for debugging
	"""
	good_slice_images_nii=nib.nifti1.Nifti1Image(good_slice_images,None)
	bad_slice_images_nii=nib.nifti1.Nifti1Image(bad_slice_images,None)

	nib.save(good_slice_images_nii,os.path.join(data_partition_images_directory,
									'%s_good_slice_images.nii.gz'%data_partition_key))
	nib.save(bad_slice_images_nii,os.path.join(data_partition_images_directory,
									'%s_bad_slice_images.nii'%data_partition_key))
	"""

	# i.e., use smaller datasets for debugging/tuning
	if dataset_mode=='debug':
		num_samples=10
		num_good=int(0.7*num_samples)
		num_bad=num_samples-num_good

	elif dataset_mode=='tune_lr':
		num_samples=200
		num_good=int(0.8*num_samples)
		num_bad=num_samples-num_good

	else:
		num_good=len(good_slices)
		num_bad=len(bad_slices)

	sample_good_slices=np.random.choice(good_slices,size=num_good,replace=False)
	sample_bad_slices=np.random.choice(bad_slices,size=num_bad,replace=False)

	assert len(np.unique(sample_good_slices))==num_good
	assert len(np.unique(sample_bad_slices))==num_bad

	data_partition_slices=np.concatenate((sample_good_slices,sample_bad_slices)) 

	print("stats pre/post filter")
	print("# stacks: %d, %d" %(len(data_partition_stacks),len(data_partition_brain_stacks)))
	print("# slices: %d, %d"%(num_slices_prior_filter,len(data_partition_slices)))


	ds.export_data_partition(data_partition_directory,
									data_partition_key,
									data_partition_slices)

	
	num_stacks=len(data_partition_brain_stacks)
	num_bad_stacks=len(filter(lambda x: x.is_contaminated_stack(),data_partition_stacks))
	frac_bad_stacks=num_bad_stacks*1.0/num_stacks
	print(" # stacks %d" %num_stacks)
	print("%% bad stacks %.2f" %frac_bad_stacks)

	num_slices=len(data_partition_slices)
	num_bad_slices=len(filter(lambda x: x.is_bad_slice(), data_partition_slices))
	frac_bad_slices=num_bad_slices*1.0/num_slices
	print(" # slices %d" %num_slices)
	print("%% bad slices %.2f" %frac_bad_slices)
	
	# for the training, compute class weights used in cost sensitive learning
	if data_partition_key=='train':
		weight_bad,weight_good=ds.compute_class_weights(num_bad_slices,num_slices)
		print("class weights: ", weight_bad, weight_good)
		np.save(os.path.join(data_partition_directory,'class_weight_bad_to_good.npy'),
				np.array([weight_bad,weight_good]))

