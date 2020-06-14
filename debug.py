import cv2
import numpy as np
import dataset as ds
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import os
import dataset as ds
from sklearn.model_selection import KFold
import pickle
import scipy.io
import train_helpers as th
import skimage.io
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from shutil import copyfile
import keras
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.activations import relu 
from keras.initializers import he_normal
from keras.preprocessing.image import ImageDataGenerator, array_to_img
from tensorflow.python.client import device_lib
import dataset as ds
import os
import numpy as np
import sys
import cv2
import csv
import time
import shutil
import glob




# visualizing saliency maps
sample_smap_data=scipy.io.loadmat(os.path.join('test/test_data/',
								'case1_stack_10_CASE1_1.MR.0011.0026.2017.10.19.16.34.59.646659.29247675.mat'))
smap=sample_smap_data['smap']
image=sample_smap_data['image']

fig, ax=plt.subplots(1,3)
ax[0].imshow(image[:,:,0],cmap='gray')
colormap=ax[1].imshow(smap,vmin=0,vmax=1,cmap=plt.get_cmap('RdYlGn_r'))
ax[2].imshow(image[:,:,0],cmap='gray')
ax[2].imshow(smap,vmin=0,vmax=1,alpha=0.3,cmap=plt.get_cmap('RdYlGn_r'))
plt.colorbar(colormap)
plt.savefig(os.path.join('test/test_data/sample_smap'))


# export the labels file and diom names into the newly organized folder
curr_dir=os.getcwd()
os.chdir("../../../../../d/datasets_for_iqa/original_iqa_dataset/")
path_new_folder="dicoms/reorganized_combined_dataset"
path_old_folder="only_images/combined_dataset_sans_images"
for subject_folder in os.listdir(path_old_folder):
	subject_folder_path=os.path.join(path_old_folder,subject_folder)
	for stack_folder in os.listdir(subject_folder_path):
		stack_folder_path=os.path.join(subject_folder_path,stack_folder)
		if not os.path.isdir(stack_folder_path):
			continue
		# copy over the labels, dicom names to the new folder
		tmp=stack_folder.index("_")
		stack_index=stack_folder[tmp+1:]
		labels_file=glob.glob(os.path.join(stack_folder_path,"*.csv"))[0]
		dest_path=os.path.join(path_new_folder,subject_folder,'stack_%s'%stack_index,'labels.csv')
		print(labels_file,dest_path)
		shutil.copyfile(labels_file,dest_path)
		shutil.copyfile(os.path.join(stack_folder_path,'stack_dicom_names.npy'),
						os.path.join(path_new_folder,subject_folder,'stack_%s'%stack_index,'stack_dicom_names.npy'))

print("finished exporting labels and dicom files")
os.chdir(curr_dir)
# example stack data

"""
for slice_index in range(5):
	color='r' if slice_quality[slice_index]==1 else 'b'
	plt.bar(x_coord,height,width,bottom=bottom,color=color)
	bottom+=height


plt.xlabel("stack index")
plt.ylabel("slice index")
plt.xlim(left=0,right=num_stacks+1)
plt.xticks(np.arange(num_stacks))
plt.yticks(np.arange(10))
plt.savefig('sample_quality_by_slice')
plt.close()
"""



height=1
width=0.2
bottom=0


# generate random stack data
data_by_stack=[]
num_stacks=2
max_num_slices=20
for stack_index in range(num_stacks):
	num_slices=np.random.randint(low=5,high=max_num_slices)
	num_slices_no_roi=np.random.randint(low=1,high=3)
	slice_indices=np.arange(num_slices)
	slice_indices_no_roi=np.random.choice(slice_indices,size=num_slices_no_roi,replace=False)
	stack_data=[]
	for slice_index in slice_indices:
		roi_label=1
		if slice_index in slice_indices_no_roi:
			roi_label=0
		quality_label=np.random.randint(low=0,high=2)
		stack_data.append(ds.SliceData(roi_label,quality_label))

	data_by_stack.append(stack_data)


"""
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
for stack_index in range(num_stacks):
	x_coord=stack_index
	bottom=0
	stack_data=data_by_stack[stack_index]
	# create a bar for this stack
	for slice_index in range(num_slices):
		if stack_data[slice_index].roi_label==1
			color='r' if stack_data[slice_index]==1 else 'b'
		ax.bar(x_coord,height,width,bottom=bottom,color=color,align='center',
			edgecolor='black')
		bottom+=height


ax.set_xlabel("stack index")
ax.set_ylabel("slice index")
xleft=0-1
ax.set_xlim(left=xleft,right=num_stacks+1)
ax.set_xticks(np.arange(num_stacks))
ylims=ax.get_ylim()
max_num_slices=int(ylims[1])+1
ax.set_yticks(np.arange(max_num_slices)+0.5)
ax.set_yticklabels(np.arange(max_num_slices))
subject_name='Sample subject'
ax.set_title("Stack distribution for subject %s" %(subject_name))
fig.savefig('sample_quality_by_slice')
"""

fig=plt.figure()
ax=fig.add_subplot(1,1,1)

for stack_index,stack_data in enumerate(data_by_stack):
    # create a bar for this stack
    for slice_index,slice_data in enumerate(stack_data):
        color='gray'
        if slice_data.roi_label==1:
            color='r' if slice_data.quality_label==1 else 'b'
        ax.bar(stack_index,height=1,width=0.2,bottom=slice_index,color=color,align='center',
            edgecolor='black')


ax.set_xlabel("stack index")
ax.set_ylabel("slice index")
xleft=0-1
ax.set_xlim(left=xleft,right=num_stacks+1)
ax.set_xticks(np.arange(num_stacks))
ylims=ax.get_ylim()
max_num_slices=int(ylims[1])+1
ax.set_yticks(np.arange(max_num_slices)+0.5)
ax.set_yticklabels(np.arange(max_num_slices))
subject_name='Sample subject'
ax.set_title("Stack distribution for subject %s" %(subject_name))

fig.savefig('sample_quality_by_slice')

print("finished generating sample_quality_by_slice plot")





data_source_prefix_dir=os.path.join('../../../../../d/datasets_for_iqa',
									'original_iqa_dataset','only_images',
									'combined_dataset_sans_images')
"""
for subject_data_path in os.listdir(data_source_prefix_dir):
	ds.save_dicoms_by_stack(os.path.join(data_source_prefix_dir,subject_data_path))

print("finished organizing dicoms by stack")
"""



data_source_dir=os.path.join(data_source_prefix_dir,'sample_data','HASTE_brain (5)')
subject_data=ds.load_subject_data(data_source_dir)



print("finished loading subject data")


# load single subject data


"""

case_order_path=os.path.join(data_source_prefix_dir,'dataset_split','case_order.npy')
case_order=np.load(case_order_path)

data_source_dir=os.path.join(data_source_prefix_dir,'original_iqa_dataset',
	'only_images','combined_dataset_sans_images')



dataset_by_subject=ds.load_all_subjects_data(data_source_dir,case_order_path)
print("finished loading all subjects data")






train_val_test_splits_path=os.path.join(data_source_prefix_dir,'dataset_split','best_train_val_test_split.npy')
train_ind,val_ind,test_ind=np.load(train_val_test_splits_path)
dataset_partitions=[train_ind,val_ind,test_ind]

datasets=ds.load_datasets(dataset_by_subject,remove_non_roi=1,batch_size=1,debug=0,
    num_images_debug=10,dataset_partitions=dataset_partitions)

# filter data corresponding to uncertain categories
train_partition=ds.filter_uncertain_partition(datasets['train'])
val_partition=ds.filter_uncertain_partition(datasets['val'])
test_partition=ds.filter_uncertain_partition(datasets['test'])


updated_datasets={'train':train_partition,'val':val_partition,'test':test_partition}
"""

"""
sample_im=np.load('sample_im.npy')
sample_im=sample_im.astype('float')
sample_im*=255./np.amax(sample_im)
sample_im=np.reshape(sample_im,(256,256,1))
converted_im=array_to_img(sample_im)
converted_im.save('sample_im.jpg')


# 'ty': -20, 'tx':20, 'theta': [0,360)
# zoom params: amount to zoom in by: 0.75, 0.75
				# amount to zoom out by: 

# brightness: min:0.25 max: 1.25
# channel_shift_intensity: min-20, max:20 (addititive) 
fill_mode='nearest' # constant, nearest, reflect, wrap
cval=0
img_data_generator=ImageDataGenerator(samplewise_center=True,samplewise_std_normalization=True,
	fill_mode=fill_mode,cval=cval,preprocessing_function=ds.add_noise)
transformed_image=img_data_generator.apply_transform(sample_im,{'shear':-5})
transformed_image=img_data_generator.standardize(transformed_image)
converted_im=array_to_img(transformed_image)
converted_im.save('transformed_image.jpg')

print("finished augmenting image")
"""
"""
# generate a file with all the labels for each pair
data_source_dir=os.path.join('../../../../../d/datasets_for_iqa/nar_iqa_data')
dataset_types=['debug','tune_lr','full']
data_types=['train','val','test']

for dataset_type in dataset_types:
	for data_type in data_types:
		curr_folder=os.path.join(data_source_dir,dataset_type,data_type)
		labels=[]

		for pair_file in os.listdir(curr_folder):
			if not os.path.isdir(os.path.join(curr_folder,pair_file)):
				continue
			pair_folder=os.path.join(curr_folder,pair_file)

			eval_data=scipy.io.loadmat(os.path.join(pair_folder,'eval'))
			labels.append(eval_data['label'][0][0])

		labels=np.array(labels)
		np.save(os.path.join(curr_folder,'labels.npy'),labels)

print("finished generating labels file")
"""
"""
filter_by_image_quality=0
# organize data into directories
main_dataset_dir=os.path.join(data_source_prefix_dir,'nar_iqa_full')
if os.path.isdir(main_dataset_dir):
	shutil.rmtree(main_dataset_dir)
os.mkdir(main_dataset_dir)

dataset_types=datasets.keys()
for dataset_type in dataset_types:
	print("exporting images for %s " %dataset_type)
	dataset_dir=os.path.join(main_dataset_dir,dataset_type)
	if os.path.isdir(dataset_dir):
		shutil.rmtree(dataset_dir)
	os.mkdir(dataset_dir)

	dataset_partition=updated_datasets[dataset_type]
	images,iqa_labels,roi_labels,fnames=dataset_partition
	print("unique labels ", np.unique(iqa_labels))
	ds.save_ref_eval_pairs_single_directory(
		dataset_dir,images,iqa_labels,fnames,filter_by_image_quality)
"""
data_source_prefix_dir=os.path.join('../../../../../d/datasets_for_iqa')
main_dataset_dir=os.path.join(data_source_prefix_dir,'nar_iqa_data','tune_lr')
# dataset stats on the ref, eval pairs

# prepare the labels file -- this is in the order that the data is loaded
# by the generators and this has also been verified
dataset_types=['train','val','test']
for dataset_type in dataset_types:
	labels=[]
	dataset_dir=os.path.join(main_dataset_dir,dataset_type)
	num_files=len(os.listdir(dataset_dir))
	for pair_index in np.arange(num_files):
		eval_data=scipy.io.loadmat(os.path.join(dataset_dir,'pair_%d'%pair_index,'eval'))
		labels.append(eval_data['label'])

	labels=np.array(labels)
	print("distribution for dataset %s is %.4f" \
		%(dataset_type,np.sum(labels)*1.0/len(labels)))
	print(np.unique(labels))
	np.save(os.path.join(main_dataset_dir,'%s_labels.npy' %dataset_type),labels)

	# median freq rebalancing
	weight_bad,weight_good=ds.compute_class_weights(labels)
	class_weight={1: weight_bad, 0: weight_good}
	print(class_weight)
	np.save(os.path.join(main_dataset_dir,'class_weight_bad_to_good'),
		np.array([weight_bad,weight_good]))




print("finished exporting images to directories")


ans=1+0.92*0.3* \
 	(1+0.92*0.3* \
 		(1+0.92*0.3+0.08*0.7) \
 	+0.08*0.7*(1+0.92*0.3+0.08*0.7)) \
 +0.08*0.7*(1+0.92*0.3*(1+0.92*0.3+0.08*0.7) \
 			+0.08*0.7*(1+0.92*0.3+0.08*0.7))
# solve this recursion
num_steps=5
frac_diagnostic=0.92
acc_diagnostic=0.7
acc_non_diagnostic=0.7
num_initial_slices_to_acquire=27

def get_num_slices_to_acquire_helper(num_steps):
	if num_steps==0:
		return 1+frac_diagnostic*(1-acc_diagnostic) + \
						(1-frac_diagnostic)*acc_non_diagnostic
	else:
		return 1+frac_diagnostic*(1-acc_diagnostic)*get_num_slices_to_acquire_helper(num_steps-1) + \
						(1-frac_diagnostic)*acc_non_diagnostic*get_num_slices_to_acquire_helper(num_steps-1)

def get_num_slices_to_acquire():
	num_slices_to_acquire=1+frac_diagnostic*(1-acc_diagnostic)*get_num_slices_to_acquire_helper(num_steps-1) + \
						(1-frac_diagnostic)*acc_non_diagnostic*get_num_slices_to_acquire_helper(num_steps-1)
	num_good_slices=int(frac_diagnostic*num_initial_slices_to_acquire)
	num_bad_slices=num_initial_slices_to_acquire-num_good_slices
	num_slices_to_acquire=num_initial_slices_to_acquire+num_good_slices*(1-acc_diagnostic)*num_slices_to_acquire+\
		num_bad_slices*acc_non_diagnostic*num_slices_to_acquire
	# num_slices_to_acquire=25+25*0.3*get_num_slices_to_acquire_helper(num_steps)
	return num_slices_to_acquire




num_slices_to_acquire=get_num_slices_to_acquire_helper(10)
print(" number of slices to acquire %d " %num_slices_to_acquire)





data_source_dir=os.path.join('../../../../../d/exp_results/image_quality/images_labeled_dataset',
				'full_dataset','transfer_vgg_13','results_lr_0.000025','best')


# load model
t_start=time.time()

t_end=time.time()
print("loading model took ", t_end-t_start)

test_preds=[]
test_labels=[]
with open(os.path.join(data_source_dir,'test_preds.csv')) as csvfile:
	csvreader=csv.reader(csvfile)
	for row in csvreader:
		test_preds.append(float(row[0]))
with open(os.path.join(data_source_dir,'test_labels.csv')) as csvfile:
	csvreader=csv.reader(csvfile)
	for row in csvreader:
		test_labels.append(int(float(row[0])))

test_preds=np.array(test_preds)
test_labels=np.array(test_labels)
print("loaded test_preds and test_labels")


"""
# used for sorting array from largest to smallest
"""
def greater(a,b):
    if cv2.contourArea(a) < cv2.contourArea(b):
        return 1
    return -1

# determine the max. size radius among the brain masks
data_source_dir=os.path.join('../../../../../d/exp_results/image_quality/images_labeled_dataset/final_brain_mask_exp/',
	'analyzing_segmentation_quality')
image_data=scipy.io.loadmat(os.path.join(data_source_dir,'images'))
train_data=image_data['train']
train_bb_seg=train_data[1]
val_data=image_data['val']
val_bb_seg=val_data[1]
test_data=image_data['test']
test_bb_seg=test_data[1]

brain_images=np.concatenate((train_bb_seg,val_bb_seg,test_bb_seg))

max_radii=-1
example_image_max_radii=np.zeros((256,256))
for brain_image in brain_images:
	ind_non_zero=np.where(brain_image>0)
	sample_mask_original=np.zeros((256,256))
	sample_mask_original[ind_non_zero[0],ind_non_zero[1]]=1
	# sample calculation of the radius
	sample_mask=sample_mask_original.copy()
	# rescale mask for cv2 calculations
	sample_mask=sample_mask.astype('uint8')*255
	_,contours,_=cv2.findContours(sample_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
	# assert len(contours)==1
	# only a single contour
	sorted_contours=sorted(contours,cmp=greater)
	_,radius=cv2.minEnclosingCircle(sorted_contours[0])
	radius=int(radius)
	if radius>max_radii:
		max_radii=max(radius,max_radii)
		example_image_max_radii=brain_image








data_source_main_dir=os.path.join('../../../../../d/datasets_for_iqa')
data_source_dir=os.path.join(data_source_main_dir,'only_images/combined_dataset_sans_images')
dataset_by_subject=ds.load_all_subjects_data(data_source_dir)
num_subjects=len(dataset_by_subject[0])
num_train=int(0.6*num_subjects)
num_val=int(0.2*num_subjects)
num_test=num_subjects-num_train-num_val
subject_indices=np.arange(num_subjects)

# randomly sample, 2 fold -- train/val/test: 20/6/6 
train_val_test_splits_path=os.path.join(data_source_main_dir,'dataset_split/best_train_val_test_split_v1.npy')
if len(train_val_test_splits_path)==0:
    np.random.shuffle(subject_indices)
    train_ind=subject_indices[0:20]
    val_ind=subject_indices[20:26]
    test_ind=subject_indices[26:]
    np.save(os.path.join(data_source_main_dir,'dataset_split/best_train_val_test_split_v1.npy'),[train_ind,val_ind,test_ind])
else:
    train_ind,val_ind,test_ind=np.load(train_val_test_splits_path)

dataset_partitions=[train_ind,val_ind,test_ind]
remove_non_roi=1
batch_size=50
debug=0
num_images_debug=-1
problem_type='iqa'
datasets=ds.load_dataset(data_source_dir,remove_non_roi=0)






font={'size':8}
matplotlib.rc('font',**font)

NUM_EPOCHS=500
CHOSEN_LR=['1e-1', '1e-2',  '1e-3','1e-4','1e-5', '1e-6']
CHOSEN_LR=['1e-1', '1e-2',  '0.0075', '0.005', '0.0025', '1e-3','1e-4','1e-5', '1e-6']
home_dir=os.path.expanduser("~")

if len(device_lib.list_local_devices())==2:
	prefix_source_dir=os.path.join(home_dir,'image_quality')
	final_source_dir=prefix_source_dir
else:
	prefix_source_dir=os.path.join(home_dir,'..','..','mnt','d',
		'exp_results/image_quality/images_labeled_dataset/full_dataset')
	final_source_dir=os.path.join(prefix_source_dir,'no_transfer_resnet18','tune_lr','with_data_augs')


# iterate over various learning rates
"""
plt.figure()
epoch_indices=[]
for lr_res_folder in os.listdir(final_source_dir):
	
	if not lr_res_folder.startswith('results_lr'):
		continue
	start_index=lr_res_folder.index('_')
	lr=lr_res_folder[start_index+1:]
	if lr[3:] not in CHOSEN_LR:
		continue

	train_loss=np.load(os.path.join(final_source_dir,
		lr_res_folder,'final/train_loss.npy'))
	if len(epoch_indices)==0:
		epoch_indices=np.arange(len(train_loss))
	plt.plot(epoch_indices,train_loss,label=lr)

# format the figure
plt.legend()
plt.xlabel('Epoch index')
plt.ylabel('Training loss')
plt.title('Comparing Learning Rates')
plt.savefig(os.path.join(final_source_dir,'loss_lr.png'))
plt.close()
"""

# combine all loss plots into 1 figure
fig=plt.figure()
#fig,ax=plt.subplots(3,3,sharex='col',sharey='row')
i=1
#for lr_res_folder in os.listdir(final_source_dir):
for lr in CHOSEN_LR:
	lr_res_folder='results_lr_'+lr
	"""
	if not lr_res_folder.startswith('results_lr'):
		continue
	start_index=lr_res_folder.index('_')
	lr=lr_res_folder[start_index+1:]

	if not lr[3:] in CHOSEN_LR:
		continue
	"""
	train_loss=np.load(os.path.join(final_source_dir,
		lr_res_folder,'final/train_loss.npy'))[:NUM_EPOCHS]
	
	epoch_indices=np.arange(len(train_loss))

	ax=fig.add_subplot(3,3,i)
	ax.plot(epoch_indices,train_loss,label=lr)
	ax.set_title('LR= %s' %lr)
	ax.set_xlabel('Epoch index')
	ax.set_ylabel('Loss')
	"""
	if i-1 in [6,7,8]:
		ax.set_xlabel('Epoch index')
		ax.set_xticks(np.linspace(0,len(train_loss),num=100))
	else:
		# ax.set_xticks([])
		ax.tick_params(axis='x',which='both',bottom=False,top=False,labelbottom=False)
		ax.set_xlabel('')

	if i-1 in [0,3,6]:
		ax.set_ylabel('Training loss')
	else:
		ax.tick_params(axis='y',which='both',left=False,right=False,labelleft=False)
		ax.set_ylabel('')
	"""
	i+=1

plt.tight_layout()
plt.savefig(os.path.join(final_source_dir,'loss_by_lr.png'))
plt.close()



"""
prefix_source_dir=os.path.join(home_dir,'..','..','mnt','d',
	'exp_results/image_quality/images_labeled_dataset/variable_architectures/transfer_resnet50/class_weight/100_percent_data/FFOV/'+
	'results_lr_1e-4/results_ig_black_vs_black_preprocess/test_attrib/')


image_types=['tp','fp']
baseline_types=['black','black_preprocess']

for image_type in image_types:
	for baseline_type in baseline_types:
		data_source_dir=os.path.join(prefix_source_dir,image_type,'im_0',baseline_type)

		attributions_data=scipy.io.loadmat(os.path.join(data_source_dir,'attributions.mat'))

		attributions=attributions_data['attributions']
		img=attributions_data['image_unscaled']
		img_scaled=(img*1.0/np.amax(img))*255

		plot_attribution_distribution('attributions_dist.png',attributions)

		attributions_image=pil_image(Visualize(
		              attributions, img_scaled, 
		              polarity='both',
		              clip_above_percentile=99,
		              clip_below_percentile=60,
		              overlay=True))

		attributions_image.save(os.path.join(data_source_dir,'lower_threshold_60_both_attributions.png'))
"""




home_dir=os.path.expanduser("~")
data_source_dir=os.path.join(home_dir,'..','..','mnt','d',
	'exp_results/image_quality/images_labeled_dataset/variable_architectures/transfer_resnet50/results_fold_0/final')

# code below compares various architectures and experiments

"""
modifies the figure
"""
def add_roc(figure,stats,label):
	fpr=stats['fpr'][0]
	tpr=stats['tpr'][0]
	auc=stats['auc'][0]
	ax=figure.axes[0]
	ax.plot(fpr,tpr,label=label+'_AUROC=%0.2f'%auc)
    

def initialize_plot(figure,dataset_title):
	ax=figure.add_subplot(1,1,1)
	ax.set_xlabel('False Positive Rate')
	ax.set_ylabel('True Positive Rate')
	ax.set_xlim([0.0, 1.0])
	ax.set_ylim([0.0, 1.05])
	ax.set_title(dataset_title+'_ROC curves')
	#ax.legend(loc="lower right")
	# plot references
	# ax.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
	# 

def finalize_plot(figure):
	ax=figure.add_subplot(1,1,1)
	ax.legend(loc="lower right")


source_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset/variable_architectures/')
architectures_file_paths=['transfer_resnet18','transfer_resnet34','transfer_resnet50']
architectures_file_paths=['transfer_resnet18']

# # plot comparing the ROCs
# fig_train=plt.figure()
# fig_val=plt.figure()
# fig_test=plt.figure()

# # initial formatting of the ROC plots
# initialize_plot(fig_train,'Train')
# initialize_plot(fig_val,'Val')
# initialize_plot(fig_test,'Test')

"""
# initiatlize ROC plots
figure.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
    figure.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
"""




# for architecture_fp in architectures_file_paths:

# 	"""
# 	folders=os.listdir(os.path.join(source_dir,architecture_fp))
# 	results_folder=None
# 	for folder_path in folders:
# 		if 'results_' in folder_path:
# 			results_folder=folder_path
# 		break
# 	"""
# 	results_folder='results_more_epochs'
# 	final_model_source_dir=os.path.join(source_dir,architecture_fp,results_folder,'final')


# 	train_stats=scipy.io.loadmat(os.path.join(final_model_source_dir,'train_whole_stats'))
# 	add_roc(fig_train,train_stats,architecture_fp+'_final_')

# 	val_stats=scipy.io.loadmat(os.path.join(final_model_source_dir,'val_whole_stats'))
# 	add_roc(fig_val,val_stats,architecture_fp+'_final_')

# 	test_stats=scipy.io.loadmat(os.path.join(final_model_source_dir,'test_whole_stats'))
# 	add_roc(fig_test,test_stats,architecture_fp+'_final_')

# 	# if the best model is avialable evaluate
# 	subfolders=os.listdir(os.path.join(source_dir,architecture_fp,results_folder))
	
# 	"""
# 	best_model_present=False
# 	for subfolder in subfolders:
# 		if subfolder=='best':
# 			best_model_present=True

# 	if best_model_present:
# 		best_model_source_dir=os.path.join(source_dir,architecture_fp,results_folder,'best')
# 		val_stats=scipy.io.loadmat(os.path.join(best_model_source_dir,'val_whole_stats'))
# 		add_roc(fig_val,val_stats,architecture_fp+'_best_')

# 		test_stats=scipy.io.loadmat(os.path.join(best_model_source_dir,'test_whole_stats'))
# 		add_roc(fig_test,test_stats,architecture_fp+'_best_')
# 	"""

# map(finalize_plot,[fig_train,fig_val,fig_test])

# fig_train.savefig(os.path.join(source_dir,'train_roc'))
# fig_val.savefig(os.path.join(source_dir,'val_roc'))
# fig_test.savefig(os.path.join(source_dir,'test_roc'))


# plot comparing the ROCs
# fig_train=plt.figure()
# fig_val=plt.figure()
# fig_test=plt.figure()

# # initial formatting of the ROC plots
# initialize_plot(fig_train,'Transfer-Resnet18 Train')
# initialize_plot(fig_val,'Transfer-Resnet18 Val')
# initialize_plot(fig_test,'Transfer-Resnet18 Test')

# # class weight vs no class weight, 100% data
# source_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset/variable_architectures/')

# main_source_dir=os.path.join(source_dir,'transfer_resnet18')
# model_types=['best','final']
# weight_modes=['class_weight','no_class_weight']

# for weight_mode in weight_modes:
# 	for model_type in model_types:

# 		if model_type=='final':
# 			train_stats=scipy.io.loadmat(os.path.join(main_source_dir,weight_mode,'100_percent_data','results_lr_1e-4',model_type,'train_whole_stats.mat'))
# 			add_roc(fig_train,train_stats,'%s_%s_'%(weight_mode,model_type))

# 		val_stats=scipy.io.loadmat(os.path.join(main_source_dir,weight_mode,'100_percent_data','results_lr_1e-4',model_type,'val_whole_stats.mat'))
# 		add_roc(fig_val,val_stats,'%s_%s_'%(weight_mode,model_type))

# 		test_stats=scipy.io.loadmat(os.path.join(main_source_dir,weight_mode,'100_percent_data','results_lr_1e-4',model_type,'test_whole_stats.mat'))
# 		add_roc(fig_test,test_stats,'%s_%s_'%(weight_mode,model_type))


# map(finalize_plot,[fig_train,fig_val,fig_test])

# fig_train.savefig(os.path.join(main_source_dir,'train_roc'))
# fig_val.savefig(os.path.join(main_source_dir,'val_roc'))
# fig_test.savefig(os.path.join(main_source_dir,'test_roc'))



# # compare the non-vs transfer resnets

# fig_train=plt.figure()
# fig_val=plt.figure()
# fig_test=plt.figure()

# # initial formatting of the ROC plots
# initialize_plot(fig_train,'Train')
# initialize_plot(fig_val,'Val')
# initialize_plot(fig_test,'Test')

# # class weight vs no class weight, 100% data
# source_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset/variable_architectures/')

# architectures=['resnet18','resnet34','resnet50']
# transfer_modes=['no_transfer']
# model_types=['best','final']
# for architecture in architectures:

# 	# add the transfer/no transfer results
	
# 	for transfer_mode in transfer_modes:
# 		for model_type in model_types:
# 			main_source_dir=os.path.join(source_dir,transfer_mode+'_'+architecture,'results_lr_1e-4')
# 			# back out what epoch the best model was saved around
# 			if model_type=='best':
# 				val_scores=[]
# 				with open(os.path.join(main_source_dir,'final','training.log')) as f:
# 					f=f.readlines()[1:]
# 				for line in f:
# 					val_scores.append(float(line.split(',')[-1]))
# 				best_epoch=np.argmin(val_scores)

# 			if model_type=='final':
# 				train_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'train_whole_stats.mat'))
# 				add_roc(fig_train,train_stats,'%s_%s_%s'%(architecture, model_type, transfer_mode))
			
# 			if model_type=='final':

# 				val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
# 				add_roc(fig_val,val_stats,'%s_%s_%s'%(architecture, model_type, transfer_mode))
# 			else:
# 				val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
# 				add_roc(fig_val,val_stats,'%s_%s_%s'%(architecture, model_type+'_epoch_'+str(best_epoch), transfer_mode))

# 			if model_type=='final':
# 				test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
# 				add_roc(fig_test,test_stats,'%s_%s_%s'%(architecture, model_type, transfer_mode))
# 			else:
# 				test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
# 				add_roc(fig_test,test_stats,'%s_%s_%s'%(architecture, model_type+'_epoch_'+str(best_epoch), transfer_mode))


# map(finalize_plot,[fig_train,fig_val,fig_test])

# fig_train.savefig(os.path.join(source_dir,'train_roc'))
# fig_val.savefig(os.path.join(source_dir,'val_roc'))
# fig_test.savefig(os.path.join(source_dir,'test_roc'))



# evaluate learning curve
"""
fig_train=plt.figure()
fig_val=plt.figure()
fig_test=plt.figure()

# initial formatting of the ROC plots
initialize_plot(fig_train,'Class-weighted Transfer Resnet-50 Train')
initialize_plot(fig_val,'Class-weighted Transfer Resnet-50 Val')
initialize_plot(fig_test,'Class-weighted Transfer Resnet-50 Test')

# class weight vs no class weight, 100% data
pre_source_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset/variable_architectures/transfer_resnet50/class_weight')

train_fractions=[5,50,100]
model_types=['best']
for train_fraction in train_fractions:
	source_dir=os.path.join(pre_source_dir,str(train_fraction)+'_percent_data')
	
	for model_type in model_types:
		main_source_dir=os.path.join(source_dir,'results_lr_1e-4')
		# back out what epoch the best model was saved around
		if model_type=='best':
			val_scores=[]
			with open(os.path.join(main_source_dir,'final','training.log')) as f:
				f=f.readlines()[1:]
			for line in f:
				val_scores.append(float(line.split(',')[-1]))
			best_epoch=np.argmin(val_scores)

		# only evaluated the final model on the training data
		if model_type=='final':
			train_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'train_whole_stats.mat'))
			add_roc(fig_train,train_stats,'%d_percent_train_%s'%(train_fraction, model_type))
		
		if model_type=='final':
			val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
			add_roc(fig_val,val_stats,'%d_percent_train_%s'%(train_fraction, model_type))
		else:
			val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
			add_roc(fig_val,val_stats,'%d_percent_train_%s'%(train_fraction, model_type+'_epoch_'+str(best_epoch)))

		if model_type=='final':
			test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
			add_roc(fig_test,test_stats,'%d_percent_train_%s'%(train_fraction, model_type))
		else:
			test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
			add_roc(fig_test,test_stats,'%d_percent_train_%s'%(train_fraction, model_type+'_epoch_'+str(best_epoch)))


map(finalize_plot,[fig_train,fig_val,fig_test])

fig_train.savefig(os.path.join(pre_source_dir,'train_roc'))
fig_val.savefig(os.path.join(pre_source_dir,'val_roc'))
fig_test.savefig(os.path.join(pre_source_dir,'test_roc'))
"""
    
# compare effect of brain masking
fig_train=plt.figure()
fig_val=plt.figure()
fig_test=plt.figure()

# initial formatting of the ROC plots
initialize_plot(fig_train,'Transfer Resnet-50 Train')
initialize_plot(fig_val,'Transfer Resnet-50 Val')
initialize_plot(fig_test,'Transfer Resnet-50 Test')

# class weight vs no class weight, 100% data
pre_source_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset/variable_architectures/transfer_resnet50/')

model_types=['best']
class_weight_modes=['class_weight']
brain_mask_modes=['FFOV','brain_no_rescale','brain_rescale']


for class_weight in class_weight_modes:
	for brain_mask_mode in brain_mask_modes:
		source_dir=os.path.join(pre_source_dir,class_weight,'100_percent_data',brain_mask_mode)
		
		for model_type in model_types:
			main_source_dir=os.path.join(source_dir,'results_lr_1e-4')
			# back out what epoch the best model was saved around
			if model_type=='best':
				val_scores=[]
				with open(os.path.join(main_source_dir,'final','training.log')) as f:
					f=f.readlines()[1:]
				for line in f:
					val_scores.append(float(line.split(',')[-1]))
				best_epoch=np.argmin(val_scores)

			# only evaluated the final model on the training data
			if model_type=='final':
				train_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'train_whole_stats.mat'))
				add_roc(fig_train,train_stats,'%s_%s_loss_%s'%(brain_mask_mode,class_weight, model_type))
			
			if model_type=='final':
				val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
				add_roc(fig_val,val_stats,'%s_%s_loss_%s'%(brain_mask_mode,class_weight, model_type))
			else:
				val_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'val_whole_stats.mat'))
				add_roc(fig_val,val_stats,'%s_%s_loss_%s'%(brain_mask_mode,class_weight, model_type+'_epoch_'+str(best_epoch)))

			if model_type=='final':
				test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
				add_roc(fig_test,test_stats,'%s_%s_loss_%s'%(brain_mask_mode,class_weight, model_type))
			else:
				test_stats=scipy.io.loadmat(os.path.join(main_source_dir,model_type,'test_whole_stats.mat'))
				add_roc(fig_test,test_stats,'%s_%s_loss_%s'%(brain_mask_mode,class_weight, model_type+'_epoch_'+str(best_epoch)))


map(finalize_plot,[fig_train,fig_val,fig_test])

fig_train.savefig(os.path.join(pre_source_dir,'train_roc'))
fig_val.savefig(os.path.join(pre_source_dir,'val_roc'))
fig_test.savefig(os.path.join(pre_source_dir,'test_roc'))





def get_stats(dataset_by_subject,partitions_by_subject_ind):
	datasets=ds.load_datasets(dataset_by_subject,1,1,0,0,partitions_by_subject_ind)

	# filter data corresponding to uncertain categories
	train_partition=ds.filter_uncertain_partition(datasets['train'])
	val_partition=ds.filter_uncertain_partition(datasets['val'])
	test_partition=ds.filter_uncertain_partition(datasets['test'])

	dist_by_dataset=[]
	for dataset_partition in [train_partition,val_partition,test_partition]:


		images,labels,roi_labels,fnames=dataset_partition

		num_bad=len(np.where(labels==1)[0])
		frac_bad=num_bad*1.0/len(labels)
		#print("fraction bad %.2f" % ())
		dist_by_dataset.append(frac_bad)
	return np.array(dist_by_dataset)


# load datasets

dist_train=[]
dist_val=[]
dist_test=[]



home_dir=os.path.expanduser("~")
#data_source_dir=os.path.join(home_dir,'..','..','mnt','d','datasets_for_iqa','only_images','combined_dataset_sans_images')
data_source_dir='../data/combined_dataset_sans_images'
dataset_by_subject=ds.load_all_subjects_data(data_source_dir)
num_subjects=len(dataset_by_subject[0])
subject_indices=np.arange(num_subjects)

"""
np.random.shuffle(subject_indices)
train_ind=subject_indices[0:20]
val_ind=subject_indices[20:26]
test_ind=subject_indices[26:]
"""

partitions_by_subject_ind=np.load('data/best_train_val_test_split_v1.npy')
train_val_test_stats=get_stats(dataset_by_subject,partitions_by_subject_ind) # sanity check
print(train_val_test_stats)
datasets=ds.load_datasets(dataset_by_subject,1,1,0,0,partitions_by_subject_ind)

# filter data corresponding to uncertain categories
train_partition=ds.filter_uncertain_partition(datasets['train'])
val_partition=ds.filter_uncertain_partition(datasets['val'])
test_partition=ds.filter_uncertain_partition(datasets['test'])


def compute_average_bad_fraction_given_sample_size(sample_size,num_iterations_to_average,complete_labels):
	bad_frac_samples=[]
	image_ind=np.arange(len(complete_labels))
	
	for i in range(num_iterations_to_average):
		# compute tehe avg. distribution in training set
		sample_ind=np.random.choice(image_ind,size=sample_size,replace=False)
		sample_labels=complete_labels[sample_ind]
		num_bad=len(np.where(sample_labels==1)[0])
		frac_bad=num_bad*1.0/len(sample_labels)
		bad_frac_samples.append(frac_bad)
	bad_frac_samples=np.array(bad_frac_samples)

	return np.mean(bad_frac_samples),np.var(bad_frac_samples)

# on avg, how the distribution compares by undersampling training dataset
images,labels,roi_labels,fnames=train_partition
image_ind=np.arange(len(images))
max_num_images=len(images)
sample_sizes=[10, 50, 100, 500, 1000, 2000, max_num_images]
num_iterations_to_average=50
for sample_size in sample_sizes:
	average_bad_frac,var_bad_frac=compute_average_bad_fraction_given_sample_size(sample_size,num_iterations_to_average,labels)
	print("for sample size %d: avg frac bad : %.4f (%.4f)" % (sample_size,average_bad_frac,var_bad_frac))





# train_ind=np.arange(0,20)
# val_ind=np.arange(20,26)
# test_ind=np.arange(26,32)

# train_bad,val_bad,test_bad=get_stats(dataset_by_subject,[train_ind,val_ind,test_ind])
# print(train_bad,val_bad,test_bad)
# """
# dist_train.append(train_bad)
# dist_val.append(val_bad)
# dist_test.append(test_bad)
# """
# for i in range(100):
# 	np.random.shuffle(subject_indices)
# 	train_ind=subject_indices[0:20]
# 	val_ind=subject_indices[20:26]
# 	test_ind=subject_indices[26:]
# 	all_ind=np.concatenate((train_ind,val_ind,test_ind))
# 	train_bad,val_bad,test_bad=get_stats(dataset_by_subject,[train_ind,val_ind,test_ind])

# 	if 0.08 < train_bad and train_bad<0.1:
# 		if 0.08 < val_bad and val_bad<0.1:
# 			if 0.08 < test_bad and test_bad<0.1:
# 				print(train_bad,val_bad,test_bad)
# 				np.save('data/best_train_val_test_split.npy',[train_ind,val_ind,test_ind])
# 	dist_train.append(train_bad)
# 	dist_val.append(val_bad)
# 	dist_test.append(test_bad)

# dist_train=np.array(dist_train)
# dist_val=np.array(dist_val)
# dist_test=np.array(dist_test)

# print(np.mean(dist_train),np.std(dist_train))
# print(np.mean(dist_val),np.std(dist_val))
# print(np.mean(dist_test),np.std(dist_test))




"""
print("number of subjects: %d"%num_subjects)
print("number of images: %d"%len(dataset_flattened[0]))
print("number of no roi: %d, with roi slices: %d" %(len(np.where(dataset_flattened[1]=='no')[0]),
												len(np.where(dataset_flattened[1]=='yes')[0])))


print("analyze on the dataset with ROI")
dataset_flattened=ds.resize_and_filter_partition(dataset_flattened,remove_non_roi=1,batch_size=1)
dataset_flattened=ds.filter_uncertain_partition(dataset_flattened)
print("num of bad: %d, num of good: %d, num of uncertain: %d" %(len(np.where(dataset_flattened[2]==1)[0]),
															len(np.where(dataset_flattened[2]==0)[0]),
															len(np.where(dataset_flattened[2]==-1)[0])))
"""

"""
datasets=ds.load_datasets(dataset_by_subject,remove_non_roi,batch_size,debug,dataset_partitions)

# filter data corresponding to uncertain categories
train_partition=ds.filter_uncertain_partition(datasets['train'])
val_partition=ds.filter_uncertain_partition(datasets['val'])
test_partition=ds.filter_uncertain_partition(datasets['test'])

train_images,train_labels,train_roi_labels,train_fnames=train_partition
val_images,val_labels,val_roi_labels,val_fnames=val_partition
test_images,test_labels,test_roi_labels,test_fnames=test_partition

# save reference of the original FFOV images
original_train_images=train_images.copy()
original_val_images=val_images.copy()
original_test_images=test_images.copy()

# get initial segmentations
roi_images_by_dataset,masks_by_dataset=sh.get_roi_images_by_dataset([train_images,val_images,test_images],seg_model_path)
train_images_seg,val_images_seg,test_images_seg=roi_images_by_dataset
train_masks_seg,val_masks_seg,test_masks_seg=masks_by_dataset

# heuristically fix segmentations
noise_threshold=50
train_masks_bb=sh.get_bounding_boxes(train_masks_seg,noise_threshold,use_context_padding)
val_masks_bb=sh.get_bounding_boxes(val_masks_seg,noise_threshold,use_context_padding)
test_masks_bb=sh.get_bounding_boxes(test_masks_seg,noise_threshold,use_context_padding)

train_images_bb=original_train_images*train_masks_bb
val_images_bb=original_val_images*val_masks_bb
test_images_bb=original_test_images*test_masks_bb

# filter dataset of images corresponding to 0 images

original_train_images,train_images_seg,train_images_bb,train_masks_bb,train_labels,train_fnames=ds.get_non_empty_dataset(original_train_images,
    train_images_seg,train_images_bb,train_masks_bb,train_labels,train_fnames)

original_val_images,val_images_seg,val_images_bb,val_masks_bb,val_labels,val_fnames=ds.get_non_empty_dataset(original_val_images,
    val_images_seg,val_images_bb,val_masks_bb,val_labels,val_fnames)

original_test_images,test_images_seg,test_images_bb,test_masks_bb,test_labels,test_fnames=ds.get_non_empty_dataset(original_test_images,
    test_images_seg,test_images_bb,test_masks_bb,test_labels,test_fnames)
"""

"""
home_dir=os.path.expanduser("~")
prefix_res_source_dir=os.path.join(home_dir,'..','..','mnt','d','exp_results','image_quality','images_labeled_dataset',
	'5_fold_cv')

res_source_dir=os.path.join(prefix_res_source_dir,'scratch')
main_source_dir=os.path.join(prefix_res_source_dir,'svm_scratch')



num_folds=len(os.listdir(res_source_dir))

# transfer relevant files to svm_scratch
# transfer class balance, without data augs, C=10.0
for fold_index in range(num_folds):
	fold_source_dir=os.path.join(main_source_dir,'results_fold_%d'%fold_index)
	if not os.path.isdir(fold_source_dir):
		os.mkdir(fold_source_dir)
	fold_source_dir=os.path.join(fold_source_dir,'final')
	if not os.path.isdir(fold_source_dir):
		os.mkdir(fold_source_dir)
	svm_source_dir=os.path.join(res_source_dir,'results_fold_%d'%fold_index,'results_svm','class_balance_1_C_10.0_data_aug_0')
	copyfile(os.path.join(svm_source_dir,'train_stats.mat'),os.path.join(fold_source_dir,'train_whole_stats.mat'))
	copyfile(os.path.join(svm_source_dir,'test_stats.mat'),os.path.join(fold_source_dir,'test_whole_stats.mat'))


print("finished transferring files")


# generate mean ROC curve from the k-fold CV
train_tprs=[]
train_fprs=[]
train_aucs=[]
train_thresholds=[]
min_train_thresholds=[]

test_tprs=[]
test_fprs=[]
test_aucs=[]
test_thresholds=[]
min_test_thresholds=[]

# iterate over the files and load the appropriate stats

home_dir=os.path.expanduser("~")
res_source_dir=os.path.join(home_dir,'..','..','mnt','d','exp_results','image_quality','images_labeled_dataset',
	'5_fold_cv','scratch')

for res_path in os.listdir(res_source_dir):
	final_res_path=os.path.join(res_source_dir,res_path,'final')
	train_stats=scipy.io.loadmat(os.path.join(final_res_path,'train_roi_stats'))
	test_stats=scipy.io.loadmat(os.path.join(final_res_path,'test_roi_stats'))

	train_tprs.append(train_stats['tpr'][0])
	train_fprs.append(train_stats['fpr'][0])
	train_aucs.append(train_stats['auc'][0])
	train_thresholds.append(train_stats['thresholds'][0])

	if len(min_train_thresholds)==0:
		min_train_thresholds=train_stats['thresholds'][0]
	else:
		if min_train_thresholds.shape[0] > train_stats['thresholds'][0].shape[0]:
			min_train_thresholds=train_stats['thresholds'][0]


	test_tprs.append(test_stats['tpr'][0])
	test_fprs.append(test_stats['fpr'][0])
	test_aucs.append(test_stats['auc'][0])
	test_thresholds.append(test_stats['thresholds'][0])


	if len(min_test_thresholds)==0:
		min_test_thresholds=test_stats['thresholds'][0]
	else:
		if min_test_thresholds.shape[0] > test_stats['thresholds'][0].shape[0]:
			min_test_thresholds=test_stats['thresholds'][0]


# due to variable size of thresholds, need to create a corresponding set of thresholds
# for computing avg tprs, etc.
def find_nearest(array,value):
	index=(np.abs(array-value)).argmin()
	return index


corr_train_tprs=[]
corr_train_fprs=[]
corr_train_aucs=[]

corr_test_tprs=[]
corr_test_fprs=[]
corr_test_aucs=[]

num_folds=len(train_tprs)
for i in range(num_folds):
	fold_train_tprs=train_tprs[i]
	fold_train_fprs=train_fprs[i]
	fold_train_aucs=train_aucs[i]
	fold_train_thresholds=train_thresholds[i]

	# find the closest corresponding set of thresholds based on the "min" threshold set
	ind_corr_thresholds=map(lambda x: find_nearest(fold_train_thresholds,x), min_train_thresholds)
	corr_train_tprs.append(fold_train_tprs[ind_corr_thresholds])
	corr_train_fprs.append(fold_train_fprs[ind_corr_thresholds])
	corr_train_aucs.append(sklearn.metrics.auc(corr_train_fprs[-1],corr_train_tprs[-1]))

	fold_test_tprs=test_tprs[i]
	fold_test_fprs=test_fprs[i]
	fold_test_aucs=test_aucs[i]
	fold_test_thresholds=test_thresholds[i]

	# find the closest corresponding set of thresholds based on the "min" threshold set
	ind_corr_thresholds=map(lambda x: find_nearest(fold_test_thresholds,x), min_test_thresholds)
	corr_test_tprs.append(fold_test_tprs[ind_corr_thresholds])
	corr_test_fprs.append(fold_test_fprs[ind_corr_thresholds])
	corr_test_aucs.append(sklearn.metrics.auc(corr_test_fprs[-1],corr_test_tprs[-1]))




corr_train_tprs=np.array(corr_train_tprs)
corr_train_fprs=np.array(corr_train_fprs)
corr_train_aucs=np.array(corr_train_aucs)

corr_test_tprs=np.array(corr_test_tprs)
corr_test_fprs=np.array(corr_test_fprs)
corr_test_aucs=np.array(corr_test_aucs)

mean_train_tprs=np.mean(corr_train_tprs,axis=0)
mean_train_fprs=np.mean(corr_train_fprs,axis=0)
mean_train_aucs=np.mean(corr_train_aucs)

std_train_tprs=np.std(corr_train_tprs,axis=0)
std_train_fprs=np.std(corr_train_fprs,axis=0)
std_train_aucs=np.std(corr_train_aucs)

mean_test_tprs=np.mean(corr_test_tprs,axis=0)
mean_test_fprs=np.mean(corr_test_fprs,axis=0)
mean_test_aucs=np.mean(corr_test_aucs)

std_test_tprs=np.std(corr_test_tprs,axis=0)
std_test_fprs=np.std(corr_test_fprs,axis=0)
std_test_aucs=np.std(corr_test_aucs)



# plot
plt.figure()

# baselines
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
         label='Chance', alpha=.8)

# ideal ROC curve
plt.plot([0,0,1],[0,1,1],color='g',label='best')

plt.plot(mean_train_fprs,mean_train_tprs,color='b',label='fine_tune')
tprs_upper=np.minimum(mean_train_tprs+std_train_tprs,1)
tprs_lower=np.maximum(mean_train_tprs-std_train_tprs,0)

fprs_upper=np.minimum(mean_train_fprs+std_train_fprs,1)
fprs_lower=np.maximum(mean_train_fprs-std_train_fprs,0)

"""
"""
plt.errorbar(mean_train_fprs,mean_train_tprs,yerr=std_train_tprs,xerr=std_train_fprs,elinewidth=0.2,capthick=0.2,
	errorevery=1,fillstyle='full')
"""


"""
plt.fill_between(mean_train_fprs,tprs_lower,tprs_upper,color='b', alpha=.2)
plt.fill_betweenx(mean_train_tprs,fprs_lower,fprs_upper,color='b',alpha=.2)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Avg. ROC on Training Data: \n Avg. AUC %0.2f +/- %0.2f' %(mean_train_aucs,std_train_aucs))
plt.legend(loc="lower right")
plt.savefig('train_roc')


plt.figure()

# baselines
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
         label='Chance', alpha=.8)
# ideal ROC curve
plt.plot([0,0,1],[0,1,1],color='g',label='best')

plt.plot(mean_test_fprs,mean_test_tprs,color='b',label='fine_tune')
tprs_upper=np.minimum(mean_test_tprs+std_test_tprs,1)
tprs_lower=np.maximum(mean_test_tprs-std_test_tprs,0)

fprs_upper=np.minimum(mean_test_fprs+std_test_fprs,1)
fprs_lower=np.maximum(mean_test_fprs-std_test_fprs,0)
"""
"""
plt.errorbar(mean_train_fprs,mean_train_tprs,yerr=std_train_tprs,xerr=std_train_fprs,elinewidth=0.2,capthick=0.2,
	errorevery=1,fillstyle='full')
"""


"""
plt.fill_between(mean_test_fprs,tprs_lower,tprs_upper,color='b', alpha=.2)
plt.fill_betweenx(mean_test_tprs,fprs_lower,fprs_upper,color='b',alpha=.2)
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Avg. ROC on Test Data: \n Avg. AUC %0.2f +/- %0.2f' %(mean_test_aucs,std_test_aucs))
plt.legend(loc="lower right")
plt.savefig('test_roc')
"""
# try error bar plot to see how this looks)


"""
# summarize stats over the varioyus fine tunining methods for CNNs
fold_dirs=os.listdir(res_source_dir)

# aggregate fold stats by fine tuning method
methods=['all','none','some','one']
train_auc_by_method={'none':[],'all':[],'some':[],'one':[]}
test_auc_by_method={'none':[],'all':[],'some':[],'one':[]}


for fold_dir in fold_dirs:
	if 'fold' not in fold_dir:
		continue

	fold_dir=os.path.join(res_source_dir,fold_dir)
	for method in methods:
		method_res_dir=os.path.join(fold_dir,'results_layers_'+method+'_freeze_0','final')
		train_stats=scipy.io.loadmat(os.path.join(method_res_dir,'train_whole_stats.mat'))
		train_auc_by_method[method].append(train_stats['auc'])
		test_stats=scipy.io.loadmat(os.path.join(method_res_dir,'test_whole_stats.mat'))
		test_auc_by_method[method].append(test_stats['auc'])


# compute average fold stats
avg_train_auc_by_method={}
avg_test_auc_by_method={}
for method in methods:
	avg_train_auc_by_method[method]=[np.mean(train_auc_by_method[method]),np.std(train_auc_by_method[method])]
	avg_test_auc_by_method[method]=[np.mean(test_auc_by_method[method]),np.std(test_auc_by_method[method])]

np.save(os.path.join(res_source_dir,'avg_train_cnn_res.npy'),avg_train_auc_by_method)
np.save(os.path.join(res_source_dir,'avg_test_cnn_res.npy'),avg_test_auc_by_method)


train_means=[avg_train_auc_by_method['none'][0],avg_train_auc_by_method['one'][0],
	avg_train_auc_by_method['some'][0],avg_train_auc_by_method['all'][0]]
train_std=[avg_train_auc_by_method['none'][1],avg_train_auc_by_method['one'][1],
	avg_train_auc_by_method['some'][1],avg_train_auc_by_method['all'][1]]

test_means=[avg_test_auc_by_method['none'][0],avg_test_auc_by_method['one'][0],
	avg_test_auc_by_method['some'][0],avg_test_auc_by_method['all'][0]]
test_std=[avg_test_auc_by_method['none'][1],avg_test_auc_by_method['one'][1],
	avg_test_auc_by_method['some'][1],avg_test_auc_by_method['all'][1]]


# add the SVM results
# svm results
train_aucs=[0.9,0.95,0.95]
test_aucs=[0.66,0.73,0.58]

svm_train_mean=np.mean(train_aucs)
svm_train_std=np.std(train_aucs)

svm_test_mean=np.mean(test_aucs)
svm_test_std=np.std(test_aucs)

train_means.append(svm_train_mean)
train_std.append(svm_train_std)
test_means.append(svm_test_mean)
test_std.append(svm_test_std)


fig,ax=plt.subplots()
ind=np.arange(5)
width=0.25
train_data=ax.bar(ind,train_means,color='m',yerr=train_std,label='train',width=width)
test_data=ax.bar(ind+width,test_means,color='c',yerr=test_std,label='test',width=width)

ax.set_ylabel('AUC')
ax.set_title('Average AUCs')
ax.set_xticks(ind+width/2)
ax.set_xticklabels(('CNN-none','CNN-one','CNN-some','CNN-all','CNN-SVM'))
ax.legend()

plt.savefig(os.path.join(res_source_dir,'aggr_stats'))








for res_path in os.listdir(res_source_dir):
	tmp_res_path=os.path.join(res_source_dir,res_path)

	num_files=len(os.listdir(tmp_res_path))
	if num_files==0:
		continue

	tmp_res=scipy.io.loadmat(os.path.join(tmp_res_path,'train_stats'))

	# plot ROC
	plt.figure()
	lw=2
	plt.plot(tmp_res['fpr'],tmp_res['tpr'],color='darkorange')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve (area=%0.2f)'%tmp_res['auc'])
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(tmp_res_path,'train_roc'))
	plt.close()

	tmp_res=scipy.io.loadmat(os.path.join(tmp_res_path,'val_stats'))
	# plot ROC
	plt.figure()
	lw=2
	plt.plot(tmp_res['fpr'],tmp_res['tpr'],color='darkorange')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve (area=%0.2f)'%tmp_res['auc'])
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(tmp_res_path,'val_roc'))
	plt.close()


	tmp_res=scipy.io.loadmat(os.path.join(tmp_res_path,'test_stats'))
	# plot ROC
	plt.figure()
	lw=2
	plt.plot(tmp_res['fpr'],tmp_res['tpr'],color='darkorange')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve (area=%0.2f)'%tmp_res['auc'])
	plt.legend(loc="lower right")
	plt.savefig(os.path.join(tmp_res_path,'test_roc'))
	plt.close()
"""
"""
datasets=th.load_datasets(data_source_dir,1,10,1,'',None)
sample_im=datasets['train'][0][0]
sample_im_uint8=skimage.img_as_ubyte(sample_im)
skimage.io.imsave('sample_im_uint8.png',sample_im_uint8)
skimage.io.imsave('sample_im_uint16.png',sample_im)
"""
"""

def plot_roc(thresholds,fpr,tpr,auc,save_path):
	plt.figure()
	lw=2
	plt.plot(fpr,tpr,color='darkorange')
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--',label="random guessing")
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve (area=%0.2f)'%auc)
	plt.legend(loc="lower right")
	plt.savefig(save_path)
	plt.close()

# viualize results
res_dir=os.path.join(os.path.expanduser("~"),'../../mnt/d/exp_results/image_quality/images_labeled_dataset')

methods=['scratch','fine_tune']

datasets=['train','val','test']
# generate ROC plots over train, val, test 
# generate loss plots
for method in methods:
	method_res_dir=os.path.join(res_dir,method,'results')



	# loss plots
	train_loss=np.load(os.path.join(method_res_dir,'final','train_loss.npy'))
	val_loss=np.load(os.path.join(method_res_dir,'final','val_loss.npy'))
	num_epochs=len(train_loss)
	epoch_ind=np.arange(num_epochs)
	plt.figure()
	plt.plot(epoch_ind,train_loss,'r',label='train')
	plt.plot(epoch_ind,val_loss,'b',label='validation')
	plt.legend(loc="lower right")
	plt.xlabel("Epoch indices")
	plt.ylabel("Loss")
	plt.savefig(os.path.join(method_res_dir,'loss_plot'))
	plt.close()


	models=['best','final']


	# roc curves

	for model in models:
		model_method_res_dir=os.path.join(method_res_dir,model)

		for dataset_name in datasets:
			dataset_stats=scipy.io.loadmat(os.path.join(model_method_res_dir,dataset_name+'_stats'))
			plot_roc(dataset_stats['thresholds'][0],dataset_stats['fpr'][0], dataset_stats['tpr'][0],dataset_stats['auc'][0],os.path.join(model_method_res_dir,dataset_name+'_roc'))

"""

"""
subject_source=os.path.join('data/all_subjects/sample_subject_1')
true_images=scipy.io.loadmat(os.path.join(subject_source,'loaded_haste_vols_v2'))['subject_data_by_vol'][0]
subject_1_data=ds.load_subject_data(subject_source)
vol_1_images=np.transpose(subject_1_data[0][0],[2,0,1])


sample_im=vol_1_images[0]


# simulate noise
noise=np.random.normal(size=(256,256))*200

plt.figure()
plt.imshow(sample_im,cmap='gray')
plt.savefig('orig_im')
plt.imshow(noise,cmap='gray')
plt.savefig('noise_img')

plt.figure()
plt.imshow(sample_im+noise,cmap='gray')
plt.savefig('noise_corr_img')
"""

"""
sample_im=sample_im.astype('uint8')



seq = iaa.Sequential([iaa.AdditiveGaussianNoise(scale=0)])
aug_im=seq.augment_images(sample_im)
plt.figure()
plt.imshow(aug_im[:,:,0],cmap='gray')
plt.savefig('aug_im')
plt.close()

plt.figure()
plt.imshow(sample_im[:,:,0],cmap='gray')
plt.savefig('orig_im')
plt.close()
"""
"""
home_dir=os.path.expanduser("~")
model_source_dir=os.path.join(home_dir,'..','..','mnt','d','exp_results','image_quality','images_labeled_dataset','resnet','200_epochs_no_weight_decay')
model_fine_tuned_path=os.path.join(model_source_dir,'fine_tuned','model_0')
model_scratch_path=os.path.join(model_source_dir,'scratch','model_0')

saved_params_model_fine_tuned=pickle.load(open(model_fine_tuned_path,'r'))
saved_params_model_scratch=pickle.load(open(model_scratch_path,'r'))
"""

#saved_params_path=open('results/fold_2/models/fine_tuned_resnet','r')
"""
home_dir=os.path.expanduser("~")
res_source_dir=os.path.join(home_dir,'..','..','mnt','d','exp_results','image_quality','images_labeled_dataset','resnet_svm','resnet_200_epochs','fined_tuned','results')
reg_values=[0.01,0.1,1,10,100]
for reg_val in reg_values:
	print("reg val", reg_val)
	train_res=np.load(os.path.join(res_source_dir,'train_stats_'+str(reg_val)+'.npy'))
	test_res=np.load(os.path.join(res_source_dir,'test_stats_'+str(reg_val)+'.npy'))
	print("unbalanced")
	print(train_res)
	print(test_res)

	train_res=np.load(os.path.join(res_source_dir,'train_balanced_stats_'+str(reg_val)+'.npy'))
	test_res=np.load(os.path.join(res_source_dir,'test_balanced_stats_'+str(reg_val)+'.npy'))
	print("balanced")
	print(train_res)
	print(test_res)
"""

"""

lasagne.layers.set_all_param_values(network['pool5'],saved_params[:-2])
"""

"""
home_dir=os.path.expanduser("~")
data_source_dir=os.path.join(home_dir,'..','..','mnt','d','raw_dicoms','single','haste','haste_good_bad')
dataset_by_subject=ds.load_subject_data(data_source_dir)
train_subject_indices=np.arange(1)
train_partition=ds.getDatasetPartition(dataset_by_subject,train_subject_indices)
train_flattened_partition=ds.flattenPartition(train_partition,1)
sample_im=train_flattened_partition[0][20]

np.save('sample_im',sample_im)
"""

"""
img=np.load('data/sample_im_2.npy')
plt.figure()
plt.imshow(img,cmap="gray")
plt.savefig('results/orig')

trans_img=ds.translate(img,30,30)
plt.figure()
plt.imshow(trans_img,cmap="gray")
plt.savefig('results/demo_trans')
print(trans_img.shape)

rot_img=ds.rotate(img,180)
plt.figure()
plt.imshow(rot_img,cmap="gray")
plt.savefig('results/demo_rot')
print(rot_img.shape)

flip_img=cv2.flip(img,-1)
print(flip_img.shape)
plt.figure()
plt.imshow(flip_img,cmap="gray")
plt.savefig('results/hflip')


flip_img=cv2.flip(img,0)
print(flip_img.shape)
plt.figure()
plt.imshow(flip_img,cmap="gray")
plt.savefig('results/vflip')

flip_img=cv2.flip(img,1)
print(flip_img.shape)
plt.figure()
plt.imshow(flip_img,cmap="gray")
plt.savefig('results/both_flip')



plt.figure()
plt.imshow(ds.transform(img),cmap='gray')
plt.savefig('results/trans_img')

plt.close("all")
"""

"""
home_dir=os.path.expanduser("~")
data_source_dir=os.path.join(home_dir,'..','..','mnt','d','raw_dicoms','single','haste','haste_preload')

dataset_by_subject=ds.load_all_subjects_data(data_source_dir)
num_subjects=10


kf=KFold(5)
fold_index=0
for train_subject_indices,test_subject_indices in kf.split(np.arange(num_subjects)):
	print("fold index", str(fold_index))
	train_partition=ds.getDatasetPartition(dataset_by_subject,train_subject_indices)
	test_partition=ds.getDatasetPartition(dataset_by_subject,test_subject_indices)

	train_flattened,test_flattened=map(ds.flattenPartition,
													[train_partition,test_partition],
													[1]*2)

	train_partition_preprocessed,test_partition_preprocessed=map(ds.preprocessPartition,
																						[train_flattened,test_flattened],
																						[0]*2,[1]*2,[1]*2)

	train_images,train_roi_labels,train_labels=train_partition_preprocessed
	test_images,test_roi_labels,test_labels=test_partition_preprocessed

	num_train_bad=len(np.where(train_labels==0)[0])
	num_train_good=len(np.where(train_labels==1)[0])
	num_test_bad=len(np.where(test_labels==0)[0])
	num_test_good=len(np.where(test_labels==1)[0])

	print("number of images in train good vs bad", num_train_good,num_train_bad, num_train_bad*1.0/(num_train_bad+num_train_good))
	print("number of images in test good vs bad", num_test_good,num_test_bad,num_test_bad*1.0/(num_test_bad+num_test_good))

	# post removal from train and test
	num_good_images_to_remove_from_train=int(num_train_good-(100.0*num_train_bad/25))
	updated_train_partition=ds.remove_quality_label_images(train_partition_preprocessed,num_good_images_to_remove_from_train,1)


	num_good_images_to_remove_from_test=int(num_test_good-(100.0*num_test_bad/25))
	updated_test_partition=ds.remove_quality_label_images(test_partition_preprocessed,num_good_images_to_remove_from_test,1)


	train_images,train_roi_labels,train_labels=updated_train_partition
	test_images,test_roi_labels,test_labels=updated_test_partition

	num_train_bad=len(np.where(train_labels==0)[0])
	num_train_good=len(np.where(train_labels==1)[0])
	num_test_bad=len(np.where(test_labels==0)[0])
	num_test_good=len(np.where(test_labels==1)[0])
	print("post update")
	print("number of images in train good vs bad", num_train_good,num_train_bad, num_train_bad*1.0/(num_train_bad+num_train_good))
	print("number of images in test good vs bad", num_test_good,num_test_bad,num_test_bad*1.0/(num_test_bad+num_test_good))



	fold_index+=1
"""