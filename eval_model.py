import keras
import dataset as ds
import argparse
import os
import shutil
import numpy as np
import glob
import vis
from vis.visualization import visualize_cam
import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import matplotlib.gridspec as gridspec




"""
model: keras instance
image: 3d array should have the appropriate preprocessing applied since this is used for model evaluation

Uses Grad-CAM to project the final convolutional layer onto the original input image.

Returns the normalized, upsampled activation map. This includes the positive and negative gradients.

"""
def get_smap(model,image):
	smap= visualize_cam(model,layer_idx=-1,filter_indices=0,seed_input=image)
	pos_heatmap,neg_heatmap,raw_heatmap,raw_heatmap_resized,feature_maps=smap
	# normalize the heatmap to maintain the direction and rel magnitude
	max_magnitude=np.amax(abs(raw_heatmap_resized))
	heatmap_normalized=raw_heatmap_resized*1.0/max_magnitude
	return heatmap_normalized,raw_heatmap,raw_heatmap_resized,feature_maps


"""
dir_to_save_to: str where to make this folder
confidence_bin: 2d tuple of float (lower_bound,upper_bound)

Outputs a string representing the path of the directory
"""
def create_confidence_bin_dir(dir_to_save_to,confidence_bin):
	confidence_bin_lb=confidence_bin[0]
	confidence_bin_ub=confidence_bin[1]
	confidence_bin_directory=os.path.join(dir_to_save_to,'smaps_confidence_bin_%.2f_%.2f'
					%(confidence_bin_lb,confidence_bin_ub))
	os.mkdir(confidence_bin_directory)
	os.mkdir(os.path.join(confidence_bin_directory,'correct'))
	os.mkdir(os.path.join(confidence_bin_directory,'incorrect'))

	return confidence_bin_directory
"""
Produces a panel where the image, the smap, and the smap overlaid on the image is shown.


image: 2d array unprocessed
smap: 2d array Normalized heatmap intensities scaled to [-1,1]
"""
def visualize_smap(image,smap,fg_color="white",include_cbar=False):
	fig,ax=plt.subplots(1,3,figsize=(20,20))
	ax[0].imshow(image,cmap='gray')
	
	im_with_colormap=ax[1].imshow(smap,vmin=-1,vmax=1,cmap=plt.get_cmap('RdYlGn_r'))
	ax[2].imshow(image,cmap='gray')
	ax[2].imshow(smap,vmin=-1,vmax=1,alpha=0.4,cmap=plt.get_cmap('RdYlGn_r'))
	
	for i in range(3):
		ax[i].tick_params(which='both',
						bottom=False,top=False,left=False,right=False,
						labelbottom=False,labelleft=False)
	
	if include_cbar:
		cbar=fig.colorbar(im_with_colormap,ax=ax,shrink=0.8)
		cbar.set_label("normalized grad-cam values",
						rotation=270,color=fg_color,labelpad=80,
						size=30)
		ticks=np.arange(-1,1.1,0.5)
		cbar.set_ticks(ticks)
		# tick_labels=np.flip(ticks)
		cbar.set_ticklabels(ticks)
		cbar.ax.tick_params(axis='y',which="major",pad=8,labelsize=30)
		cbar.ax.yaxis.set_tick_params(color=fg_color)
		cbar.outline.set_edgecolor(fg_color)
		plt.setp(plt.getp(cbar.ax.axes,'yticklabels'),color=fg_color)
	
	return fig

"""
images
smaps
scores
labels
fnames
"""
def compare_smaps(images,smaps,scores,labels,include_cbar):
	num_images=len(images)
	fig,ax=plt.subplots(num_images,3,figsize=(40,40))
	gs=gridspec.GridSpec(num_images,3)
	gs.update(wspace=0.02,hspace=0.02)
	for i in range(num_images):
		ax[i,0].imshow(images[i],cmap='gray')
		
		im_with_colormap=ax[i,1].imshow(smaps[i],vmin=-1,vmax=1,cmap=plt.get_cmap('RdYlGn_r'))
		ax[i,2].imshow(images[i],cmap='gray')
		ax[i,2].imshow(smaps[i],vmin=-1,vmax=1,alpha=0.4,cmap=plt.get_cmap('RdYlGn_r'))
		
		for j in range(3):
			ax[i,j].tick_params(which='both',
							bottom=False,top=False,left=False,right=False,
							labelbottom=False,labelleft=False)


	if include_cbar:
		cbar=fig.colorbar(im_with_colormap,ax=ax,shrink=0.8)
		cbar.set_label("normalized grad-cam values",
						rotation=270,color=fg_color,labelpad=80,
						size=30)
		ticks=np.arange(-1,1.1,0.5)
		cbar.set_ticks(ticks)
		# tick_labels=np.flip(ticks)
		cbar.set_ticklabels(ticks)
		cbar.ax.tick_params(axis='y',which="major",pad=8,labelsize=30)
		cbar.ax.yaxis.set_tick_params(color=fg_color)
		cbar.outline.set_edgecolor(fg_color)
		plt.setp(plt.getp(cbar.ax.axes,'yticklabels'),color=fg_color)

	fig.tight_layout()
	return fig



def save_example_feature_maps(feature_maps,heatmap,heatmap_resized,image):
	plt.figure()
	plt.imshow(image,cmap='gray')
	plt.savefig('sample_im')
	plt.close()

	sample_ind=np.array([238,239,240])
	sample_feature_maps=feature_maps[0,:,:,:]
	sample_feature_maps=np.transpose(sample_feature_maps,(2,0,1))

	for i in sample_ind:
		plt.figure()
		plt.imshow(sample_feature_maps_neg[i,:,:],cmap=plt.get_cmap('RdYlGn_r'))
		plt.savefig('sample_feature_maps_%d' %i)
		plt.close()

	plt.figure()
	plt.imshow(heatmap,cmap=plt.get_cmap('RdYlGn_r'))
	plt.savefig('heatmap')
	plt.close()

	plt.figure()
	plt.imshow(heatmap_resized,cmap=plt.get_cmap('RdYlGn_r'))
	plt.savefig('heatmap_resized')
	plt.close()



if __name__=='__main__':
	parser = argparse.ArgumentParser()
	# data/model loading/ save paths
	parser.add_argument('--model_type',default='shallow')

	parser.add_argument('--data_source_dir',default='')
	parser.add_argument('--data_partition_dir',default='')
	parser.add_argument('--main_res_path',default='results')

	# debug
	parser.add_argument('--save_images',default='0')

	# parameters
	args=parser.parse_args()

	
	model_type=args.model_type

	data_source_dir=args.data_source_dir
	data_partition_dir=args.data_partition_dir
	main_res_path=args.main_res_path

	save_images=int(args.save_images)

	best_model_res_path=os.path.join(main_res_path,'best')
	if os.path.isdir(best_model_res_path):
		shutil.rmtree(best_model_res_path)
	os.mkdir(best_model_res_path)

	best_model_file=glob.glob(os.path.join(main_res_path,"models","best.hdf5"))[0]
	model=keras.models.load_model(best_model_file)
	sess=keras.backend.get_session()

	save_images_path=''
	if save_images:
		save_images_path=os.path.join(best_model_res_path,'sample_images')
		os.mkdir(save_images_path)

	transfer=True if model_type=='vgg_16_transfer' else False
	num_channels=3 if transfer else 1
	input_shape=(256,256,num_channels)

	
	val_g=ds.DataGenerator(data_partition_path=os.path.join(data_partition_dir,'val.npy'),
							data_source_dir=data_source_dir,
							dim=input_shape,
							save_images=save_images,save_images_path=save_images_path)

	val_preds=model.predict_generator(val_g)
	np.savetxt(os.path.join(best_model_res_path,'val_preds.csv'),val_preds)


	# note shuffling is disabled, batch size is 1 so that data yielded from this generator
	# follows the order of the data in test.npy
	test_g=ds.DataGenerator(data_partition_path=os.path.join(data_partition_dir,'test.npy'),
							data_source_dir=data_source_dir,
							dim=input_shape,
							save_images=save_images,save_images_path=save_images_path)

	test_preds=model.predict_generator(test_g)
	np.savetxt(os.path.join(best_model_res_path,'test_preds.csv'),test_preds)


	# generate saliency maps
	
	bg_color="black"
	fg_color="white"
	include_cbar=True
	test_data=np.load(os.path.join(data_partition_dir,'test.npy'))

	confidence_thresholds=np.array([0,0.25,0.5,0.75])
	test_indices_by_confidence=ds.bin_by_confidence(confidence_thresholds,test_preds[:,0])
	test_accuracy_by_confidence={}
	test_distribution_by_confience={}

	num_test_samples=len(test_data)
	num_samples=10

	for confidence_bin in test_indices_by_confidence:
		confidence_bin_directory=create_confidence_bin_dir(best_model_res_path,confidence_bin)

		ind_test_samples=test_indices_by_confidence[confidence_bin]

		test_distribution_by_confience[confidence_bin]=len(ind_test_samples)*1.0/num_test_samples

		num_correct=0
		
		subset_ind_test_samples=np.random.choice(ind_test_samples,num_samples,replace=False)
		for sample_index in subset_ind_test_samples:
			sample_data=test_data[sample_index]
			_,_,subject_path,stack_path,dicom_path=sample_data

			sample_pred=test_preds[sample_index,0]
			sample_image_processed,sample_label=test_g.__getitem__(sample_index)
			pred_label=1 if sample_pred >=0.5 else 0
			is_pred_correct=pred_label==sample_label
			num_correct = num_correct+1 if is_pred_correct else num_correct

			
			sample_image_processed=sample_image_processed[0]
			sample_image_unprocessed=ds.get_image_array(data_source_dir,subject_path,stack_path,dicom_path)
			smap,heatmap,raw_heatmap_resized,feature_maps=get_smap(model,sample_image_processed)
		
		
			smap_figure=visualize_smap(sample_image_unprocessed,smap,fg_color=fg_color,include_cbar=include_cbar)
			quality='Non-Diagnostic' if sample_label==1 else 'Diagnostic'
			smap_figure.suptitle('Ground truth: %s, Confidence: %.2f, Image and Saliency Map' \
								%(quality,sample_pred),color=fg_color)
			

			dicom_path_sans_ext,ext=os.path.splitext(dicom_path)
			fname=subject_path+'_'+stack_path+'_'+dicom_path_sans_ext+'.png'
			correct_or_incorrect_dir='correct' if is_pred_correct else 'incorrect'
			plt.savefig(os.path.join(confidence_bin_directory,correct_or_incorrect_dir,fname),
				facecolor=bg_color,transparent=True)
			
			
		test_accuracy_by_confidence[confidence_bin]=num_correct*1.0/len(ind_test_samples)

	
	# smap_comparison_dir=os.path.join(best_model_res_path,'smap_comparison')
	# os.mkdir(smap_comparison_dir)
	"""
	images=[]
	smaps=[]
	labels=[]
	scores=[]
	fnames=[]
	confidence_bins=[(0,0.25),(0.25,0.5),(0.5,0.75),(0.75,1)]
	for confidence_bin in confidence_bins:
		random_sample_ind=np.random.choice(test_indices_by_confidence[confidence_bin])
		sample_image_processed,sample_label=test_g.__getitem__(random_sample_ind)
		sample_data=test_data[random_sample_ind]
		_,_,subject_path,stack_path,dicom_path=sample_data
		sample_image_unprocessed=ds.get_image_array(data_source_dir,subject_path,stack_path,dicom_path)
		sample_image_smap,_,_,_=get_smap(model,sample_image_processed)
		sample_score=test_preds[random_sample_ind]

		images.append(sample_image_unprocessed)
		smaps.append(sample_image_smap)
		labels.append(sample_label)
		scores.append(sample_score)
		fnames.append([subject_path,stack_path,dicom_path])

		smap_figure=visualize_smap(sample_image_unprocessed,
									sample_image_smap,fg_color=fg_color,include_cbar=include_cbar)
		quality='Non-Diagnostic' if sample_label==1 else 'Diagnostic'
		smap_figure.suptitle('Ground truth: %s, Confidence: %.2f, Image and Saliency Map' \
							%(quality,sample_score),color=fg_color)
		

		dicom_path_sans_ext,ext=os.path.splitext(dicom_path)
		fname=subject_path+'_'+stack_path+'_'+dicom_path_sans_ext+'.png'
		correct_or_incorrect_dir='correct' if is_pred_correct else 'incorrect'
		plt.savefig(os.path.join(smap_comparison_dir,fname),
			facecolor=bg_color,transparent=True)
		plt.close()
	compare_smap_fig=compare_smaps(images,smaps,scores,labels,fnames)
	plt.savefig(os.path.join(smap_comparison_dir,'final_comparison'),
			facecolor=bg_color,transparent=True)

	print(scores)
	print(fnames)
	print(labels)
	"""
	"""
	print("accuracy by confidence bin,", test_accuracy_by_confidence)
	print("distribution by confidence bin", test_distribution_by_confience)	
	"""
	"""
	# pick examples to compare smaps for 
	examples=np.array([['case1','stack_2','CASE1_1.MR.0003.0026.2017.10.19.16.34.59.646659.29233925'],
			['HASTE_brain (4)',
			'stack_4','PLACENTA053017_1.MR.0006.0019.2017.05.30.19.48.45.302924.506331942'],
			['HASTE_brain',
			'stack_4','PLACENTA_03162016_1.MR.0003.0023.2016.03.16.18.46.20.856518.24125245'],
			['HASTE_brain (18)',
			'stack_5',
			'PS082517_1.MR.0007.0029.2017.08.25.20.02.56.182603.597861938'],
			['HASTE_brain','stack_3','PLACENTA_03162016_1.MR.0004.0011.2016.03.16.18.46.20.856518.24131409']])
	num_examples=len(examples)
	# extract these slices to generate the desired figures

	test_data_dicom_names_sans_ext=np.array(map(lambda x: os.path.splitext(x[-1])[0],test_data))
	ind_examples=np.intersect1d(test_data_dicom_names_sans_ext,examples[:,-1],return_indices=True)[1]
	images=[]
	smaps=[]
	labels=[]
	scores=[]
	fnames=[]

	for random_sample_ind in ind_examples:
		sample_image_processed,sample_label=test_g.__getitem__(random_sample_ind)
		sample_data=test_data[random_sample_ind]
		_,_,subject_path,stack_path,dicom_path=sample_data
		dicom_path=os.path.splitext(dicom_path)[0]
		sample_image_unprocessed=ds.get_image_array(data_source_dir,subject_path,stack_path,dicom_path)
		sample_image_smap,_,_,_=get_smap(model,sample_image_processed)
		sample_score=test_preds[random_sample_ind]

		images.append(sample_image_unprocessed)
		smaps.append(sample_image_smap)
		labels.append(sample_label)
		scores.append(sample_score)

		fname="".join([subject_path,stack_path,dicom_path])
		fnames.append(fname)

	images=np.array(images)
	smaps=np.array(smaps)
	labels=np.array(labels)[:,0]
	scores=np.array(scores)[:,0]
	fnames=np.array(fnames)

	# sort the data by the confidence level
	ind_sorted_scores=np.argsort(scores)
	images=images[ind_sorted_scores]
	smaps=smaps[ind_sorted_scores]
	labels=labels[ind_sorted_scores]
	scores=scores[ind_sorted_scores]
	fnames=fnames[ind_sorted_scores]
	# produce the figure
	for i in range(num_examples):
		sample_image_unprocessed=images[i]
		sample_image_smap=smaps[i]
		sample_label=labels[i]
		sample_score=scores[i]
		fname=fnames[i]
		smap_figure=visualize_smap(sample_image_unprocessed,
									sample_image_smap,fg_color=fg_color,include_cbar=include_cbar)
		quality='Non-Diagnostic' if sample_label==1 else 'Diagnostic'
		smap_figure.suptitle('Ground truth: %s, Confidence: %.2f, Image and Saliency Map' \
							%(quality,sample_score),color=fg_color)
		
		
		plt.savefig(os.path.join(smap_comparison_dir,fname+'.png'),
			facecolor=bg_color,transparent=True)
		plt.close()
	compare_smap_fig=compare_smaps(images,smaps,scores,labels,include_cbar)
	plt.savefig(os.path.join(smap_comparison_dir,'final_comparison'),
			facecolor=bg_color,transparent=True)

	print(scores)
	print(fnames)
	print(labels)
	"""