import numpy as np
import dataset as ds
import os
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import shutil






def plot_bad_vs_good_by_subject(num_bad_by_subject,num_total_by_subject,subject_labels,
								dir_to_save_to='',data_type=''):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	width=0.4

	num_subjects=len(subject_labels)

	# order based on the fraction of bad data
	frac_bad_by_subject=num_bad_by_subject*1.0/num_total_by_subject
	num_good_by_subject=num_total_by_subject-num_bad_by_subject
	frac_good_by_subject=num_good_by_subject*1.0/num_total_by_subject

	ind_ordered_increasing=np.argsort(frac_bad_by_subject)
	frac_bad_by_subject=frac_bad_by_subject[ind_ordered_increasing]
	frac_good_by_subject=frac_good_by_subject[ind_ordered_increasing]
	subject_labels=subject_labels[ind_ordered_increasing]

	for subject_index in range(num_subjects):
		p1=ax.bar(subject_index,frac_bad_by_subject[subject_index],width=width,color='lightcoral')
		p2=ax.bar(subject_index,frac_good_by_subject[subject_index],width=width,color='mediumseagreen',
				bottom=frac_bad_by_subject[subject_index])



	xleft=-1*width
	ax.set_xlim(left=xleft,right=len(subject_names)+1)
	ax.set_xticks(np.arange(num_subjects))    
	ax.set_xticklabels(subject_labels,rotation=90)
	ax.tick_params(axis='x',labelsize=5)
	ax.set_ylabel("Percent of %s"%data_type)
	ax.set_title('Distribution of good/bad %s across subjects' %data_type)
	plt.legend((p1[0],p2[0]),('bad','good'))
	fig.savefig(os.path.join(dir_to_save_to,'frac_%s_good_bad_by_subject'%data_type))
	plt.close()


	# plot the actual numbers
	num_bad_by_subject=num_bad_by_subject[ind_ordered_increasing]
	num_good_by_subject=num_good_by_subject[ind_ordered_increasing]
	for subject_index in range(num_subjects):
		p1=ax.bar(subject_index,num_bad_by_subject[subject_index],width=width,color='lightcoral')
		p2=ax.bar(subject_index,num_good_by_subject[subject_index],width=width,color='mediumseagreen',
				bottom=num_bad_by_subject[subject_index])



	xleft=-1*width
	ax.set_xlim(left=xleft,right=len(subject_names)+1)
	ax.set_xticks(np.arange(num_subjects))    
	ax.set_xticklabels(subject_labels,rotation=90)
	ax.tick_params(axis='x',labelsize=5)
	ax.set_ylabel("# of %s"%data_type)
	ax.set_title('Distribution of good/bad across subjects')
	plt.legend((p1[0],p2[0]),('bad','good'))
	fig.savefig(os.path.join(dir_to_save_to,'num_%s_good_bad_by_subject'%data_type))
	plt.close()



"""
Generates histogram and bar chart by subject for analyzing the distribution

data_by_subject: 1d array
data_type : str
save_path: str
"""
def plot_distribution(data_by_subject,subject_labels,data_name,dir_to_save_to='',plot_in_order=False):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	width=0.4

	num_subjects=len(subject_labels)

	if plot_in_order:
		ind_ordered_increasing=np.argsort(data_by_subject)
		data_by_subject=data_by_subject[ind_ordered_increasing]
		subject_labels=subject_labels[ind_ordered_increasing]

	for subject_index in range(num_subjects):
		ax.bar(subject_index,data_by_subject[subject_index],width=width,color='lightcoral')



	xleft=-1*width
	ax.set_xlim(left=xleft,right=len(subject_names)+1)
	ax.set_xticks(np.arange(num_subjects))    
	ax.set_xticklabels(subject_labels,rotation=90)
	ax.tick_params(axis='x',labelsize=5)
	ax.set_ylabel(data_name)
	ax.set_title('Distribution of %s across subjects'%data_name)
	fig.savefig(os.path.join(dir_to_save_to,'%s_by_subject'%data_name))
	plt.close()


	plt.figure()
	plt.hist(data_by_subject,density=False)
	plt.xlabel(data_name)
	plt.ylabel('# subjects')
	plt.title('Distribution of %s across subjects'%data_name)
	plt.savefig(os.path.join(dir_to_save_to,'%s_hist'%data_name))
	plt.close()



def plot_distribution_across_stacks(dataset_by_stack,stack_labels,data_name,dir_to_save_to='',plot_in_order=False):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	width=0.4

	num_stacks=len(stack_labels)

	if plot_in_order:
		ind_ordered_increasing=np.argsort(dataset_by_stack)
		dataset_by_stack=dataset_by_stack[ind_ordered_increasing]
		stack_labels=stack_labels[ind_ordered_increasing]

	for stack_index in range(num_stacks):
		ax.bar(stack_index,dataset_by_stack[stack_index],width=width)



	xleft=-1*width
	ax.set_xlim(left=xleft,right=len(stack_labels)+1)
	ax.set_xticks(np.arange(len(stack_labels))) 
	ax.set_xticklabels(stack_labels,rotation=90)
	ax.tick_params(axis='x',labelsize=5)
	ax.set_ylabel(data_name)
	ax.set_title('Distribution of %s across stacks'%data_name)
	fig.savefig(os.path.join(dir_to_save_to,'%s_by_stack'%data_name))
	plt.close()


	plt.figure()
	plt.hist(dataset_by_stack,density=False)
	plt.xlabel(data_name)
	plt.ylabel('# stacks')
	plt.title('Distribution of %s across stacks'%data_name)
	plt.savefig(os.path.join(dir_to_save_to,'%s_hist'%data_name))
	plt.close()



# TODO set the source and save directory
data_source_prefix_dir=os.path.join('../../../../../d/datasets_for_iqa/')
source_path=os.path.join(data_source_prefix_dir,'iqa_data_source',
						'reorganized_combined_dataset')

dir_to_save_to=os.path.join('.','dataset_analysis')
if os.path.isdir(dir_to_save_to):
	shutil.rmtree(dir_to_save_to)
os.mkdir(dir_to_save_to)

frac_contaminated_stacks_by_subject=[]
num_bad_slices_by_subject=[]
num_roi_slices_by_subject=[]
num_contaminated_stacks_by_subject=[]
num_diagnostic_stacks=[]
num_stacks_by_subject=[]


subject_names=[]
for subject_data_path in os.listdir(source_path)[:]:
	subject_data=ds.get_subject_data(os.path.join(source_path,subject_data_path))
	subject_data=subject_data.get_subject_data_with_brain_stacks()

	subject_stack_profile_fig=ds.plot_subject_stack_distribution(subject_data)
	plt.savefig(os.path.join(dir_to_save_to,'%s_stack_profile'%subject_data.subject_folder_name))
	plt.close()
	
	# subject_data_fig=ds.plot_subject_stack_distribution(subject_data)
	# subject_data_fig.savefig(os.path.join(source_path,subject_data_path,'stack_distribution'))
	# plt.close()
	
	subject_names.append(subject_data.subject_folder_name)
	frac_contaminated_stacks_by_subject.extend(subject_data.get_fraction_contamination_per_stack())
	num_diagnostic_stacks.append(subject_data.num_stacks-subject_data.get_number_contaminated_stacks())
	num_contaminated_stacks_by_subject.append(subject_data.get_number_contaminated_stacks())
	num_stacks_by_subject.append(subject_data.num_stacks)
	num_bad_slices_by_subject.append(subject_data.get_number_bad_slices())
	num_roi_slices_by_subject.append(subject_data.get_number_roi_slices())


frac_contaminated_stacks_by_subject=np.array(frac_contaminated_stacks_by_subject)
num_contaminated_stacks_by_subject=np.array(num_contaminated_stacks_by_subject)
num_stacks_by_subject=np.array(num_stacks_by_subject)
num_diagnostic_stacks=np.array(num_diagnostic_stacks)


num_bad_slices_by_subject=np.array(num_bad_slices_by_subject)
num_roi_slices_by_subject=np.array(num_roi_slices_by_subject)
subject_names=np.array(subject_names)

"""
plot_bad_vs_good_by_subject(num_bad_slices_by_subject,num_roi_slices_by_subject,subject_names,
								dir_to_save_to,'slices')

plot_bad_vs_good_by_subject(num_contaminated_stacks_by_subject,num_stacks_by_subject,subject_names,
								dir_to_save_to,'stacks')

"""

"""
num_stacks=np.sum(num_stacks_by_subject)
stack_indices=np.arange(num_stacks)
plot_distribution_across_stacks(num_bad_slices_stack_by_subject,
								stack_indices,
								'num_bad_slices_per_stack',
								dir_to_save_to,
								True)

plot_distribution_across_stacks(num_roi_slices_stack_by_subject,
								stack_indices,
								'num_roi_slices_per_stack',
								dir_to_save_to,
								True)

plot_distribution_across_stacks(frac_contaminated_stacks_by_subject,
								stack_indices,
								'fraction_slices_bad_per_stack',
								dir_to_save_to,
								True)

"""

frac_bad_slices_by_subject=num_bad_slices_by_subject*1.0/num_roi_slices_by_subject
plot_distribution(frac_bad_slices_by_subject,subject_names,'frac_bad_slices',
				dir_to_save_to,True)
plot_distribution(num_bad_slices_by_subject,subject_names,'num_bad_slices',dir_to_save_to,True)


"""
# analyzing bad slices 
total_num_bad_slices=np.sum(num_bad_slices_by_subject)
#num_bad_slices_by_subject=num_bad_slices_by_subject*1.0/total_num_bad_slices
ind_ordered_increasing_fraction=np.argsort(num_bad_slices_by_subject)
num_bad_slices_by_subject_ordered=num_bad_slices_by_subject[ind_ordered_increasing_fraction]
subject_names_ordered=subject_names[ind_ordered_increasing_fraction]

fig=plt.figure()
ax=fig.add_subplot(111)
width=0.4

num_subjects=len(num_bad_slices_by_subject)

for subject_index in range(num_subjects):
	ax.bar(subject_index,num_bad_slices_by_subject_ordered[subject_index],width=width)


xleft=-1*width
ax.set_xlim(left=xleft,right=len(subject_names)+1)
ax.set_xticks(np.arange(num_subjects))    
ax.set_xticklabels(subject_names_ordered,rotation=90)
ax.tick_params(axis='x',labelsize=5)
ax.set_ylabel('total bad slices among (%d)'%total_num_bad_slices)
ax.set_title('Distribution of bad slices across subjects')
fig.savefig('bad_slice_distrib')
plt.close()


plt.figure()
bins=np.arange(70)
plt.hist(num_bad_slices_by_subject,bins=bins,density=False)
plt.xlabel('# bad slices')
bin_loc=np.arange(70,step=5)
plt.xticks(bin_loc)
plt.ylabel('# subjects')
plt.title('Distribution of bad slices across subjects')
plt.savefig(os.path.join('bad_slice_hist'))
plt.close()


# distribution over total ROI slices
total_num_roi_slices=np.sum(num_roi_slices_by_subject)
ind_ordered_increasing_fraction=np.argsort(num_roi_slices_by_subject)
num_roi_slices_by_subject_ordered=num_roi_slices_by_subject[ind_ordered_increasing_fraction]
subject_names_ordered=subject_names[ind_ordered_increasing_fraction]

fig=plt.figure()
ax=fig.add_subplot(111)
width=0.4

num_subjects=len(num_bad_slices_by_subject)

for subject_index in range(num_subjects):
	ax.bar(subject_index,num_roi_slices_by_subject_ordered[subject_index],width=width)


xleft=-1*width
ax.set_xlim(left=xleft,right=len(subject_names)+1)
ax.set_xticks(np.arange(num_subjects))    
ax.set_xticklabels(subject_names_ordered,rotation=90)
ax.tick_params(axis='x',labelsize=5)
ax.set_ylabel('total roi slices among (%d)'%total_num_bad_slices)
ax.set_title('Distribution of roi slices across subjects')
fig.savefig('roi_slice_distrib')
plt.close()


plt.figure()
plt.hist(num_roi_slices_by_subject,density=False)
plt.xlabel('# roi slices')
plt.ylabel('# subjects')
plt.title('Distribution of roi slices across subjects')
plt.savefig(os.path.join('roi_slice_hist'))
plt.close()


# plot fraction of bad 
num_bad_slices_by_subject=np.array(num_bad_slices_by_subject)
num_roi_slices_by_subject=np.array(num_roi_slices_by_subject)
ratio_bad_roi_by_subject=num_bad_slices_by_subject*1.0/num_roi_slices_by_subject

ind_ordered_increasing_fraction=np.argsort(ratio_bad_roi_by_subject)
ratio_bad_roi_by_subject_ordered=ratio_bad_roi_by_subject[ind_ordered_increasing_fraction]
subject_names_ordered=subject_names[ind_ordered_increasing_fraction]



fig=plt.figure()
ax=fig.add_subplot(111)
width=0.4

num_subjects=len(ratio_bad_roi_by_subject)

for subject_index in range(num_subjects):
	ax.bar(subject_index,ratio_bad_roi_by_subject_ordered[subject_index],width=width)


xleft=-1*width
ax.set_xlim(left=xleft,right=len(subject_names)+1)
ax.set_xticks(np.arange(num_subjects))    
ax.set_xticklabels(subject_names_ordered,rotation=90)
ax.tick_params(axis='x',labelsize=5)
ax.set_ylabel('fraction_bad (among the subject\'s roi slices)')
ax.set_title('Distribution of fraction of bads across subjects')
fig.savefig('bad_frac_distrib')
plt.close()

plt.figure()
plt.hist(ratio_bad_roi_by_subject,density=False)
plt.xlabel('fraction_bad (among the subject\'s roi slices)')
plt.ylabel('# subjects')
plt.title('Distribution of fraction of bads across subjects')
plt.savefig(os.path.join('bad_frac_hist'))
plt.close()

# num_bad_slices_by_subject=np.array(num_bad_slices_by_subject)
# plt.hist(num_bad_slices_by_subject,density=True)
# plt.xlabel("# bad slices")
# plt.ylabel('subject fraction')
# plt.title('Distribution of bad slices')
# plt.savefig('bad_slice_distrib')
"""

"""
subject_names=[]
num_brain_stacks_by_subject=[]
num_stacks_by_subject=[]
for subject_data_path in os.listdir(source_path):
	print("subject_data_path ", subject_data_path)
	subject_data=ds.get_subject_data(os.path.join(source_path,subject_data_path))
	subject_data_brain_stacks=subject_data.get_brain_stacks()
	num_brain_stacks_by_subject.append(len(subject_data_brain_stacks))
	num_stacks_by_subject.append(subject_data.num_stacks)
	subject_names.append(subject_data.subject_name)


subject_names=np.array(subject_names)

# plot stacks by name to see if there is any correspondence between 
plt.figure()
plt.bar(np.arange(len(subject_names)),num_brain_stacks_by_subject)
plt.xlabel("subject ids")
plt.ylabel("# brain stacks")
plt.xticks(np.arange(len(subject_names)),subject_names,
					rotation='vertical',fontsize='xx-small')
plt.title('Number of Fetal Brain HASTE stacks')
plt.savefig('num_brain_stacks_by_subject')
plt.close()



num_brain_stacks_by_subject=np.array(num_brain_stacks_by_subject)
plt.figure()
bins=np.arange(20)
plt.hist(num_brain_stacks_by_subject,bins=bins,density=False)
plt.xlabel('# brain stacks')
plt.xticks(bins)
plt.ylabel('subject fraction')
plt.title('Distribution of Fetal Brain HASTE stacks')
plt.savefig(os.path.join('num_brain_stacks'))
plt.close()

num_stacks_by_subject=np.array(num_stacks_by_subject)
bins=np.arange(25)
plt.figure()
plt.hist(num_stacks_by_subject,bins=bins,density=False)
plt.xlabel('# stacks')
plt.xticks(bins)
plt.ylabel('subject fraction')
plt.title('Distribution of Fetal HASTE stacks')
plt.savefig(os.path.join('num_stacks'))
plt.close(0)
"""