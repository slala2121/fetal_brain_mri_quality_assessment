library(ROCR)
library(binom)
library(pROC)
library(boot)

rm(list=ls())



# set the source_path which contains results and will be used for saving ROC plot
# setwd("d:/")
# source_path=file.path("d:","exp_results","image_quality","images_labeled_dataset",
#                       "reorganized_dataset")

setwd("c:/")
source_path=file.path("c:/","Users","ls2121","Desktop")


# # load the predictions, labels
# methods_list=list("2_vgg_16_transfer_lr_0.000025")
# methods_name=list("transfer_vgg_16")
# num_methods=length(methods_list)
# preds_list=vector("list",num_methods)
# labels_path=file.path("d:","datasets_for_iqa","iqa_data_source","dataset_partition","complete","test_labels.csv")
# true_labels=read.csv(file.path(labels_path),header=FALSE) 
# true_labels=unlist(true_labels)
# 
# 
# source_path=file.path("d:","exp_results","image_quality","images_labeled_dataset",
#                       "full_dataset/class_weighted_loss")
# methods_list=list("class_weight_0","class_weight_1")
# methods_name=list("vgg_16_no_class_weighted_loss","vgg_16_with_class_weighted_loss")


methods_list=list("vgg_test")
methods_name=list("vgg_test")
num_methods=length(methods_list)
preds_list=vector("list",num_methods)
labels_path=file.path(source_path,"test_labels.csv")
true_labels=read.csv(file.path(labels_path),header=FALSE) 
true_labels=unlist(true_labels)




# for (i in seq(from=1,to=num_methods,by=1)) {
#   
#   # ground_truth_labels=read.csv(file.path(source_path,methods[[i]],'best','test_labels.csv'),header=FALSE)
#   # ground_truth_labels=unlist(ground_truth_labels)
#   # true_labels_list[[i]]=ground_truth_labels
#   
#   preds_method=read.csv(file.path(source_path,methods_list[[i]],'best','test_preds.csv'),header=FALSE)
#   preds_method=unlist(preds_method)
#   preds_list[[i]]=preds_method
# }


preds_method=read.csv(file.path(source_path,'original','test_preds.csv'),header=FALSE)
preds_method=unlist(preds_method)
preds_list[[1]]=preds_method



#compute CI intervals
ci_auc=vector("list",num_methods)
ci_se=vector("list",num_methods)
ci_sp=vector("list",num_methods)
threshold=0.5
ci_se_threshold=vector("list",num_methods)
ci_sp_threshold=vector("list",num_methods)
ci_acc_threshold=vector("list",num_methods)
for (i in seq(from=1,to=num_methods,by=1)) {
	preds_method=preds_list[[i]]
	roc_method=smooth(roc(true_labels,preds_method))
	ci_auc[[i]]=ci.auc(roc_method)
	ci_se[[i]]=ci.se(roc_method)
	ci_sp[[i]]=ci.sp(roc_method)
	
	num_tp<-length(which(true_labels==1 & preds_method>=threshold))
	num_total_pos<-length(which(true_labels==1))
	sens<-binom.confint(x=num_tp,n=num_total_pos,methods='wilson')[4:6]
	ci_se_threshold[[i]]=sens
	
	num_tn<-length(which(true_labels==0 & preds_method<threshold))
	num_total_neg<-length(which(true_labels==0))
	spec<-binom.confint(x=num_tn,n=num_total_neg,methods='wilson')[4:6]
	ci_sp_threshold[[i]]=spec

	num_correct<-num_tp+num_tn
	num_total<-length(true_labels)
	acc<-binom.confint(x=num_correct,n=num_total,methods='wilson')[4:6]
	ci_acc_threshold[[i]]=acc
}


roc_objects_list=list()
display_legend=TRUE
file_name='compare_rocs'
pdf(file.path(source_path,paste(file_name,'.pdf',sep='')))
par(bg="black")
alpha=0.55
line_colors_list=c(rgb(89, 219, 212,max=255),
                   rgb(255,0,0,255,max=255),rgb(0,255,0,255,max=255),rgb(0,0,255,255,max=255),
                   rgb(255,255,255,max=255),rgb(241,66,244,max=255),rgb(244,143,66,max=255))
ci_colors_list=c(rgb(89, 219, 212,max=255),rgb(255,0,0,alpha*255,max=255),rgb(0,255,0,alpha*255,max=255),
                 rgb(0,0,255,alpha*255,max=255),rgb(255,255,255,alpha*255,max=255),
                 rgb(241,66,244,alpha*255,max=255),rgb(244,143,66,alpha*255,max=255))
LWD=7
LWD_CI=3

# method_indices=list(1,2)
method_indices=seq(from=1,to=num_methods)

for (i in seq(from=1,to=length(method_indices),by=1)) {
  method_index=method_indices[[i]]
  
	roc_obj=smooth(roc(true_labels,preds_list[[method_index]]))
	roc_objects_list[[i]]=roc_obj
	if (i==1) {
		plot(roc_obj,col=line_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=LWD,
		main='Test Set Receiver Operator Characteristic Curves',xlab='Accuracy on Diagnostic Images',ylab='Accuracy on Non-Diagnostic Images',
		cex.lab=1.5,cex.axis=1.5,cex.main=1.5,col.lab="white",col.axis="white",col.main="white")
		plot(ci_se[[i]],add=TRUE,type="bar",col=ci_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=3)
		plot(ci_sp[[i]],add=TRUE,type="bar",col=ci_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=3)	
	} else {
		plot(roc_obj,add=TRUE,col=line_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=LWD)
		plot(ci_se[[i]],type='bar',add=TRUE,col=ci_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=3)
		plot(ci_sp[[i]],type="bar",add=TRUE,col=ci_colors_list[[i]],xlim=c(1.0,0.0),ylim=c(0.0,1.0),lwd=3)
	}	
}



if (display_legend) {
  legend_methods=vector("list",length(method_indices))
  for (i in seq(from=1,to=length(method_indices),by=1)) {
    method_index=method_indices[[i]]
    legend_methods[[i]]=paste(methods_name[[method_index]],"- AUC:", signif(ci_auc[[method_index]][2],digits=2),
                              ", 95% CI:", signif(ci_auc[[method_index]][1],digits=2),"-",signif(ci_auc[[method_index]][3],digits=2),sep=" ")
  }
  
  legend(x='bottomright',legend= legend_methods,fill=line_colors_list,cex=1,pt.cex=1,
         text.col="white",col="black")
}




# for (i in seq(from=1,to=length(method_indices),by=1)) {
#   method_index=method_indices[[i]]
# 	x=ci_sp_threshold[[method_index]][[1]]
# 	y=ci_se_threshold[[method_index]][[1]]
# 	points(x,y,col=line_colors_list[i],pch=18,cex=2)
# }

# points(1.0,1.0,col='red',pch=18,cex=2)

LWD=2.5

# draw the baseline and ideal test curve for reference
ideal_curve_y=seq(from=0,to=1,by=0.2)
ideal_curve_x=rep(1,length(ideal_curve_y))
lines(ideal_curve_x,ideal_curve_y,col='gray',type="l",lty='solid')
ideal_curve_x=seq(from=0,to=1,by=0.2)
ideal_curve_y=rep(1,length(ideal_curve_x))
lines(ideal_curve_x,ideal_curve_y,col='gray',type="l",lty='solid')

# random guess baseline 
baseline_curve_x=seq(from=1,to=0,by=-0.2)
baseline_curve_y=seq(from=0,to=1,by=0.2)
lines(baseline_curve_x,baseline_curve_y,col='gray',type="l",lty='solid')

dev.off()

# compare roc curves and get p-value
# hypothesis_testing=roc.test(roc_objects_list[[1]],roc_objects_list[[3]])
# p_value=hypothesis_testing[[7]]
# print(p_value)


# # generate baseline ROC curves for reference 
# display_legend=TRUE
# file_name='baseline_rocs'
# pdf(file.path(source_path,paste(file_name,'.pdf',sep='')))
# par(bg="black")
# LWD=2.5
# 
# 
# plot(ideal_curve_x,ideal_curve_y,type="n",
#      xlim=c(1.0,0.0),ylim=c(0.0,1.0),
#      main='Test Set Receiver Operator Characteristic Curves',xlab='Accuracy on Diagnostic Images',ylab='Accuracy on Non-Diagnostic Images',
#      cex.lab=1.5,cex.axis=1.5,cex.main=1.5,col.lab="white",col.axis="white",col.main="white")
# ideal_curve_y=seq(from=0,to=1,by=0.2)
# ideal_curve_x=rep(1,length(ideal_curve_y))
# lines(ideal_curve_x,ideal_curve_y,col='gray',type="l",lty='solid')
# ideal_curve_x=seq(from=0,to=1,by=0.2)
# ideal_curve_y=rep(1,length(ideal_curve_x))
# lines(ideal_curve_x,ideal_curve_y,col='gray',type="l",lty='solid')
# baseline_curve_x=seq(from=1,to=0,by=-0.2)
# baseline_curve_y=seq(from=0,to=1,by=0.2)
# lines(baseline_curve_x,baseline_curve_y,col='gray',type="l",lty='solid')
# 
# 
# 
# 
# dev.off()



