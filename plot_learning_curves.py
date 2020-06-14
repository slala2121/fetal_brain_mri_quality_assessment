import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser()

    # what problem to solve
    parser.add_argument('--learning_rates',nargs="+",default=[])
    parser.add_argument('--res_path',default='.')
    parser.add_argument('--lr_path_prefix',default="")
    parser.add_argument('--save_path_prefix',default="")
   
    # parameters
    args=parser.parse_args()
    learning_rates=args.learning_rates
    res_path=args.res_path.strip()
    lr_path_prefix=args.lr_path_prefix
    save_path_prefix=args.save_path_prefix


    
    plt.figure()
    index=1
    for lr in learning_rates:
    	
    	# get results
    	lr_path=lr_path_prefix+'_%s' %lr
        print(lr_path)
    	train_loss=np.load(os.path.join(lr_path,'final','train_loss.npy'))
    	epoch_indices=np.arange(len(train_loss))

    	plt.subplot(2,3,index)
    	plt.plot(epoch_indices,train_loss)
    	plt.title('Learning rate %s' %lr)
    	
    	index+=1



    plt.tight_layout()
    plt.savefig(os.path.join(res_path,'%s_compare_lr' %save_path_prefix))
    plt.close("all")

    plt.figure()
    for lr in learning_rates:
    	# get results
    	lr_path=lr_path_prefix+'_%s' %lr
    	train_loss=np.load(os.path.join(lr_path,'final','train_loss.npy'))
    	epoch_indices=np.arange(len(train_loss))
    	plt.plot(epoch_indices,train_loss,label=lr)

    plt.legend(loc='upper right')
    plt.savefig(os.path.join(res_path,'%s_compare_lr_combined'%save_path_prefix))
    plt.close('all')



