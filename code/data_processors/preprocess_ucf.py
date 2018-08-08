# import cv2
import sys
sys.path.append('./')
import os
from helpers import util, visualize
import numpy as np
import glob
import scipy.misc
import scipy.stats
import multiprocessing


def compare_rgb(old_rgb,new_rgb_folder):
	old_rgb = np.load(old_rgb)
	new_rgb = make_rgb_np(new_rgb_folder,224)
	assert new_rgb.shape[1]==old_rgb.shape[1]

	# print len(new_rgb_paths)
	print old_rgb.shape

	
	print old_rgb.shape, new_rgb.shape
	print np.min(old_rgb), np.max(old_rgb),np.mean(old_rgb)
	print np.min(new_rgb), np.max(new_rgb),np.mean(new_rgb)

	diff_mat = np.abs(new_rgb-old_rgb)
	diff_mat = np.around(diff_mat,3)
	uni_val, counts = np.unique(diff_mat,return_counts = True)
	mode = uni_val[np.argmax(counts)]
	print counts[np.argmax(counts)]

	print np.min(diff_mat), np.max(diff_mat), np.mean(diff_mat), mode


def compare_flo(old_flo, new_flo_folders):
	old_flo = np.load(old_flo)
	new_flo = make_flo_np(new_flo_folders[0],new_flo_folders[1], 224)

	assert new_flo.shape[1]==old_flo.shape[1]
	print old_flo.shape, new_flo.shape
	print np.min(old_flo), np.max(old_flo),np.mean(old_flo)
	print np.min(new_flo), np.max(new_flo),np.mean(new_flo)

	diff_mat = np.abs(new_flo-old_flo)
	diff_mat = np.around(diff_mat,5)
	uni_val, counts = np.unique(diff_mat,return_counts = True)
	mode = uni_val[np.argmax(counts)]
	print counts[np.argmax(counts)]

	print np.min(diff_mat), np.max(diff_mat), np.mean(diff_mat), mode

def make_flo_np(dir_flo_u,dir_flo_v,crop_size):
	new_flo_paths = glob.glob(os.path.join(dir_flo_u,'*.jpg'))
	new_flo_paths.sort()
	
	new_flo = np.zeros((1,len(new_flo_paths),crop_size,crop_size,2))
	for idx_im_curr, flo_u in enumerate(new_flo_paths):
		# print idx_im_curr
		flo_v = os.path.join(dir_flo_v,os.path.split(flo_u)[1])
		assert os.path.exists(flo_v)
		new_flo[0,idx_im_curr]= preprocess_flo(flo_u,flo_v)
	return new_flo


def make_rgb_np(new_rgb_folder,crop_size):
	new_rgb_paths = glob.glob(os.path.join(new_rgb_folder,'*.jpg'))
	new_rgb_paths.sort()
	# new_rgb_paths = new_rgb_paths
	# [:-1]
	new_rgb = np.zeros((1,len(new_rgb_paths),crop_size,crop_size,3))
	for idx_im_curr, im_curr in enumerate(new_rgb_paths):
		new_rgb[0,idx_im_curr]= preprocess_rgb(im_curr)

	return new_rgb


def preprocess_flo(flo_u,flo_v,crop_size = 224):
	u_im = scipy.misc.imread(flo_u).astype(float)[:,:,np.newaxis]
	v_im = scipy.misc.imread(flo_v).astype(float)[:,:,np.newaxis]
	flo = np.concatenate([u_im,v_im],2)
	
	start_row = (flo.shape[0]-crop_size)//2 
	start_col = (flo.shape[1]-crop_size)//2 

	flo_crop = flo[start_row:start_row+crop_size,start_col:start_col+crop_size]
	flo_crop = flo_crop/255. *2 -1
	return flo_crop

def preprocess_rgb(im_path,crop_size = 224):
	im = scipy.misc.imread(im_path).astype(float)
	
	start_row = (im.shape[0]-crop_size)//2 
	start_col = (im.shape[1]-crop_size)//2 

	im_crop = im[start_row:start_row+crop_size,start_col:start_col+crop_size]

	im_crop = im_crop/255 *2 -1
	return im_crop

def save_numpy((video, dir_rgb, dir_flos, crop_size, out_file, idx_video)):
	if idx_video%10==0:
		print idx_video

	try:
		rgb_dir = os.path.join(dir_rgb,video)
		u_dir = os.path.join(dir_flos[0], video)
		v_dir = os.path.join(dir_flos[1], video)
		assert os.path.exists(u_dir)
		assert os.path.exists(v_dir)
		assert os.path.exists(rgb_dir)

		rgb_np = make_rgb_np(rgb_dir, crop_size)
		flo_np = make_flo_np(u_dir, v_dir, crop_size)
		# print rgb_np.shape, flo_np.shape

		assert rgb_np.shape[1]>=flo_np.shape[1]

		if rgb_np.shape[1]> flo_np.shape[1]:
			rgb_np = rgb_np[:,:flo_np.shape[1],:,:]

		assert np.all(rgb_np.shape[:-1]==flo_np.shape[:-1])

		# raw_input()
		# print out_file
		np.savez(out_file, rgb= rgb_np, flo=flo_np)
	except:
		print 'ERROR', out_file, dir_rgb

def get_numpys(video, dir_rgb, dir_flos, crop_size):
	rgb_dir = os.path.join(dir_rgb,video)
	u_dir = os.path.join(dir_flos[0], video)
	v_dir = os.path.join(dir_flos[1], video)
	assert os.path.exists(u_dir)
	assert os.path.exists(v_dir)
	assert os.path.exists(rgb_dir)

	rgb_np = make_rgb_np(rgb_dir, crop_size)
	flo_np = make_flo_np(u_dir, v_dir, crop_size)
	# print rgb_np.shape, flo_np.shape

	assert rgb_np.shape[1]>=flo_np.shape[1]

	if rgb_np.shape[1]> flo_np.shape[1]:
		rgb_np = rgb_np[:,:flo_np.shape[1],:,:]

	assert np.all(rgb_np.shape[:-1]==flo_np.shape[:-1])
	return rgb_np, flo_np



def script_save_numpys():
	dir_meta = '../data/ucf101'
	
	out_dir = os.path.join(dir_meta,'npys')
	util.mkdir(out_dir)

	dir_rgb = os.path.join(dir_meta, 'rgb_ziss/jpegs_256')

	dir_flos = os.path.join(dir_meta,'flow_ziss/tvl1_flow')
	dir_flos = [os.path.join(dir_flos,'u'),os.path.join(dir_flos,'v')]

	videos = [os.path.split(dir_curr)[1] for dir_curr in glob.glob(os.path.join(dir_rgb,'*')) if os.path.isdir(dir_curr)]
	print len(videos)

	args = []
	for idx_video, video in enumerate(videos):
		out_file = os.path.join(out_dir,video+'.npz')
		if os.path.exists(out_file):
			continue

		arg_curr = (video, dir_rgb, dir_flos, 224, out_file, idx_video)
		args.append(arg_curr)

	print len(args)

	# for arg_curr in args:
	# 	save_numpy(arg_curr)
	# 	break

	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(save_numpy,args)


def redo_numpys():
	out_dir = '../data/ucf101/npys'
	all_numpys = glob.glob(os.path.join(out_dir,'*.npz'))[:10]
	problem_numpys = []
	problem_count = 0
	
	for idx_np_curr, np_curr in enumerate(all_numpys):
		
		# if idx_np_curr%100==0:
		print idx_np_curr

		# try:
		# print 'no problem'
		data = np.load(np_curr)
			# np.savez_compressed(np_curr, rgb= data['rgb'], flo=data['flo'])
		# except:
			# print 'problem'
			# problem_count+=1
			# problem_numpys.append(np_curr)


	print len(all_numpys), problem_count
	out_file = 'problem_npys.txt'
	util.writeFile(out_file, problem_npys)




def main():
	redo_numpys();
	# script_save_numpys()
	# npz = np.load('../data/ucf101/npys/v_RopeClimbing_g06_c01.npz')
	# rgb = npz['rgb']
	# flo= npz['flo']
	# print rgb.shape, np.min(rgb), np.max(rgb)
	# print flo.shape, np.min(flo), np.max(flo)

	return 
	out_dir = '../scratch/checking_ucf'
	util.mkdir(out_dir)

	old_rgb = '../kinetics-i3d-master/data/v_CricketShot_g04_c01_rgb.npy'
	old_flo = '../kinetics-i3d-master/data/v_CricketShot_g04_c01_flow.npy'
	video_name = 'v_CricketShot_g04_c01'
	dir_rgb = '../data/ucf101/rgb_ziss/jpegs_256'
	dir_flo = '../data/ucf101/flow_ziss/tvl1_flow'
	dir_flos = [os.path.join(dir_flo,'u'),os.path.join(dir_flo,'v')]

	new_rgb_folder = os.path.join(dir_rgb,video_name)
	new_flo_folders = [os.path.join(dir_curr,video_name) for dir_curr in dir_flos]
	# compare_rgb(old_rgb, new_rgb_folder)
	# compare_flo(old_flo, new_flo_folders)

	new_flo = make_flo_np(new_flo_folders, 224)
	print new_flo.shape, np.min(new_flo), np.max(new_flo), np.mean(new_flo)
	new_rgb = make_rgb_np(new_rgb_folder, 224)
	print new_rgb.shape, np.min(new_rgb), np.max(new_rgb), np.mean(new_rgb)

	np.save(os.path.join(out_dir,'new_rgb.npy'),new_rgb)
	np.save(os.path.join(out_dir,'new_flo.npy'),new_flo)


	return 
	# out_dir = '../../scratch/kin_look_at'
	rgb_path = '../kinetics-i3d-master/data/v_CricketShot_g04_c01_rgb.npy'
	flow_path = '../kinetics-i3d-master/data/v_CricketShot_g04_c01_flow.npy'

	dir_flow = '../data/ucf101/flow_ziss/tvl1_flow'


	im_paths = glob.glob(os.path.join(dir_rgb,video_name,'*.jpg'))
	im_paths.sort()
	print im_paths[0]

	u_flow_paths = glob.glob(os.path.join(dir_flow,'u',video_name,'*'))
	v_flow_paths = glob.glob(os.path.join(dir_flow,'v',video_name,'*'))
	u_flow_paths.sort()
	v_flow_paths.sort()
	print u_flow_paths[0],v_flow_paths[0]

	u_im = scipy.misc.imread(u_flow_paths[0]).astype(float)[:,:,np.newaxis]
	v_im = scipy.misc.imread(v_flow_paths[1]).astype(float)[:,:,np.newaxis]
	flo = np.concatenate([u_im,v_im],2)
	# flo = flo/255. 

	start_row = (flo.shape[0]-224)//2 
	start_col = (flo.shape[1]-224)//2 

	flo_crop = flo[start_row:start_row+224,start_col:start_col+224]
	flo_crop = np.concatenate([flo_crop,128*np.ones((flo_crop.shape[0],flo_crop.shape[1],1))], 2)
	print flo_crop.shape
	print np.min(flo_crop),np.max(flo_crop)

	
	flo = np.load(flow_path)
	print 'min all',np.min(flo),np.max(flo)
	flo_check = flo[0,0] 
	flo_check = flo_check+0.5
	flo_check = np.concatenate([flo_check,0.5*np.ones((flo_check.shape[0],flo_check.shape[1],1))], 2) * 255

	print flo_check.shape
	print np.min(flo_check),np.max(flo_check)


	diff = np.abs(flo_check - flo_crop)
	print np.min(diff),np.max(diff)
	print 'hello'
	
	scipy.misc.imsave('../scratch/flo_diff.jpg',diff)
	scipy.misc.imsave('../scratch/flo_npy.jpg',flo_check)
	scipy.misc.imsave('../scratch/flo_crop.jpg',flo_crop)

	visualize.writeHTMLForFolder('../scratch')

	return

	im = scipy.misc.imread(im_paths[0]).astype(float)
	
	start_row = (im.shape[0]-224)//2 
	start_col = (im.shape[1]-224)//2 

	im_crop = im[start_row:start_row+224,start_col:start_col+224]

	# im_crop = im_crop/255 *2 -1

	print im_crop.shape, np.min(im_crop),np.max(im_crop)


	# print im.shape,np.min(im),np.max(im)


	rgb = np.load(rgb_path)

	im_check = rgb[0,0]
	print im_check.shape
	im_check = (im_check+1)/2. * 255.
	print np.allclose(im_crop,im_check)
	diff = np.abs(im_check - im_crop)
	print np.min(diff),np.max(diff)
	print 'hello'
	# print diff[:,:,0]


	# flow = np.load(flow_path)

	# print rgb.shape
	# print np.min(rgb)
	# print np.max(rgb)

	# print flow.shape
	# print np.min(flow)
	# print np.max(flow)

	single_frame = (rgb[0,0,:,:]+1)/2. *255.
	print np.min(single_frame),np.max(single_frame)
	scipy.misc.imsave('../scratch/diff.jpg',diff)
	scipy.misc.imsave('../scratch/npy.jpg',single_frame)
	scipy.misc.imsave('../scratch/crop.jpg',im_crop)





	# out_file = os.path.join(,'first.jpg')
	# print rgb.shape

if __name__=='__main__':
	main()