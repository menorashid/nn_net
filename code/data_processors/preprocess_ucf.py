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
import subprocess


dir_meta = '../data/ucf101'

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


def extract_frames((input_path,out_dir,fps,out_res,idx)):
	# command = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of default=nw=1:nk=1 '
	# command = command+input_path
	# ret_vals = os.popen(command).readlines()
	
	# ret_vals = [int(val.strip('\n')) for val in ret_vals]
	
	# if ret_vals[0]>ret_vals[1]:
	# 	pass
	# else:
	# 	print 'width bigger!!!'
	print idx
	util.mkdir(out_dir)
	command = ['ffmpeg']

	command += ['-i',input_path]

	command += ['-r' , str(fps)]
	command += ['-vf' ,'scale='+ '-1:'+str(out_res)]
	command += [os.path.join(out_dir,'frame%06d.jpg')]
	command += ['-hide_banner']
	command += ['> /dev/null 2> err.txt']
	command = ' '.join(command)

	# print command
	subprocess.call(command,shell=True)


def checking_frame_extraction(in_dir=None,out_dir=None, small_dim = 256, fps = 10):
	
	if in_dir is None:
		in_dir = os.path.join(dir_meta,'val_data','validation')

	if out_dir is None:
		out_dir = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))

	util.mkdir(out_dir)

	in_files = glob.glob(os.path.join(in_dir,'*.mp4'))
	in_files.sort()

	print len(in_files)
	
	out_file_problem = os.path.join(out_dir,'problem_files.txt')

	problem = []
	for idx_in_file, in_file in enumerate(in_files):
		print idx_in_file
		out_dir_curr = os.path.split(in_file)[1]
		out_dir_curr = out_dir_curr[:out_dir_curr.rindex('.')]
		out_dir_curr = os.path.join(out_dir,out_dir_curr)

		command = ['ffmpeg', '-i']
		command += [in_file, '2>&1']
		command += ['|','grep "Duration"']
		command += ['|', 'cut -d', "' '",'-f 4']
		command += ['|', 'sed s/,//']
		# command += ['|', 'sed','s@\..*@@g' ]
		# command += ['|', 'awk',"'{", 'split($1, A, ":");', 'split(A[3], B, ".");', 'print 3600*A[1] + 60*A[2] + B[1]', "}'"]
		command = ' '.join(command)
		# print command

		secs = os.popen(command).readlines()
		try:
			secs = secs[0].strip('\n')
			secs = secs.split(':')
			assert len(secs)==3
			secs = secs[:-1] + secs[-1].split('.')[:1]
			secs = int(secs[0])*3600+int(secs[1])*60+int(secs[2])
			# print secs
			num_frames = secs*fps

			num_frames_ac = len(glob.glob(os.path.join(out_dir_curr,'*.jpg')))
			if (num_frames_ac-num_frames)<0:
				print 'PROBLEM',num_frames_ac,num_frames
				problem.append(' '.join([in_file,str(num_frames_ac),str(num_frames)]))
		except:
			print 'PROBLEM SERIOUS',in_file
			problem.append(in_file)

	util.writeFile(out_file_problem,problem)

		# break
		

def script_extract_frames(in_dir=None,out_dir=None, small_dim = 256, fps = 10):
	
	if in_dir is None:
		in_dir = os.path.join(dir_meta,'val_data','validation')

	if out_dir is None:
		out_dir = os.path.join(dir_meta,'val_data','rgb_'+str(fps)+'_fps_'+str(small_dim))

	util.mkdir(out_dir)

	in_files = glob.glob(os.path.join(in_dir,'*.mp4'))
	in_files.sort()

	print len(in_files)

	args = []
	for idx_in_file, in_file in enumerate(in_files):
		out_dir_curr = os.path.split(in_file)[1]
		out_dir_curr = out_dir_curr[:out_dir_curr.rindex('.')]
		out_dir_curr = os.path.join(out_dir,out_dir_curr)
		
		if os.path.exists(os.path.join(out_dir_curr,'frame000001.jpg')):
			continue

		args.append((in_file, out_dir_curr, fps, small_dim, idx_in_file))

	print len(args)
	# print args[0]
	# extract_frames(args[0])
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	pool.map(extract_frames, args)
	# print 'done'


def main():

	

	in_dir = os.path.join(dir_meta,'test_data','TH14_test_set_mp4')
	out_dir = os.path.join(dir_meta,'test_data','rgb_10_fps_256')

	# script_extract_frames(in_dir, out_dir)
	checking_frame_extraction(in_dir, out_dir)



if __name__=='__main__':
	main()