import numpy as np

def main():
	rgb_path = '../data/v_CricketShot_g04_c01_rgb.npy'
	rgb = np.load(rgb_path)
	print rgb.shape

if __name__=='__main__':
	main()