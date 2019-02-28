import h5py
import numpy as np
import os


def main():
	
	filename = 'data/with_det_scores_test.h5'
	f = h5py.File(filename, 'r')

	labels = f['labels']
	print type(labels)
	print labels.shape
	print labels[0]

	file_curr = '../scratch/voc_2007_test_scores/output_prod/1.npy'
	pred_score = np.load(file_curr)
	print pred_score.shape
	print np.min(pred_score), np.max(pred_score)
	file_curr = '../scratch/voc_2007_test_scores/output_softmax/1.npy'
	pred_score = np.load(file_curr)
	print pred_score.shape
	print np.min(pred_score[0],axis= 1), np.max(pred_score[0],axis = 1)

	det_scores = f['outputs']['output_softmax']
	pred_scores = f['outputs']['output_prod']

	print len(det_scores)
	for det_score in det_scores:
		print type(det_score)
		print det_score
		raw_input()


	for idx_test in range(len(labels)):
		labels_curr = labels[idx_test]
		det_scores_curr = det_scores[idx_test]
		pred_scores_curr = pred_scores[idx_test]

		print labels_curr.shape, det_scores_curr.shape, pred_scores_curr.shape
		raw_input()
	
	# print len(det_scores)
	# print len(pred_scores)

	# print  f['outputs'].keys()

	# print labels.shape
	# print len(outputs)


	# List all groups
	print("Keys: %s" % f.keys())	

if __name__=='__main__':
	main()
