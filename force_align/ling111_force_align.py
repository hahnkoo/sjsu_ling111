"""Forced alignment using HMM-GMM -- A script for LING 111 Final Project Annotation #2
"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import argparse, sys, time
import numpy as np
import librosa
import hmm, trellis, baum_welch, viterbi_train, dumb_train, ling111_extract_words

def extract_features(x, fs, frame_shift, frame_size):
	"""Extract acoustic features for each frame: 13 MFCCs, 13 deltas, 13 delta-deltas."""
	c = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=13, n_fft=int(fs*frame_size), hop_length=int(fs*frame_shift), center=False)
	d = librosa.feature.delta(c, width=3, order=1, mode='constant')
	d2 = librosa.feature.delta(c, width=3, order=2, mode='constant')
	out = np.vstack([c, d, d2]).T
	return out

def create_model_dict(alphabet, gmm_config={'output_dim':39, 'n_components':2, 'diagonal':True}, sos_dict={'sos':'<s>'}, eos_dict={'eos':'</s>'}, model_type='left-to-right', n_emitting_states=1):
	"""Create a model dictionary covering all phones found in ys."""
	md = {'eos':hmm.Model('eos', 1, 'singleton', hmm.emit_final, eos_dict), 'sos':hmm.Model('sos', 1, 'singleton', hmm.emit_start, sos_dict)}
	for alph in alphabet:
		md[alph] = hmm.Model(alph, n_emitting_states, model_type, hmm.emit_GMM, gmm_config)
	return md

def align(x, y, md):
	"""Align x (audio) and y (text)."""
	mc = hmm.Chain([md[m] for m in y])
	if type(x) != list: output_sequence = list(x)
	tr = trellis.Trellis(mc, x)
	tr.build_viterbi()
	best_path = tr.decode()
	msl = []
	for p in best_path:
		entry = []
		for s in p.states:
			entry.append((mc.index[s][0], mc.index[s][1]))
		msl.append(entry)
	frame_stamps = []
	for i, path in enumerate(msl[1:]):
		for m, s in path:
			if s == 0: frame_stamps.append((m, i))
	return frame_stamps 

def get_phone_intervals(xs, ys, md, stamps, hop_length, sampling_rate):
	"""List phone intervals to include in the output textgrid file.

	Args:
	- xs: output sequences including <s> and </s>
	- ys: model name sequences including sos and eos
	- md: model dictionary
	- stamps: sorted word interval dictionary; stamps[(start_sample_index, end_sample_index)] = {'word':word, 'phones':[phones]}
	"""
	decode_output = {}
	for i in range(len(xs)):
		n_frames = len(xs[i])
		frame_stamps = align(xs[i], ys[i], md)
		stamp = stamps[i]
		sample_offset = stamp[0]
		entries = []
		for (p, fi) in frame_stamps:
			sample_stamp = fi*hop_length + sample_offset
			if p == 'eos': sample_stamp = stamp[1]
			entries.append((p, sample_stamp))
		decode_output[stamp] = entries
	phone_intervals = []
	for key in decode_output:
		for n in range(len(decode_output[key])-1):
			p = decode_output[key][n][0]
			sn = decode_output[key][n][1]
			en = decode_output[key][n+1][1]
			phone_intervals.append((sn/sampling_rate, en/sampling_rate, p))
	return phone_intervals

def to_textgrid(ofn, wav, fs, words, phone_intervals):
	"""Write to Praat textgridfile."""
	phone_intervals.sort()
	xmax = len(wav) / fs 
	header = 'File type = "ooTextFile"\nObject class = "TextGrid"\n'
	header += 'xmin = 0.0\nxmax = ' + str(xmax) + '\n'
	header += 'tiers? <exists>\nsize = 2\n'
	header += 'item []:\n'
	with open(ofn, 'w') as f:
		f.write(header)
		f.write('\n\titem [1]:\n\t\tclass = "IntervalTier"\n\t\tname = "Phone"\n\t\txmin = 0.0\n\t\txmax = ' +str(xmax)+'\n')
		f.write('\t\tintervals: size = '+str(len(phone_intervals)) + '\n')
		for n, entry in enumerate(phone_intervals):
			line = '\t\tintervals [' + str(n+1) + ']:\n'
			line += '\t\t\txmin = ' + str(entry[0]) +'\n'
			line += '\t\t\txmax = ' + str(entry[1]) + '\n'
			line += '\t\t\ttext = "' + entry[2] + '"\n'
			f.write(line)
		f.write('\n\n\titem [2]:\n\t\tclass = "IntervalTier"\n\t\tname = "Word"\n\t\txmin = 0.0\n\t\txmax = ' +str(xmax)+'\n')
		f.write('\t\tintervals: size = '+str(len(words)) + '\n')
		for n, stamp in enumerate(sorted(words)):
			line = '\t\t\tintervals [' + str(n+1) + ']:\n'
			line += '\t\t\txmin = ' + str(stamp[0]/fs) +'\n'
			line += '\t\t\txmax = ' + str(stamp[1]/fs) + '\n'
			line += '\t\t\ttext = "' + words[stamp]['word'] + '"\n'
			f.write(line)

def main(wav_file, textgrid_file, output_file, frame_shift=0.01, frame_size=0.025):
	wav, fs, words = ling111_extract_words.extract_words(wav_file, textgrid_file)
	xs = []; ys = []
	stamps = sorted(words)
	for (si, ei) in stamps:
		x = ['<s>'] + list(extract_features(wav[si:ei], fs, frame_shift, frame_size)) + ['</s>']
		y = ['sos'] + words[(si, ei)]['phones'] + ['eos']
		xs.append(x)
		ys.append(y)
	alphabet = set([p for y in ys for p in y]); alphabet -= set(['sos', 'eos'])
	md = create_model_dict(alphabet)
	# Dumb training
	sys.stderr.write('# Dumb training...\n')
	scops = dumb_train.E_step(xs, ys, md)
	dumb_train.M_step(md, scops)
	# Viterbi training
	sys.stderr.write('# Viterbi training...\n')
	sts, sops, scops, ld = viterbi_train.E_step(xs, ys, md)
	baum_welch.M_step(md, sts, sops, scops)
	# Baum-Welch training
	sys.stderr.write('# Baum-Welch training...\n')
	n_iter = 1; max_iter = 50 
	prev_ll = -np.inf; min_improvement = 1e-3
	stop = False 
	while not stop:
		sys.stderr.write('# Iteration ' + str(n_iter) + '...\n'); strt = time.time()
		sts, sops, scops, ll = baum_welch.E_step(xs, ys, md)
		sys.stderr.write('## log-likelihood = ' + str(ll) + '...\n')
		endt = time.time(); sys.stderr.write('## E-step complete in ' + str(endt-strt) + ' seconds...\n')
		mst = time.time()
		baum_welch.M_step(md, sts, sops, scops)
		met = time.time(); sys.stderr.write('## M-step complete in ' + str(met-mst) + ' seconds...\n')
		diff = ll - prev_ll
		stop = (n_iter >= max_iter) or (diff < min_improvement)
		n_iter += 1
		prev_ll = ll
	# Decoding
	phone_intervals = get_phone_intervals(xs, ys, md, stamps, int(fs*frame_shift), fs)
	to_textgrid(output_file, wav, fs, words, phone_intervals)
	sys.stderr.write('\n\n# Forced-alignment complete. Results saved as ' + output_file + '\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--wav', type=str, help='wav file')
	parser.add_argument('--textgrid', type=str, help='textgrid file')
	parser.add_argument('--frame_shift', type=float, default=0.01, help='frame shift for MFCC')
	parser.add_argument('--frame_size', type=float, default=0.025, help='frame size for MFCC')
	parser.add_argument('--output', type=str, default='./out.TextGrid', help='path to output textgrid')
	args = parser.parse_args()
	main(args.wav, args.textgrid, args.output, frame_shift=args.frame_shift, frame_size=args.frame_size)
