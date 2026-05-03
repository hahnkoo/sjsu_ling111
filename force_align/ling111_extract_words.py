"""Extract word intervals and their corresponding phonetic transcriptions -- A script for LING 111 Final Project Annotation #2
"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import argparse, re
import numpy as np
import librosa

def list_intervals_in_tier(textgrid, tier):
	"""List all intervals in the textgrid tier."""
	out = []
	with open(textgrid) as f:
		in_tier = False 
		entry = None 
		for line in f:
			line = line.strip()
			if re.match(r'name = ', line):
				tier_name = line.split('=')[-1].strip()
				if tier_name[1:-1].lower() == tier.lower(): in_tier = True
				else: in_tier = False
			elif re.match(r'item', line): in_tier = False
			if in_tier:
				if re.match(r'intervals \[\d+\]:', line):
					if entry != [] and not entry is None: out.append(entry)
					entry = []
				elif not entry is None:
					entry.append(line)
		if entry != []: out.append(entry)
	return out

def clean_interval_entries(entries):
	"""Clean interval entries: e.g. digit string -> float, strip white-space and " from label."""
	out = []
	for entry in entries:
		st, et, label = entry[:3]
		st = float(st.strip().split('=')[-1].strip())
		et = float(et.strip().split('=')[-1].strip())
		label = label.strip().split('=')[-1].strip()[1:-1] # label originally surrounded by "
		if label == '': label = '<NO_LABEL>'
		out.append([st, et, label])
	return out

def intervals_by_sample(sampling_rate, tier_entries):
	"""For each sample, label which entry it belongs to in terms of interval entry index in the tier."""
	out = []
	for i, entry in enumerate(tier_entries):
		st, et, label = entry
		n_st = int(st * sampling_rate)
		n_et = int(et * sampling_rate)
		sample_label = [i] * (n_et - n_st)
		out += sample_label
	return np.array(out)

def classify(small_intervals, big_intervals):
	"""For each interval in small_intervals, decide which interval in big_intervals it overlaps with the most and vice versa."""
	s2b = {}; b2s = {}
	for n in range(small_intervals[-1]+1):
		si = np.where(small_intervals == n)[0]
		sis, sie = si[0], si[-1]
		bi = big_intervals[sis:sie+1]
		bic = np.bincount(bi).argmax()
		s2b[n] = bic
		if not bic in b2s: b2s[bic] = []
		b2s[bic].append(n)
	return s2b, b2s 

def extract_words(wav, textgrid):
	"""Extract words and the corresponding phonetic transcriptions."""
	intervals = {}
	x, fs = librosa.load(wav, sr=None)
	word_tier_entries = clean_interval_entries(list_intervals_in_tier(textgrid, 'word'))
	wis = intervals_by_sample(fs, word_tier_entries)
	phone_tier_entries = clean_interval_entries(list_intervals_in_tier(textgrid, 'phone'))
	pis = intervals_by_sample(fs, phone_tier_entries)
	p2w, w2p = classify(pis, wis)
	for w in w2p:
		st = word_tier_entries[w][0]
		n_st = int(st * fs)
		et = word_tier_entries[w][1]
		n_et = int(et * fs)
		word = word_tier_entries[w][-1]
		p_trans = []
		for p in w2p[w]: p_trans += phone_tier_entries[p][-1].strip().split() 
		intervals[(n_st, n_et)] = {'word':word, 'phones':p_trans}
	return x, fs, intervals

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--wav', type=str)
	parser.add_argument('--textgrid', type=str)
	args = parser.parse_args()
	x, fs, intervals = extract_words(args.wav, args.textgrid)
	print(intervals)
