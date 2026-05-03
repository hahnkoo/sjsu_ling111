"""Viterbi training"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import sys, time
import numpy as np
import hmm, trellis, baum_welch 

def E_step_example(output_sequence, model_name_sequence, md, state_transitions, state_output_pairs, state_component_output_pairs, verbose=False):
	mc = hmm.Chain([md[m] for m in model_name_sequence])
	if type(output_sequence) != list: output_sequence = list(output_sequence)
	if verbose: sys.stderr.write('# Initializing trellis... '); strt = time.time()
	tr = trellis.Trellis(mc, output_sequence)
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Adding columns... '); strt = time.time()
	tr.build_viterbi()
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Decoding... '); strt = time.time()
	best_path = tr.decode()
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing xi, gamma_t, and lambda_t... '); strt = time.time()
	xi = {}; gamma_t = {}; ssl = []
	for t, path in enumerate(best_path):
		gamma_t[t] = {path.states[-1]:1.0}
		ssl += path.states
	for i in range(len(ssl)-1):
		if not ssl[i] in xi: xi[ssl[i]] = {}
		if not ssl[i+1] in xi[ssl[i]]: xi[ssl[i]][ssl[i+1]] = 0
		xi[ssl[i]][ssl[i+1]] += 1
	lambda_t = baum_welch.compute_lambda_t(tr, gamma_t)
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	for i in xi:
		mi = (mc.index[i][0], mc.index[i][1])
		for j in xi[i]:
			mj = (mc.index[j][0], mc.index[j][1])
			if not (mi, mj) in state_transitions: state_transitions[(mi, mj)] = 0.0
			state_transitions[(mi, mj)] += xi[i][j]
	for t in gamma_t:
		for j in gamma_t[t]:
			mj = (mc.index[j][0], mc.index[j][1])
			if not mj in state_output_pairs: state_output_pairs[mj] = []
			state_output_pairs[mj].append([output_sequence[t], gamma_t[t][j]])
	for t in lambda_t:
		for j in lambda_t[t]:
			key = (mc.index[j][0], mc.index[j][1])
			if not key in state_component_output_pairs: state_component_output_pairs[key] = {}
			for m in range(len(lambda_t[t][j])):
				if not m in state_component_output_pairs[key]: state_component_output_pairs[key][m] = []
				state_component_output_pairs[key][m].append([output_sequence[t], lambda_t[t][j][m]])
	return best_path[-1].log_delta # return log-delta of the best path

def E_step(xs, ys, md):
	ld = 0.0 # log delta
	sts = {}; sops = {}; scops = {}
	for n in range(len(xs)):
		ldn = E_step_example(xs[n], ys[n], md, state_transitions=sts, state_output_pairs=sops, state_component_output_pairs=scops)
		ld += ldn
	ld /= len(xs)
	return sts, sops, scops, ld


def test():
	np.set_printoptions(precision=2)
	D = 39 
	n_components = 4
	#gmm_config = {'output_dim':D, 'n_components':n_components}
	gmm_config = {'output_dim':D, 'n_components':n_components, 'diagonal':True}
	#x_min_length = 30; x_max_length = 31; y_min_length = 5; y_max_length = 8
	x_min_length = 300; x_max_length = 301; y_min_length = 50; y_max_length = 51
	#x_min_length = 3000; x_max_length = 3001; y_min_length = 300; y_max_length = 301 
	n_examples = 1 
	md = {'eos':hmm.Model('eos', 1, 'singleton', hmm.emit_final, {'eos':'</s>'}), 'sos':hmm.Model('sos', 1, 'singleton', hmm.emit_start, {'sos':'<s>'})}
	#alphabet = ['a', 'b', 'c']
	alphabet = list('abcdefghijklmnopqrstuvwxyz')
	for alph in alphabet:
		md[alph] = hmm.Model(alph, 3, 'left-to-right', hmm.emit_GMM, gmm_config)
	xs = [['<s>'] + list(np.random.random((np.random.randint(x_min_length, x_max_length), D))) + ['</s>'] for _ in range(n_examples)]
	ys = []
	for x in xs:
		y_len = np.random.randint(y_min_length, y_max_length)
		y = ['sos'] + list(np.random.choice(alphabet, y_len).astype(str)) + ['eos']
		ys.append(y)

	n_iter = 1; max_iter = 3
	prev_ll = -np.inf; min_improvement = 1e-2
	stop = False
	while not stop:
		sys.stderr.write('# Iteration ' + str(n_iter) + '...\n')
		est = time.time()
		sts, sops, scops, ll = E_step(xs, ys, md)
		eet = time.time()
		sys.stderr.write('## log-likelihood = ' + str(ll) + '...\n')
		sys.stderr.write('## E-step complete in ' + str(eet-est) + ' seconds...\n')
		mst = time.time()
		baum_welch.M_step(md, sts, sops, scops)
		met = time.time()
		sys.stderr.write('## M-step complete in ' + str(met-mst) + ' seconds...\n')
		diff = ll - prev_ll
		stop = (n_iter >= max_iter) or (diff < min_improvement) 
		n_iter += 1
		prev_ll = ll

if __name__ == '__main__':
	test()
