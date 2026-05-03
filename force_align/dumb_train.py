"""Dumb training: Split output sequence into equally long intervals and train"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import sys, time
import numpy as np
import hmm, baum_welch 

def E_step_example(output_sequence, model_name_sequence, md, state_component_output_pairs):
	"""Distribute outputs equally across emitting states and then randomly across GMM components."""
	mc = hmm.Chain([md[m] for m in model_name_sequence])
	es = []
	for i, s in enumerate(mc.states):
		if s.type == 'EmittingState':
			if s.emission.__name__ == 'emit_GMM':
				es.append(i)
	n_outputs = len(output_sequence[1:-1])
	n_states = len(es)
	lcm = np.lcm(n_outputs, n_states)
	output_indices = np.repeat(np.arange(1, n_outputs+1), int(lcm/n_outputs))
	state_indices = np.repeat(np.arange(n_states), int(lcm/n_states))
	for i in range(lcm):
		obs = output_sequence[output_indices[i]]
		j = es[state_indices[i]]
		key = (mc.index[j][0], mc.index[j][1])
		if not key in state_component_output_pairs: state_component_output_pairs[key] = {}
		nc = mc.states[j].emission_args['n_components']
		for c in range(nc):
			m = np.random.choice(np.arange(nc))
			if not m in state_component_output_pairs[key]: state_component_output_pairs[key][m] = []
			state_component_output_pairs[key][m].append([obs, 1.0])

def E_step(xs, ys, md):
	scops = {}
	for n in range(len(xs)):
		E_step_example(xs[n], ys[n], md, scops)
	return scops 

def M_step(md, state_component_output_pairs):
	for m in md:
		md[m].update_b(state_component_output_pairs, baum_welch.update_GMM, {})

def test():
	np.set_printoptions(precision=2)
	D = 39 
	n_components = 4
	#gmm_config = {'output_dim':D, 'n_components':n_components}
	gmm_config = {'output_dim':D, 'n_components':n_components, 'diagonal':True}
	x_min_length = 30; x_max_length = 50; y_min_length = 5; y_max_length = 8
	#x_min_length = 300; x_max_length = 501; y_min_length = 50; y_max_length = 80
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
	scops = E_step(xs, ys, md)
	M_step(md, scops)

if __name__ == '__main__':
	test()
