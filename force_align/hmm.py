"""HMM"""

# TODO:
# Load trained model 

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import numpy as np
from scipy import special

def emit_start(x, args):
	if type(x) == str:
		if x == args['sos']: return 0
	return -np.inf

def emit_final(x, args):
	if type(x) == str:
		if x == args['eos']: return 0
	return -np.inf

def gmm_component_logpdfs(x, args):
	"""Compute logpdf(x) for each GMM component."""
	diagonal = args.get('diagonal', False)
	M, D = args['means'].shape
	log_probs = np.zeros(M)
	if diagonal:
		vars_ = np.array([np.diag(args['covs'][m]) for m in range(M)])
		diff = x[None, :] - args['means']
		log_gauss = -0.5 * ( np.sum(np.log(2*np.pi*vars_), axis=1) + np.sum((diff*diff)/vars_, axis=1))
		log_probs = np.log(args['weights']) + log_gauss
	else:
		for m in range(M):
			diff = x - args['means'][m]
			L = np.linalg.cholesky(args['covs'][m]) # cholesky decomposition
			y = np.linalg.solve(L, diff) # solve L y = diff
			quad = np.dot(y, y)
			log_det = 2.0 * np.sum(np.log(np.diag(L)))
			log_gauss = -0.5 * (D * np.log(2*np.pi) + log_det + quad)
			log_probs[m] = np.log(args['weights'][m]) + log_gauss
	return log_probs

def emit_GMM(x, args):
	if type(x) != np.ndarray: return -np.inf
	log_probs = gmm_component_logpdfs(x, args)
	return special.logsumexp(log_probs)

class NonEmittingState():

	def __init__(self):
		self.type = 'NonEmittingState'

	def __str__(self):
		return 'Non-emitting state'


class EmittingState:

	def __init__(self, emission_func, emission_config):
		self.type = 'EmittingState'
		self.emission = emission_func
		if self.emission.__name__ == 'emit_GMM':
			n_components = emission_config['n_components']
			output_dim = emission_config['output_dim']
			self.emission_args = {}
			self.emission_args['diagonal'] = emission_config.get('diagonal', False)
			self.emission_args['n_components'] = n_components
			self.emission_args['weights'] = np.ones(n_components) / n_components
			self.emission_args['means'] = np.random.random((n_components, output_dim))
			self.emission_args['covs'] = np.tile(np.diag([1.]*output_dim).reshape(1, output_dim, output_dim), (n_components, 1, 1))
		elif self.emission.__name__ == 'emit_start':
			sos = emission_config['sos'] 
			self.emission_args = {'sos':sos}
		elif self.emission.__name__ == 'emit_final':
			eos = emission_config['eos'] 
			self.emission_args = {'eos':eos}

	def __str__(self):
		out = self.type
		out += '\n Emission function: ' + self.emission.__name__
		for arg in sorted(self.emission_args):
			out += '\n\n  ' + arg + ': ' + str(self.emission_args[arg])
		out += '\n'
		return out

	def logpdf(self, x):
		return self.emission(x, self.emission_args)

class Model:

	def __init__(self, name, n_emitting_states, model_type, emission_func, emission_config):
		self.name = name
		self.type = model_type
		self.states = [EmittingState(emission_func, emission_config) for n in range(n_emitting_states)]
		if model_type != 'singleton': self.states = [NonEmittingState()] + self.states + [NonEmittingState()]
		self.trans = np.zeros((len(self.states), len(self.states)))
		if model_type!= 'singleton':
			self.trans[0, 1] = 1.0
			if model_type == 'ergodic':
				self.trans[1:-1, 1:] = np.random.random((n_emitting_states, n_emitting_states+1))
			elif model_type in ['linear', 'bakis', 'left-to-right']:
				self.trans[1:-1, 1:-1] += np.diag([1]*n_emitting_states)
				self.trans[1:-1, 1:] += np.diag([1]*n_emitting_states, k=1)[:-1, :]
				if model_type in ['bakis', 'left-to-right'] and n_emitting_states >= 2:
					self.trans[1:-1, 1:] += np.diag([1]*(n_emitting_states-1), k=2)[:-1, :]
				if model_type == 'left-to-right':
					for j in range(2, n_emitting_states):
						self.trans[1:-1, 1:] += np.diag([1]*(n_emitting_states-j), k=j+1)[:-1, :] 
			self.trans[1:-1, 1:] /= self.trans[1:-1, 1:].sum(axis=1).reshape(-1, 1)
		self.allowed_transitions = self.trans > 0

	def __str__(self):
		out = '# Model ' + self.name + '\n'
		out += '## Type: ' + self.type 
		out += '\n## States:\n'
		for i, s in enumerate(self.states): out += '\n### ' + str(i) + '\n' + str(s)
		out += '\n\n## Transitional probabilities:\n'
		out += str(self.trans)
		out += '\n\n\n'
		return out

	def update_a(self, bigram_counts):
		"""Update transition probabilities.

		Args:
		- bigram_counts: a dictionary of state transition counts
		-- dict[ (model1, state1), (model2, state2) ] = count
		-- model1 and/or model2 may not equal the current model
		"""
		if self.type != 'singleton':
			eps = 1e-6
			new_trans = np.zeros_like(self.trans) + eps
			for (m1,s1), (m2,s2) in bigram_counts:
				if m1 == self.name and m2 == self.name:
					new_trans[s1, s2] = bigram_counts[((m1,s1), (m2,s2))]
			new_trans *= self.allowed_transitions
			for i in range(len(self.trans)):
				row_sum = new_trans[i].sum()
				if row_sum > 0: new_trans[i] /= row_sum
				else: new_trans[i] = self.trans[i] # do not update if no new data
			self.trans = new_trans
		
	def update_b(self, state_obs_pairs, update_func, update_func_kwargs):
		if self.type != 'singleton':
			for i, s in enumerate(self.states[1:-1]):
				data = state_obs_pairs.get((self.name, i+1))
				if not data is None: update_func(s, data, update_func_kwargs) 


class Chain:

	def __init__(self, models):
		self.states = []
		self.index = [] # (model name, state index) to map each chain state to model and state
		for m in models:
			self.states += m.states
			for j in range(len(m.states)):
				self.index.append((m.name, j))
		n_states = len(self.states)
		self.trans = np.zeros((n_states, n_states))
		i = 0
		for j, m in enumerate(models):
			n = len(m.states)
			self.trans[i:i+n, i:i+n] = m.trans
			if j < len(models)-1: self.trans[i+n-1, i+n] = 1.0
			i += n

				
def test():
	gmm_config = {'output_dim':2, 'n_components':3}
	seq_len = 6
	#m = Model('bakis', 4, 'bakis', emit_GMM, gmm_config)
	#m = Model('bakis', 3, 'linear', emit_GMM, gmm_config)
	#m = Model('l2r', 3, 'left-to-right', emit_GMM, gmm_config)
	m = Model('erg', 1, 'ergodic', emit_GMM, gmm_config)
	sos = Model('sos', 1, 'singleton', emit_start, {'sos':'<s>'})
	eos = Model('eos', 1, 'singleton', emit_final, {'eos':'</s>'})
	mc = Chain([sos, m, eos])
	for s in mc.states: print(s)

if __name__ == '__main__':
	test()
