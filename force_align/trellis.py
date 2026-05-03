"""Trellis library"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import numpy as np
from scipy import special
import hmm

def paths_to_emitting(state_index, model):
	"""List all paths to emitting states from the state."""
	paths = []
	logprobs = []

	def dfs(i, cur_path, logprob):
		cur_path = cur_path + [i]
		if model.states[i].type == 'EmittingState':
			paths.append(cur_path)
			logprobs.append(logprob)
			return
		for j in np.where(model.trans[i] > 0)[0]:
			dfs(int(j), cur_path, logprob + np.log(model.trans[i, j]))

	dfs(state_index, [], 0.0)

	return paths, logprobs


class Path:

	def __init__(self, states, logprob, source):
		self.states = states # states constituting the path
		self.logprob = logprob # log probability of the path itself
		self.source = source # state that the path is expanding

	def __str__(self):
		out = ''
		for key, val in vars(self).items():
			out += key + ':' + str(val) + ', '
		out = out.strip(', ')
		return out

class Trellis:

	def __init__(self, model, obs):
		self.model = model
		self.obs = obs
		self.columns = [ {0:[Path([0], 0.0, None)]} ] # a column = {state:[Paths to state]}; the first state and obs are assumed to be a special start state (sos) and a special start output <s>
		self.precompute_log_emissions()
		self.precompute_paths()

	def __str__(self):
		out = ''
		for t in range(len(self.columns)):
			out += '\n\n# Column ' + str(t)
			for q in self.columns[t]:
				out += '\n## State: ' + str(q)
				for path in self.columns[t][q]:
					out += '\n' + path.__str__()
		return out

	def precompute_log_emissions(self):
		"""Compute log emission probabilities and save for later."""
		self.log_emit = []
		self.log_gmm_emit = []
		for obs_t in self.obs:
			row = {}
			row_comp = {}
			for j, state in enumerate(self.model.states):
				if state.type == 'EmittingState':
					if state.emission.__name__ == 'emit_GMM' and type(obs_t) == np.ndarray:
						lpdf = hmm.gmm_component_logpdfs(obs_t, state.emission_args)
						for m in range(len(lpdf)): row_comp[(j, m)] = lpdf[m]
						row[j] = special.logsumexp(lpdf)
					else:
						row[j] = state.logpdf(obs_t)
			self.log_emit.append(row)
			self.log_gmm_emit.append(row_comp)

	def precompute_paths(self):
		"""Compute paths to emitting states and save for later."""
		self.path_cache = {}
		for i in range(len(self.model.states)):
			self.path_cache[i] = paths_to_emitting(i, self.model)

	def add_column(self):
		"""Add a column to the trellis."""
		t = len(self.columns)
		col = {}
		for i in self.columns[-1]:
			for j in np.where(self.model.trans[i] > 0)[0]:
				paths_j, logprobs_j = self.path_cache[int(j)]
				for k in range(len(paths_j)):
					q = paths_j[k][-1]
					if self.log_emit[t][q] > -np.inf:
						if not q in col: col[q] = []
						col[q].append( Path(paths_j[k], logprobs_j[k], i) )
		self.columns.append(col)

	def viterbi_column(self, t):
		"""Revise entries in self.columns[t] for the Viterbi algorithm:
		For each emission state,
		(1) identify the best path and remove the rest;
		(2) add log-delta and crumb information to the best path.
		"""
		for q in self.columns[t]:
			if t == 0:
				self.columns[t][q][0].log_delta = 0
				self.columns[t][q][0].crumb = None
			else:
				log_delta = -np.inf
				crumb = None
				best_path = None
				for path in self.columns[t][q]:
					prev = path.source
					cand_log_delta = self.columns[t-1][prev][0].log_delta + path.logprob
					if cand_log_delta > log_delta:
						log_delta = cand_log_delta
						crumb = prev
						best_path = path
				log_delta += self.log_emit[t][q]
				best_path.log_delta = log_delta
				best_path.crumb = crumb
				self.columns[t][q] = [best_path]

	def build_generic(self):
		stop = (len(self.columns) == len(self.obs)) or (self.columns[-1] == {})
		while not stop:
			self.add_column()
			stop = (len(self.columns) == len(self.obs)) or (self.columns[-1] == {})

	def build_viterbi(self):
		self.viterbi_column(0)
		stop = (len(self.columns) == len(self.obs)) or (self.columns[-1] == {})
		while not stop:
			t = len(self.columns)
			self.add_column()
			self.viterbi_column(t)
			stop = (len(self.columns) == len(self.obs)) or (self.columns[-1] == {})

	def decode(self):
		seq = []
		for t in range(len(self.columns)-1, -1, -1):
			if t == len(self.columns)-1:
				best = None; max_delta = -np.inf
				for s in self.columns[t]:
					if self.columns[t][s][0].log_delta > max_delta:
						best = s
						max_delta = self.columns[t][s][0].log_delta
				if best is None: break
				else:
					seq.append((self.columns[t][best][0], self.columns[t][best][0].crumb))
			else:
				best = seq[-1][1]
				seq.append((self.columns[t][best][0], self.columns[t][best][0].crumb))
		seq.reverse()
		out = [s[0] for s in seq]
		return out
				

	def forward(self):
		self.columns[0][0][0].log_alpha = 0.0
		self.src_log_alpha = {}
		for t in range(1, len(self.columns)):
			self.src_log_alpha[t] = {}
			for i in self.columns[t-1]:
				self.src_log_alpha[t][i] = special.logsumexp([path.log_alpha for path in self.columns[t-1][i]])
			for j in self.columns[t]:
				for path in self.columns[t][j]:
					i = path.source
					path.log_alpha = self.src_log_alpha[t][i] 
					path.log_alpha += np.log(self.model.trans[i][path.states[0]]) + path.logprob
					path.log_alpha += self.log_emit[t][j]

	def backward(self):
		for j in self.columns[-1]:
			for path in self.columns[-1][j]:
				path.log_beta = 0.0
		for t in range(len(self.columns)-2, -1, -1):
			sources = []
			vals = []
			for j in self.columns[t+1]:
				for path in self.columns[t+1][j]:
					i = path.source
					log_bp = path.log_beta 
					log_bp += self.log_emit[t+1][j]
					log_bp += path.logprob + np.log(self.model.trans[i][path.states[0]])
					sources.append(i)
					vals.append(log_bp)
			sources = np.asarray(sources, dtype=int)
			vals = np.asarray(vals)
			beta_by_source = np.full(len(self.model.states), -np.inf)
			np.logaddexp.at(beta_by_source, sources, vals)
			for i in self.columns[t]:
				beta_i = beta_by_source[i]
				for path in self.columns[t][i]:
					path.log_beta = beta_i
 
def test():
	np.set_printoptions(precision=2)
	D = 5
	n_components = 2
	gmm_config = {'output_dim':D, 'n_components':n_components}
	min_length = 5; max_length = 10
	n_examples = 20
	md = {'eos':hmm.Model('eos', 1, 'singleton', hmm.emit_final, {'eos':'</s>'}), 'sos':hmm.Model('sos', 1, 'singleton', hmm.emit_start, {'sos':'<s>'})}
	alphabet = ['a', 'b', 'c']
	#alphabet = list('abcdefghijklmnopqrstuvwxyz')
	for alph in alphabet:
		md[alph] = hmm.Model(alph, 3, 'left-to-right', hmm.emit_GMM, gmm_config)
	xs = [['<s>'] + list(np.random.random((np.random.randint(min_length, max_length), D))) + ['</s>'] for _ in range(n_examples)]
	ys = []
	for x in xs:
		y_len = np.random.randint(1, len(x)-2)
		y = ['sos'] + list(np.random.choice(alphabet, y_len).astype(str)) + ['eos']
		ys.append(y)

	mc = hmm.Chain([md[m] for m in ys[0]])
	tr = Trellis(mc, xs[0])
	#tr.build_generic()
	#tr.forward()
	#tr.backward()
	tr.build_viterbi()
	print(tr)
	best_path = tr.decode()
	print('\nbest path:')
	for p in best_path: print(p)

if __name__ == '__main__':
	test()
