"""Baum-Welch Algorithm"""

__author__ = 'Hahn Koo (hahn.koo@sjsu.edu)'

import sys, time
import numpy as np
from scipy import special
import hmm, trellis

def compute_xi(tr):
	"""Compute xi(i, j) = sum_t P(q_t=i, q_{t+1}=j|o)."""
	xi = {}
	log_p_obs = tr.columns[0][0][0].log_beta
	for t in range(1, len(tr.columns)):
		for q in tr.columns[t]:
			for path in tr.columns[t][q]:
				i = path.source
				if not i in xi: xi[i] = {}
				log_p_trans_obs = tr.src_log_alpha[t][i]
				log_p_trans_obs += np.log(tr.model.trans[i][path.states[0]]) 
				log_p_trans_obs += path.logprob
				log_p_trans_obs += tr.log_emit[t][q]
				log_p_trans_obs += path.log_beta
				p_trans_obs = np.exp(log_p_trans_obs - log_p_obs)
				for j in path.states:
					if not j in xi[i]: xi[i][j] = 0.0 
					xi[i][j] += p_trans_obs 
	return xi 

def compute_gamma_t(tr):
	"""Compute gamma_t(j) = P(q_t=j|o)"""
	gamma_t = {}
	log_p_obs = tr.columns[0][0][0].log_beta
	for t in range(len(tr.columns)):
		gamma_t[t] = {}
		for j in tr.columns[t]:
			gamma_t[t][j] = 0.0
			for path in tr.columns[t][j]:
				log_gamma = path.log_alpha + path.log_beta - log_p_obs
				gamma_t[t][j] += np.exp(log_gamma) 
	return gamma_t

def compute_lambda_t(tr, gamma_t):
	"""Compute mixture posterior L_t(j, m) = P(q_t=j, c_j=m|o)

	where 
	- j is an emitting state with GMM for emission pdf
	- c_j=m means GMM component m was chosen to emit output from state j

	L_t(j, m) is gamma_t(j) weighted by the likelihood per component m normalized by the sum of likelihoods over all components (i.e. likelihood per the whole GMM), i.e.
 
	L_t(j, m) = gamma_t(j) * c_j(m) * N(o_t | mean_m, var_m) / pdf(o_t | GMM)
	"""
	lambda_t = {}
	for t in gamma_t:
		for j in gamma_t[t]:
			if gamma_t[t][j] == 0: continue # skip if state posterior is zero
			if tr.model.states[j].emission.__name__ == 'emit_GMM':
				if not t in lambda_t: lambda_t[t] = {}
				n_components = tr.model.states[j].emission_args['n_components']

				log_gamma = np.log(gamma_t[t][j])
				log_comp = np.zeros(n_components) # log component likelihoods
				for m in range(n_components):
					log_comp[m] = tr.log_gmm_emit[t][j, m]
				log_state = tr.log_emit[t][j]
				log_lambda = log_gamma + log_comp - log_state
				lambda_t[t][j] = np.exp(log_lambda)
	return lambda_t

def E_step_example(output_sequence, model_name_sequence, md, state_transitions, state_output_pairs, state_component_output_pairs, verbose=False):
	mc = hmm.Chain([md[m] for m in model_name_sequence])
	if type(output_sequence) != list: output_sequence = list(output_sequence)
	if verbose: sys.stderr.write('# Initializing trellis... '); strt = time.time()
	tr = trellis.Trellis(mc, output_sequence)
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Adding columns... '); strt = time.time()
	tr.build_generic()
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing alphas... '); strt = time.time()
	tr.forward()
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing betas... '); strt = time.time()
	tr.backward()
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing xis... '); strt = time.time()
	xi = compute_xi(tr)
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing gammas... '); strt = time.time()
	gamma_t = compute_gamma_t(tr)
	if verbose: endt = time.time(); sys.stderr.write('complete in ' + str(endt-strt) + ' seconds.\n')
	if verbose: sys.stderr.write('# Computing lambdas... '); strt = time.time()
	lambda_t = compute_lambda_t(tr, gamma_t)
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
	return tr.columns[0][0][0].log_beta # return log likelihood of the example

def E_step(xs, ys, md):
	ll = 0.0 # log likelihood
	sts = {}; sops = {}; scops = {}
	for n in range(len(xs)):
		lln = E_step_example(xs[n], ys[n], md, state_transitions=sts, state_output_pairs=sops, state_component_output_pairs=scops)
		ll += lln
	ll /= len(xs)
	return sts, sops, scops, ll 


def update_GMM(state, component_output_pairs, kwargs):
	eps_weight = kwargs.get('eps_weight', 1e-6) 
	eps_var = kwargs.get('eps_var', 1e-4)
	weights = np.zeros_like(state.emission_args['weights'])
	means = state.emission_args['means'].copy()
	covs = state.emission_args['covs'].copy()
	for m in range(len(weights)):
		data = component_output_pairs.get(m)
		# if no data or sum prob < eps, set weight to eps and skip updating
		m_total = 0.0
		if not data is None: m_total = sum(p for o, p in data)
		if m_total <= eps_weight:
			weights[m] = eps_weight
			continue
		# else, update
		weights[m] = m_total
		mean = np.sum([p*o for o, p in data], axis=0)
		means[m] = mean / m_total
		cov = np.zeros_like(covs[m])		
		for o, p in data:
			diff = (o - means[m]).reshape(-1, 1)
			cov += p * (diff @ diff.T)
		cov /= m_total
		cov += np.eye(cov.shape[0]) * eps_var # add eps to diagonal to prevent underflow
		covs[m] = cov
	weights /= weights.sum()
	state.emission_args['weights'] = weights
	state.emission_args['means'] = means
	state.emission_args['covs'] = covs

def M_step(md, state_transitions, state_output_pairs, state_component_output_pairs):
	for m in md:
		md[m].update_a(state_transitions)
		md[m].update_b(state_component_output_pairs, update_GMM, {})
	

def test():
	np.set_printoptions(precision=2)
	D = 39 
	n_components = 4
	#gmm_config = {'output_dim':D, 'n_components':n_components}
	gmm_config = {'output_dim':D, 'n_components':n_components, 'diagonal':True}
	x_min_length = 30; x_max_length = 100; y_min_length = 2; y_max_length = 15
	#x_min_length = 300; x_max_length = 301; y_min_length = 50; y_max_length = 51
	#x_min_length = 3000; x_max_length = 3001; y_min_length = 300; y_max_length = 301 
	n_examples = 50 
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

#	print('# Before training:')
#	for m in md: print(md[m])

	n_iter = 1; max_iter = 10
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
		M_step(md, sts, sops, scops)
		met = time.time()
		sys.stderr.write('## M-step complete in ' + str(met-mst) + ' seconds...\n')
		diff = ll - prev_ll
		stop = (n_iter >= max_iter) or (diff < min_improvement) 
		n_iter += 1
		prev_ll = ll

#	print('# After training:')
#	for m in md: print(md[m])

if __name__ == '__main__':
	test()
