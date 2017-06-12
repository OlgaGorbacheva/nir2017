import numpy as np

class InterException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def sin_func(a=1, b=0, c=1, d=0):
	if a == 0 or c == 0:
		raise InterException('Not sin function at all, check args')
	def _sin(x):
		return (c * np.sin(a * x + b) + d)
	return _sin

class SampleGenerator:
	def __init__(self, x_left_bound, x_right_bound, params_bounds):
		self.x_left_bound = x_left_bound
		self.x_right_bound = x_right_bound
		self.params_bounds = params_bounds

	def __call__(self, sample_size=100, set_size=10000, t_size=0.1, s_type='simple_rnn'):
		x = np.linspace(self.x_left_bound, self.x_right_bound, sample_size)
		X_set, Y_set = [], []
		for _ in range(set_size):
			params = [np.random.uniform(p_bound[0], p_bound[1]) for p_bound in self.params_bounds]
			fnc = sin_func(*params)
			X_set.append(np.array(params))
			Y_set.append(np.array(fnc(x)))
		ntest = int(round(set_size * (1 - t_size)))
		if s_type == 'simple_rnn':
			new_X_set = []
			for x in X_set: 
				a = []
				for _ in range(sample_size):
					a.append(x)
				new_X_set.append(np.array(a))
			X_set = new_X_set
		return np.array(X_set[:ntest]), np.array(Y_set[:ntest]), np.array(X_set[ntest:]), np.array(Y_set[ntest:])

