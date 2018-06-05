# This class defines the API to add Ops to train a model. 
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import state_ops
from tensorflow.python.training import optimizer
import tensorflow as tf

class CustomOptimizer(optimizer.Optimizer):
	"""Implementation of CustomOptimizer.
	"""
	def __init__(self, learning_rate=0.001, use_locking=False, name="CustomOptimizer"):
		super(CustomOptimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._beta1 = 0.9
		self._beta2 = self._beta3 = 0.999 
		self._rmsprop_beta = 0.9
		self._epsilon = 1e-8
		self._code = [1,1]
		# Tensor versions of the constructor arguments, created in _prepare().
		self._lr_t = None
		self._beta1_t = None
		self._beta2_t = None
		self._beta3_t = None
		self._rmsprop_beta_t = None
		self._epsilon_t = None

	def _prepare(self):
		self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate")
		self._beta1_t = ops.convert_to_tensor(self._beta1, name="beta1_t")
		self._beta2_t = ops.convert_to_tensor(self._beta2, name="beta2_t")
		self._beta3_t = ops.convert_to_tensor(self._beta3, name="beta3_t")
		self._rmsprop_beta_t = ops.convert_to_tensor(self._rmsprop_beta, name="rmsprop_beta_t")
		self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="rmsprop_beta_t")

	def _create_slots(self, var_list):
		# default var_list as all trainable variables
		# Create slots for the first and second moments.
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
			self._zeros_slot(v, "v", self._name)
			self._zeros_slot(v, "r", self._name)
			self._zeros_slot(v, "rmsprop_v", self._name)

	def unary(self, var, key):
		unary_opts = {
			1:lambda x: x,
			2:lambda x: -x,
			3:lambda x: tf.exp(x),
			4:lambda x: tf.log(tf.abs(var)),
			5:lambda x: tf.sqrt(tf.abs(var)),
			6:lambda x: tf.clip(x, -1e-5, 1e-5),
			7:lambda x: tf.clip(x, -1e-4, 1e-4),
			8:lambda x: tf.clip(x, -1e-3, 1e-3),
			9:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[1.0, 9.0]]), 1)[0][0], 0), 0, x),
			10:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[3.0, 7.0]]), 1)[0][0], 0), 0, x),
			11:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[5.0, 5.0]]), 1)[0][0], 0), 0, x),
			12:lambda x: tf.sign(x)
		}
		return unary_opts[key](var)

	def _apply_dense(self, grad, var):
		operand_names = ['g', 'g2', 'g3', 'm', 'v', 'r', 'sign_g', 'sign_m', '1', '2', 'epsilon', '10-4w', '10-3w', '10-2w', '10-1w', 'Adam', 'RMSprop']
		operands = {oprd:None for oprd in operand_names}
		# Prepare constants
		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
		beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
		beta3_t = math_ops.cast(self._beta3_t, var.dtype.base_dtype)
		rmsprop_beta_t = math_ops.cast(self._rmsprop_beta_t, var.dtype.base_dtype)
		epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
		eps = 1e-7 #cap for moving average
		# Prepare primitives
		grad2 = grad * grad
		grad3 = grad2 * grad
		m = self.get_slot(var, "m")
		v = self.get_slot(var, "v")
		r = self.get_slot(var, "r")
		rmsprop_v = self.get_slot(var, "rmsprop_v")
		# Compute 
		#m_t = m.assign(tf.maximum(beta_t * m + eps, tf.abs(grad)))
		m_t = state_ops.assign(m, (m * beta1_t) + grad * (1 - beta1_t))
		v_t = state_ops.assign(v, (v * beta2_t) + grad2 * (1 - beta2_t))
		r_t = state_ops.assign(r, (r * beta3_t) + grad3 * (1 - beta3_t))
		rmsprop_v_t = state_ops.assign(rmsprop_v, (rmsprop_v * rmsprop_beta_t) + grad2 * (1 - rmsprop_beta_t))
		rmsprop_v_t_sqrt = math_ops.sqrt(rmsprop_v_t + epsilon_t)
		# assumption: adam and rmsprop output without mutiplying learning rate
		rmsprop =  grad / rmsprop_v_t_sqrt
		m_t_hat = m_t / (1 - beta1_t)
		v_t_hat = v_t / (1 - beta2_t)
		adam =  m_t_hat / (math_ops.sqrt(v_t_hat) + epsilon_t)
		operands = {
			1: grad,
			2: grad2,
			3: grad3,
			4: m_t,
			5: v_t,
			6: r_t,
			7: tf.sign(grad),
			8: tf.sign(grad),
			9: 1,
			10: 2,
			11: epsilon_t,
			12: 1e-4*var,
			13: 1e-3*var,
			14: 1e-2*var,
			15: 1e-1*var,
			16: adam,
			17: rmsprop
		}
		operand1 = operands[self._code[0]]
		opt1 = self.unary(operand1, self._code[1])
		result = opt1

		var_update = state_ops.assign_sub(var, lr_t * result, use_locking=self._use_locking)
		#var_update = state_ops.assign_sub(var, operands[17], use_locking=self._use_locking)

		#var_update = state_ops.assign_sub(var, lr_t*grad*tf.exp(tf.sign(grad)*tf.sign(m_t))) #Update 'ref' by subtracting 'value
		#Create an op that groups multiple operations.
		#When this op finishes, all ops in input have finished
		return control_flow_ops.group(*[var_update, m_t])

	def _apply_sparse(self, grad, var):
		raise NotImplementedError("Sparse gradient updates are not supported.")