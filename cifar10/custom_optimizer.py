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
	def __init__(self, learning_rate, optimizer_code, use_locking=False, name="CustomOptimizer"):
		super(CustomOptimizer, self).__init__(use_locking, name)
		self._lr = learning_rate
		self._beta1 = 0.9
		self._beta2 = 0.9
		self._beta3 = 0.999 
		self._rmsprop_beta = 0.9
		self._epsilon = 1e-8
		self._code = optimizer_code
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
		self._epsilon_t = ops.convert_to_tensor(self._epsilon, name="epsilon_t")

	def _create_slots(self, var_list):
		# default var_list as all trainable variables
		# Create slots for the first and second moments.
		for v in var_list:
			self._zeros_slot(v, "m", self._name)
			self._zeros_slot(v, "v", self._name)
			self._zeros_slot(v, "r", self._name)
			self._zeros_slot(v, "rmsprop_v", self._name)

	def tf_unary(self, x, key):
		epsilon_t = math_ops.cast(self._epsilon_t, x.dtype.base_dtype)
		unary = tf.case({
			tf.equal(key,0):lambda: x,
			tf.equal(key,1):lambda: -x,
			tf.equal(key,2):lambda: tf.exp(x),
			tf.equal(key,3):lambda: tf.log(tf.abs(x) + epsilon_t),
			tf.equal(key,4):lambda: tf.sqrt(tf.abs(x) + epsilon_t),
			tf.equal(key,5):lambda: tf.clip_by_value(x, -1e-5, 1e-5),
			tf.equal(key,6):lambda: tf.clip_by_value(x, -1e-4, 1e-4),
			tf.equal(key,7):lambda: tf.clip_by_value(x, -1e-3, 1e-3),
			tf.equal(key,8):lambda: tf.cond(tf.equal(tf.multinomial(tf.log([[1.0, 9.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			tf.equal(key,9):lambda: tf.cond(tf.equal(tf.multinomial(tf.log([[3.0, 7.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			tf.equal(key,10):lambda: tf.cond(tf.equal(tf.multinomial(tf.log([[5.0, 5.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			tf.equal(key,11):lambda: tf.sign(x)
		}, default=lambda:x, exclusive=True)
		return unary

	def unary(self, var, key):
		epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
		unary_opts = {
			0:lambda x: x,
			1:lambda x: -x,
			2:lambda x: tf.exp(x),
			3:lambda x: tf.log(tf.abs(var) + epsilon_t),
			4:lambda x: tf.sqrt(tf.abs(var) + epsilon_t),
			5:lambda x: tf.clip_by_value(x, -1e-5, 1e-5),
			6:lambda x: tf.clip_by_value(x, -1e-4, 1e-4),
			7:lambda x: tf.clip_by_value(x, -1e-3, 1e-3),
			8:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[1.0, 9.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			9:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[3.0, 7.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			10:lambda x: tf.cond(tf.equal(tf.multinomial(tf.log([[5.0, 5.0]]), 1)[0][0], 0), lambda:tf.zeros(tf.shape(x)), lambda:x),
			11:lambda x: tf.sign(x)
		}
		return unary_opts[key](var)

	def tf_binary(self, x, y, key):
		delta = 1e-8
		binary = tf.case({
			tf.equal(key,0):lambda: x + y,
			tf.equal(key,1):lambda: x - y,
			tf.equal(key,2):lambda: x * y,
			tf.equal(key,3):lambda: x / (y + delta),
			tf.equal(key,4):lambda: tf.pow(x, y),
			tf.equal(key,5):lambda: x
		}, default=lambda:x, exclusive=True)
		return binary

	def binary(self, var1, var2, key):
		delta = 1e-8
		binary_opts = {
			0:lambda x, y: x + y,
			1:lambda x, y: x - y,
			2:lambda x, y: x * y,
			3:lambda x, y: x / (y + delta),
			4:lambda x, y: tf.pow(x, y),
			5:lambda x, y: x
		}
		return binary_opts[key](var1, var2)

	def _apply_dense(self, grad, var):
		operand_names = ['g', 'g2', 'g3', 'm', 'v', 'r', 'sign_g', 'sign_m', '1', '2', 'epsilon', '10-4w', '10-3w', '10-2w', '10-1w', 'Adam', 'RMSprop']
		operands = {oprd:None for oprd in operand_names}
		#grad = tf.Print(grad, [tf.shape(grad)], 'grad shape')
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
		operands_list = [
			grad,
			grad2,
			grad3,
			m_t,
			v_t,
			r_t,
			tf.sign(grad),
			tf.sign(m_t),
			tf.constant(1.0, shape=grad.get_shape().as_list()),
			tf.constant(2.0, shape=grad.get_shape().as_list()),
			tf.fill(tf.shape(grad), epsilon_t),
			1e-4*var,
			1e-3*var,
			1e-2*var,
			1e-1*var,
			adam,
			rmsprop
		]
		operands = {
			0: grad,
			1: grad2,
			2: grad3,
			3: m_t,
			4: v_t,
			5: r_t,
			6: tf.sign(grad),
			7: tf.sign(m_t),
			8: tf.constant(1.0, shape=grad.get_shape().as_list()),
			9: tf.constant(2.0, shape=grad.get_shape().as_list()),
			10: tf.fill(tf.shape(grad), epsilon_t),
			11: 1e-4*var,
			12: 1e-3*var,
			13: 1e-2*var,
			14: 1e-1*var,
			15: adam,
			16: rmsprop
		}
		# decipher self._code
		def decipher(code):
			# code: list of integers
			if len(code) == 2:
				# op1, unary1
				operand1 = operands[code[0]]
				unary1 = self.unary(operand1, code[1])
				result = unary1
			elif len(code) == 5:
				# op1, op2, u1, u2, b1
				operand1 = operands[code[0]]
				operand2 = operands[code[1]]
				unary1 = self.unary(operand1, code[2])
				unary2 = self.unary(operand2, code[3])
				binary1 = self.binary(unary1, unary2, code[4])
				result = binary1
			return result
		def tf_decipher(code):
			def _length5(): 
				operand1 = tf.gather(operands_list, code[0])
				#print('operand1', operand1)
				operand2 = tf.gather(operands_list, code[1])
				#print('operand2', operand2)
				unary1 = self.tf_unary(operand1, code[2])
				#print('unary1', unary1)
				unary2 = self.tf_unary(operand2, code[3])
				#print('unary2', unary2)
				binary1 = self.tf_binary(unary1, unary2, code[4])
				#print('binary1', binary1)
				return binary1
			def _length2():
				operand1 = tf.gather(operands_list, code[0])
				unary1 = self.tf_unary(operand1, code[1])
				return unary1
			return _length5()
			#return tf.case({tf.equal(tf.size(code), 5):_length5, tf.equal(tf.size(code), 2):_length2}, default=_length5, exclusive=True)
		#result = decipher(self._code)
		result = tf_decipher(self._code)
		#print result

		var_update = state_ops.assign_sub(var, lr_t * result, use_locking=self._use_locking)
		#var_update = state_ops.assign_sub(var, operands[17], use_locking=self._use_locking)

		#var_update = state_ops.assign_sub(var, lr_t*grad*tf.exp(tf.sign(grad)*tf.sign(m_t))) #Update 'ref' by subtracting 'value
		#Create an op that groups multiple operations.
		#When this op finishes, all ops in input have finished
		return control_flow_ops.group(*[var_update, m_t])

	def _apply_sparse(self, grad, var):
		raise NotImplementedError("Sparse gradient updates are not supported.")