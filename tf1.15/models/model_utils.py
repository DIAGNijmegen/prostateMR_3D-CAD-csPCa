from __future__ import unicode_literals, absolute_import, print_function, division
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.eager import context
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.training import optimizer
from tensorflow.python.ops.clip_ops import clip_by_value


'''
Prostate Cancer Detection in bpMRI
Script:         Model Utilities (LR, Activations)
Contributor:    anindox8
Target Organ:   Prostate
Target Classes: Benign(0), Malignant(1)
Update:         18/08/2020

'''


# Cyclic Learning Rate [Ref:mhmoodlan/cyclic-learning-rate]
def cyclic_learning_rate(global_step,
                         learning_rate=0.01,
                         max_lr=0.1,
                         step_size=20.,
                         gamma=0.99994,
                         mode='triangular',
                         name=None):
  """
  [1] L. Smith (2017) "Cyclical Learning Rates for Training Neural Networks", IEEE WACV
  """
  if global_step is None:
    raise ValueError("global_step is required for cyclic_learning_rate.")
  with ops.name_scope(name, "CyclicLearningRate",
                      [learning_rate, global_step]) as name:
    learning_rate = ops.convert_to_tensor(learning_rate, name="learning_rate")
    dtype = learning_rate.dtype
    global_step = math_ops.cast(global_step, dtype)
    step_size = math_ops.cast(step_size, dtype)
    def cyclic_lr():

      double_step = math_ops.multiply(2., step_size)
      global_div_double_step = math_ops.divide(global_step, double_step)
      cycle = math_ops.floor(math_ops.add(1., global_div_double_step))

      double_cycle = math_ops.multiply(2., cycle)
      global_div_step = math_ops.divide(global_step, step_size)
      tmp = math_ops.subtract(global_div_step, double_cycle)
      x = math_ops.abs(math_ops.add(1., tmp))

      a1 = math_ops.maximum(0., math_ops.subtract(1., x))
      a2 = math_ops.subtract(max_lr, learning_rate)
      clr = math_ops.multiply(a1, a2)
      if mode == 'triangular2':
        clr = math_ops.divide(clr, math_ops.cast(math_ops.pow(2, math_ops.cast(
            cycle-1, tf.int32)), tf.float32))
      if mode == 'exp_range':
        clr = math_ops.multiply(math_ops.pow(gamma, global_step), clr)
      return math_ops.add(clr, learning_rate, name=name)
    if not context.executing_eagerly():
      cyclic_lr = cyclic_lr()
    return cyclic_lr


# ReLU Activation Function [Ref:DLTK]
def leaky_relu(inputs, alpha=0.1):
    return tf.maximum(inputs, alpha * inputs)


# Probabilistic ReLU Activation Function [Ref:DLTK]
def prelu(inputs, alpha_initializer=tf.constant_initializer()):
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=alpha_initializer)
    return leaky_relu(inputs, alpha)


# AdaBound Optimizer [Ref:taki0112/AdaBound-Tensorflow]
class AdaBoundOptimizer(optimizer.Optimizer):
  """
  [1] L. Luo (2019) "Adaptive Gradient Methods with Dynamic Bound of Learning Rate", ICLR
  """
  def __init__(self, learning_rate=0.001, final_lr=0.1, beta1=0.9, beta2=0.999,
               gamma=1e-3, epsilon=1e-8, amsbound=False,
               use_locking=False, name="AdaBound"):
      super(AdaBoundOptimizer, self).__init__(use_locking, name)
      self._lr        = learning_rate
      self._final_lr  = final_lr
      self._beta1     = beta1
      self._beta2     = beta2
      self._epsilon   = epsilon

      self._gamma     = gamma
      self._amsbound  = amsbound

      self._lr_t      = None
      self._beta1_t   = None
      self._beta2_t   = None
      self._epsilon_t = None

  def _create_slots(self, var_list):
      first_var = min(var_list, key=lambda x: x.name)

      graph = None if context.executing_eagerly() else ops.get_default_graph()
      create_new = self._get_non_slot_variable("beta1_power", graph) is None
      if not create_new and context.in_graph_mode():
          create_new = (self._get_non_slot_variable("beta1_power", graph).graph is not first_var.graph)

      if create_new:
          self._create_non_slot_variable(initial_value=self._beta1,
                                         name="beta1_power",
                                         colocate_with=first_var)
          self._create_non_slot_variable(initial_value=self._beta2,
                                         name="beta2_power",
                                         colocate_with=first_var)
          self._create_non_slot_variable(initial_value=self._gamma,
                                         name="gamma_multi",
                                         colocate_with=first_var)
      for v in var_list :
          self._zeros_slot(v, "m", self._name)
          self._zeros_slot(v, "v", self._name)
          self._zeros_slot(v, "vhat", self._name)

  def _prepare(self):
      self._lr_t = ops.convert_to_tensor(self._lr)
      self._base_lr_t = ops.convert_to_tensor(self._lr)
      self._beta1_t = ops.convert_to_tensor(self._beta1)
      self._beta2_t = ops.convert_to_tensor(self._beta2)
      self._epsilon_t = ops.convert_to_tensor(self._epsilon)
      self._gamma_t = ops.convert_to_tensor(self._gamma)

  def _apply_dense(self, grad, var):
      graph = None if context.executing_eagerly() else ops.get_default_graph()
      beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
      beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
      lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
      base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
      beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
      beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
      epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
      gamma_multi = math_ops.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

      step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
      final_lr = self._final_lr * lr_t / base_lr_t
      lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
      upper_bound = final_lr * (1. + 1. / (gamma_multi))

      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, "m")
      m_scaled_g_values = grad * (1 - beta1_t)
      m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

      # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
      v = self.get_slot(var, "v")
      v_scaled_g_values = (grad * grad) * (1 - beta2_t)
      v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

      # amsgrad
      vhat = self.get_slot(var, "vhat")
      if self._amsbound :
          vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
          v_sqrt = math_ops.sqrt(vhat_t)
      else :
          vhat_t = state_ops.assign(vhat, vhat)
          v_sqrt = math_ops.sqrt(v_t)


      # Compute the bounds
      step_size_bound = step_size / (v_sqrt + epsilon_t)
      bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

      var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)
      return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

  def _resource_apply_dense(self, grad, var):
      graph = None if context.executing_eagerly() else ops.get_default_graph()
      beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), grad.dtype.base_dtype)
      beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), grad.dtype.base_dtype)
      lr_t = math_ops.cast(self._lr_t, grad.dtype.base_dtype)
      base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
      beta1_t = math_ops.cast(self._beta1_t, grad.dtype.base_dtype)
      beta2_t = math_ops.cast(self._beta2_t, grad.dtype.base_dtype)
      epsilon_t = math_ops.cast(self._epsilon_t, grad.dtype.base_dtype)
      gamma_multi = math_ops.cast(self._get_non_slot_variable("gamma_multi", graph=graph), var.dtype.base_dtype)

      step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
      final_lr = self._final_lr * lr_t / base_lr_t
      lower_bound = final_lr * (1. - 1. / (gamma_multi + 1.))
      upper_bound = final_lr * (1. + 1. / (gamma_multi))

      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, "m")
      m_scaled_g_values = grad * (1 - beta1_t)
      m_t = state_ops.assign(m, beta1_t * m + m_scaled_g_values, use_locking=self._use_locking)

      # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
      v = self.get_slot(var, "v")
      v_scaled_g_values = (grad * grad) * (1 - beta2_t)
      v_t = state_ops.assign(v, beta2_t * v + v_scaled_g_values, use_locking=self._use_locking)

      # amsgrad
      vhat = self.get_slot(var, "vhat")
      if self._amsbound:
          vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
          v_sqrt = math_ops.sqrt(vhat_t)
      else:
          vhat_t = state_ops.assign(vhat, vhat)
          v_sqrt = math_ops.sqrt(v_t)

      # Compute the bounds
      step_size_bound = step_size / (v_sqrt + epsilon_t)
      bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

      var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)

      return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

  def _apply_sparse_shared(self, grad, var, indices, scatter_add):
      graph = None if context.executing_eagerly() else ops.get_default_graph()
      beta1_power = math_ops.cast(self._get_non_slot_variable("beta1_power", graph=graph), var.dtype.base_dtype)
      beta2_power = math_ops.cast(self._get_non_slot_variable("beta2_power", graph=graph), var.dtype.base_dtype)
      lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
      base_lr_t = math_ops.cast(self._base_lr_t, var.dtype.base_dtype)
      beta1_t = math_ops.cast(self._beta1_t, var.dtype.base_dtype)
      beta2_t = math_ops.cast(self._beta2_t, var.dtype.base_dtype)
      epsilon_t = math_ops.cast(self._epsilon_t, var.dtype.base_dtype)
      gamma_t = math_ops.cast(self._gamma_t, var.dtype.base_dtype)

      step_size = (lr_t * math_ops.sqrt(1 - beta2_power) / (1 - beta1_power))
      final_lr = self._final_lr * lr_t / base_lr_t
      lower_bound = final_lr * (1. - 1. / (gamma_t + 1.))
      upper_bound = final_lr * (1. + 1. / (gamma_t))

      # m_t = beta1 * m + (1 - beta1) * g_t
      m = self.get_slot(var, "m")
      m_scaled_g_values = grad * (1 - beta1_t)
      m_t = state_ops.assign(m, m * beta1_t, use_locking=self._use_locking)
      with ops.control_dependencies([m_t]):
          m_t = scatter_add(m, indices, m_scaled_g_values)

      # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
      v = self.get_slot(var, "v")
      v_scaled_g_values = (grad * grad) * (1 - beta2_t)
      v_t = state_ops.assign(v, v * beta2_t, use_locking=self._use_locking)
      with ops.control_dependencies([v_t]):
          v_t = scatter_add(v, indices, v_scaled_g_values)

      # amsgrad
      vhat = self.get_slot(var, "vhat")
      if self._amsbound:
          vhat_t = state_ops.assign(vhat, math_ops.maximum(v_t, vhat))
          v_sqrt = math_ops.sqrt(vhat_t)
      else:
          vhat_t = state_ops.assign(vhat, vhat)
          v_sqrt = math_ops.sqrt(v_t)

      # Compute the bounds
      step_size_bound = step_size / (v_sqrt + epsilon_t)
      bounded_lr = m_t * clip_by_value(step_size_bound, lower_bound, upper_bound)

      var_update = state_ops.assign_sub(var, bounded_lr, use_locking=self._use_locking)

      return control_flow_ops.group(*[var_update, m_t, v_t, vhat_t])

  def _apply_sparse(self, grad, var):
      return self._apply_sparse_shared(
          grad.values, var, grad.indices,
          lambda x, i, v: state_ops.scatter_add(  # pylint: disable=g-long-lambda
              x, i, v, use_locking=self._use_locking))

  def _resource_scatter_add(self, x, i, v):
      with ops.control_dependencies(
              [resource_variable_ops.resource_scatter_add(x, i, v)]):
          return x.value()

  def _resource_apply_sparse(self, grad, var, indices):
      return self._apply_sparse_shared(
          grad, var, indices, self._resource_scatter_add)

  def _finish(self, update_ops, name_scope):
      # Update the power accumulators.
      with ops.control_dependencies(update_ops):
          graph = None if context.executing_eagerly() else ops.get_default_graph()
          beta1_power = self._get_non_slot_variable("beta1_power", graph=graph)
          beta2_power = self._get_non_slot_variable("beta2_power", graph=graph)
          gamma_multi = self._get_non_slot_variable("gamma_multi", graph=graph)
          with ops.colocate_with(beta1_power):
              update_beta1 = beta1_power.assign(
                  beta1_power * self._beta1_t,
                  use_locking=self._use_locking)
              update_beta2 = beta2_power.assign(
                  beta2_power * self._beta2_t,
                  use_locking=self._use_locking)
              update_gamma = gamma_multi.assign(
                  gamma_multi + self._gamma_t,
                  use_locking=self._use_locking)
      return control_flow_ops.group(*update_ops + [update_beta1, update_beta2, update_gamma],
                                    name=name_scope)