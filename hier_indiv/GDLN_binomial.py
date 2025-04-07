import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
import os
from gen_data import gen_data3
from replot import plot_outputs, combine_cmaps

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']="\\usepackage{amsmath}"

@jit
def sigmoid(x):
    return 1.0/(1+jnp.exp(-x))

def init_random_params(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  return [np.random.normal(0.0, scale, (n, m)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_random_params_bias(scale, layer_sizes, rng):
  return [(np.random.normal(0,scale,(n, m)), np.random.normal(0,scale,(n,1))) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def er_predict(params, inputs):
  return jnp.dot(params[1], jnp.maximum(0,jnp.dot(params[0], inputs)))

#@jit
def gates_predict(gate_params, inputs):
  # Propagate data forward through the network
  net_out = sigmoid(jnp.dot(gate_params[1][0], jnp.dot(gate_params[0][0], inputs) + gate_params[0][1])) # + gate_params[1][1])
  return net_out

#@jit
def predict(params, inputs, gates):
  # Propagate data forward through the network
  return jnp.dot(params[1], gates*jnp.dot(params[0], inputs))

def predict_hidden(params, inputs, gates):
  # Propagate data forward through the network
  return gates*jnp.dot(params[0], inputs)

def loss(forward_params, gate_params, batch, lkey):
  # Loss over a batch of data
  lkey1, lkey2 = random.split(lkey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_probs = jnp.copy(gate_probs)
  keep_action = jax.random.bernoulli(lkey1, keep_probs, gates.shape)
  rand_action = jax.random.bernoulli(lkey2, 0.5, gates.shape).astype(int)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(forward_params, inputs, actions)
  return jnp.mean(jnp.sum(jnp.power(preds - targets,2), axis=0))

def er_loss(er_params, forward_params, gate_params, batch, lkey):
  # Loss over a batch of data
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  actions = gates
  preds = predict(forward_params, inputs, actions)
  actual_loss =  jnp.sum(jnp.power(preds - targets,2), axis=0)
  est_loss = er_predict(er_params, inputs)
  return jnp.mean(jnp.power((est_loss - actual_loss),2))

def expected_reward(gate_params, er_params, forward_params, batch, ekey):
  ekey1, ekey2 = random.split(ekey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_probs = jnp.copy(gate_probs)
  keep_action = jax.random.bernoulli(ekey1, keep_probs, gates.shape)
  rand_action = jax.random.bernoulli(ekey2, 0.5, gates.shape)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(forward_params, inputs, actions)
  indiv_loss = ( er_predict(er_params, inputs) - jnp.sum(jnp.power(preds - targets,2), axis=0) )*actions*gate_probs
  return jnp.mean(indiv_loss, axis=0)

@jit
def mean_batch_grads(batch_grads):
  return [ [jnp.mean(batch_grads[i][j], axis=0) for j in range(len(batch_grads[0]))] for i in range(len(batch_grads)) ]

def statistics(forward_params, er_params, gate_params, batch, skey):
  skey1, skey2 = random.split(skey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_probs = jnp.copy(gate_probs)
  keep_action = jax.random.bernoulli(skey1, keep_probs, gates.shape)
  rand_action = jax.random.bernoulli(skey2, 0.5, gates.shape)
  actions = gates #*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(forward_params, inputs, actions)
  actual_indiv_loss = jnp.sum(jnp.power((preds - targets),2), axis=0)
  est_loss = er_predict(er_params, inputs)
  er_loss = jnp.mean(jnp.power((est_loss - actual_indiv_loss),2))
  expect_reward = ( jnp.sum(jnp.power(preds - targets,2), axis=0) - est_loss )*actions*gate_probs
  return jnp.mean(actual_indiv_loss), er_loss, expect_reward, est_loss, gate_probs

if __name__ == "__main__":
  
  def weight_step(forward_params, gate_params, batch, key):
    weight_grads = grad(loss)(forward_params, gate_params, batch, key)
    return [w - weight_step_size * dw for w,dw in zip(forward_params, weight_grads)]
 
  def er_step(er_params, gate_params, forward_params, batch, key):
    er_grads = grad(er_loss)(er_params, forward_params, gate_params, batch, key)
    return [w - er_step_size * dw for w, dw in zip(er_params, er_grads)]
 
  def gate_step(gate_params, er_params, forward_params, batch, key):
    indiv_gate_grads = jacrev(expected_reward)(gate_params, er_params, forward_params, batch, key)
    gate_grads = mean_batch_grads(indiv_gate_grads)
    return [(w + gate_step_size * dw, b + gate_step_size * db) for (w, b), (dw, db) in zip(gate_params, gate_grads)]

  # Data Hyper-parameters
  num_obj = 8
  X,Y = gen_data3(num_obj, diff_struct = True)
  batch_size = X.shape[1]
  new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.BrBG)

  # Training hyper-parameters
  num_hidden = 100
  forward_layer_sizes = [num_obj+3, num_hidden, Y.shape[0]]
  gate_layer_sizes = [num_obj+3, num_hidden, num_hidden]
  er_layer_sizes = [num_obj+3, num_hidden, 1]
  weight_step_size = 1e-1
  gate_step_size = 1e-0 #1e-0
  er_step_size = 2e-3 #1e-2
  param_scale = 0.001/float(num_hidden) #0.02/float(num_hidden) (for GDLN work)
  num_epochs = 10000
  warm_up_epochs = 800
  mds_sample_rate = 10
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init
  key = random.PRNGKey(seed)
  run_idx = sys.argv[1]
  print("Run Index: ", run_idx)

  # Holds the SV trajectories and loss values to be plotted
  losses = np.zeros( num_epochs )
  er_losses = np.zeros( num_epochs )
  mds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden),X.shape[1]) )

  # Create Modules
  forward_params = init_random_params(param_scale, forward_layer_sizes, seed)
  gate_params = init_random_params_bias(param_scale, gate_layer_sizes, seed)
  er_params = init_random_params(param_scale, er_layer_sizes, seed)
  ds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden),X.shape[1]) )

  for epoch in range(warm_up_epochs):
      key, forward_key, er_key, gating_key, stats_key = random.split(key, num=5)
      er_params = er_step(er_params, gate_params, forward_params, (X,Y), er_key)
      print('Epoch: ', epoch, ' ', str(er_loss(er_params, forward_params, gate_params, (X,Y), stats_key)))

  i = 0
  for epoch in range(num_epochs):
      #os.system('clear')
      key, forward_key, er_key, gating_key, stats_key = random.split(key, num=5)
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        "+str(epoch)+"   "+str(i)+"       $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      forward_params = weight_step(forward_params, gate_params, (X,Y), forward_key)
      er_params = er_step(er_params, gate_params, forward_params, (X,Y), er_key)
      gate_params = gate_step(gate_params, er_params, forward_params, (X,Y), gating_key)
      i = ((i + 1) % 20)

      print("############################################ Training Metrics #################################################")
      losses[epoch], er_losses[epoch], expect_reward, exp_losses, used_gates =\
              statistics(forward_params, er_params, gate_params, (X,Y), stats_key) 
      print('Epoch: ',epoch, 'i:',i, ', Loss: ',losses[epoch],', ER_Loss: ',er_losses[epoch], '\nExpected Losses\n', exp_losses)
      os.system('clear')
      if (epoch % mds_sample_rate) == 0:
          gate_probs = gates_predict(gate_params, X)
          gates = jnp.where(gate_probs > 0.5, 1, 0)
          preds_hidden = predict_hidden(forward_params, X, gates)
          mds[int(epoch/mds_sample_rate)] = preds_hidden
      if (epoch == 500) or (epoch == 1000) or (epoch == 1990):
          plot_outputs(predict(forward_params, X, gates), new_cmap, 'plots/'+str(epoch)+'_bin_out.pdf', vmin=-1, vmax=1)

  end_gate_probs = gates_predict(gate_params, X)
  plt.imshow(end_gate_probs, cmap='BuPu', vmin=0, vmax=1)
  plt.colorbar()
  plt.savefig('end_gates/'+str(run_idx)+'_binomial.pdf',dpi=400)
  plt.close()

  # Plot losses
  plt.plot(losses, color='red', label='Train Losses')
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper right')
  plt.grid()
  plt.savefig('losses/'+str(run_idx)+'_binomial.pdf')
  plt.close()
 
  np.savetxt('n_runs/'+str(run_idx)+'_binomial_train_losses.txt', losses)
  np.savetxt('n_runs/'+str(run_idx)+'_binomial_mds.txt',mds.transpose(0,2,1).reshape((int(num_epochs/mds_sample_rate)+1)*X.shape[1],-1))
