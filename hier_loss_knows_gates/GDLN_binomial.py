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
plt.style.use('ggplot')
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

def gen_masks(num_output, num_data, num_obj):
  masks = []
  for _ in range(7):
      masks = masks + [np.zeros((num_output, num_data))]
  masks[0][:] = 1
  masks[1][:,0*num_obj:2*num_obj] = 1
  masks[2][:,1*num_obj:3*num_obj] = 1
  masks[3][:,0*num_obj:1*num_obj] = 1
  masks[3][:,2*num_obj:3*num_obj] = 1
  masks[4][:,0*num_obj:1*num_obj] = 1
  masks[5][:,1*num_obj:2*num_obj] = 1
  masks[6][:,2*num_obj:3*num_obj] = 1
  return masks

@jit
def module_predict(params, inputs):
  # Propagate data forward through the network
  return jnp.dot(params[1], jnp.dot(params[0], inputs))

@jit
def module_predict_hidden(params, inputs):
  # Propagate data forward through the network
  return jnp.dot(params[0], inputs)

@jit
def er_predict(params, inputs):
  return jnp.dot(params[1], jnp.maximum(0,jnp.dot(params[0], inputs)))

@jit
def gates_predict(gate_params, inputs):
  # Propagate data forward through the network
  net_out = sigmoid(jnp.dot(gate_params[1], jnp.dot(gate_params[0], inputs))) # + gate_params[1][1])
  return net_out

def predict(modules_params, masks, inputs, gates, key):
  output = jnp.zeros((60, inputs.shape[1]))
  for m in range(len(modules_params)):
      output = output.at[:].add(gates[m]*masks[m]*module_predict(modules_params[m], inputs)) 
  return output #gates[0]*Y + gates[1]*-0.1

def predict_hidden(modules_params, masks, inputs, gates, key):
  output = jnp.zeros((num_hidden, inputs.shape[1]))
  for m in range(len(modules_params)-1):
      output = output.at[:].add(gates[m]*module_predict_hidden(modules_params[m], inputs))
  return output

def loss(modules_params, gate_params, masks, batch, lkey):
  # Loss over a batch of data
  lkey1, lkey2 = random.split(lkey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(lkey1, gate_probs, gates.shape)
  rand_action = jax.random.bernoulli(lkey2, 0.5, gates.shape).astype(int)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, masks, inputs, actions, lkey)
  return jnp.mean(jnp.sum(jnp.power(preds - targets,2), axis=0))

def er_loss(er_params, modules_params, gate_params, masks, batch, lkey):
  # Loss over a batch of data
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  actions = gates
  preds = predict(modules_params, masks, inputs, actions, lkey)
  actual_loss =  jnp.sum(jnp.power(preds - targets,2), axis=0)
  est_loss = er_predict(er_params, jnp.vstack([inputs, actions]))
  return jnp.mean(jnp.power((est_loss - actual_loss),2))

def expected_reward(gate_params, er_params, modules_params, masks, batch, gate_reg_rate, ekey):
  ekey1, ekey2 = random.split(ekey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(ekey1, gate_probs, gates.shape)
  rand_action = jax.random.bernoulli(ekey2, 0.5, gates.shape)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, masks, inputs, actions, ekey)
  indiv_loss = ( er_predict(er_params, jnp.vstack([inputs, actions]))- jnp.sum(jnp.power(preds - targets,2), axis=0) )*actions*gate_probs
  return jnp.mean(indiv_loss, axis=0) - gate_reg_rate*jnp.sum(gate_probs)

@jit
def mean_batch_grads(batch_grads):
  return [jnp.mean(batch_grads[i], axis=0) for i in range(len(batch_grads))]

def statistics(modules_params, er_params, gate_params, masks, batch, skey):
  skey1, skey2 = random.split(skey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(skey1, gate_probs, gates.shape)
  rand_action = jax.random.bernoulli(skey2, 0.5, gates.shape)
  actions = gates #*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, masks, inputs, actions, skey)
  actual_indiv_loss = jnp.sum(jnp.power((preds - targets),2), axis=0)
  est_loss = er_predict(er_params, jnp.vstack([inputs, actions]))
  er_loss = jnp.mean(jnp.power((est_loss - actual_indiv_loss),2))
  expect_reward = ( jnp.sum(jnp.power(preds - targets,2), axis=0) - est_loss )*actions*gate_probs
  norms = []
  actual_norms = []
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int) # do here for plotting without affecting loss stats
  for m in range(len(modules_params)):
      norms.append(jnp.mean(actions[m]*jnp.linalg.norm(jnp.dot(modules_params[m][1],modules_params[m][0]),'fro')))
      actual_norms.append(jnp.mean(gates[m]*jnp.linalg.norm(jnp.dot(modules_params[m][1],modules_params[m][0]),'fro')))
  norms = jnp.array(norms)
  actual_norms = jnp.array(actual_norms)
  return jnp.mean(actual_indiv_loss), er_loss, expect_reward, est_loss, gate_probs, norms, actual_norms

if __name__ == "__main__":
  
  def weight_step(modules_params, gate_params, masks, batch, key):
    weight_grads = grad(loss)(modules_params, gate_params, masks, batch, key)
    return [[w - weight_step_size * dw for w,dw in zip(modules_params[m], weight_grads[m])] for m in range(len(modules_params))]
 
  def er_step(er_params, gate_params, modules_params, masks, batch, key):
    er_grads = grad(er_loss)(er_params, modules_params, gate_params, masks, batch, key)
    return [w - er_step_size * dw for w, dw in zip(er_params, er_grads)]
 
  def gate_step(gate_params, er_params, modules_params, masks, batch, gate_reg_rate, key):
    indiv_gate_grads = jacrev(expected_reward)(gate_params, er_params, modules_params, masks, batch, gate_reg_rate, key)
    gate_grads = mean_batch_grads(indiv_gate_grads)
    return [w + gate_step_size * dw for w, dw in zip(gate_params, gate_grads)]

  # Data Hyper-parameters
  num_obj = 8
  X,Y = gen_data3(num_obj, diff_struct = True)
  batch_size = X.shape[1]
  new_cmap = combine_cmaps(plt.cm.RdGy_r, plt.cm.BrBG)

  # Training hyper-parameters
  num_hidden = 100
  gate_layer_sizes = [num_obj+3, num_hidden, 7]
  er_layer_sizes = [num_obj+3 + gate_layer_sizes[2], num_hidden, 1]
  weight_step_size = 1e-1
  gate_step_size = 1e-0
  er_step_size = 1e-2 #2e-3
  param_scale = 0.001/float(num_hidden) #0.02/float(num_hidden) (for GDLN work)
  num_epochs = 2000
  warm_up_epochs = 800
  mds_sample_rate = 10
  gate_reg_rate = 0.0
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init
  key = random.PRNGKey(seed)
  run_idx = sys.argv[1]
  print("Run Index: ", run_idx)

  # Holds the SV trajectories and loss values to be plotted
  losses = np.zeros( num_epochs )
  er_losses = np.zeros( num_epochs )
  module_enorms = np.zeros( (num_epochs, 7) )
  chosen_enorms = np.zeros( (num_epochs, 7) )
  mds = np.zeros( (int(num_epochs/mds_sample_rate)+1,int(num_hidden),X.shape[1]) )

  # Create Modules
  modules_params = [init_random_params(param_scale, [X.shape[0], num_hidden, Y.shape[0]], seed) for _ in range(7)]
  masks = gen_masks(Y.shape[0], Y.shape[1], num_obj)

  gate_params = init_random_params(param_scale, gate_layer_sizes, seed)
  er_params = init_random_params(param_scale, er_layer_sizes, seed)

  for epoch in range(warm_up_epochs):
      key, module_key, er_key, gating_key, stats_key = random.split(key, num=5)
      er_params = er_step(er_params, gate_params, modules_params, masks, (X,Y), er_key)
      print('Epoch: ', epoch, ' ', str(er_loss(er_params, modules_params, gate_params, masks, (X,Y), stats_key)))

  print("Starting Training")
  i = 0
  for epoch in range(num_epochs):
      #os.system('clear')
      key, module_key, er_key, gating_key, stats_key = random.split(key, num=5)
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        "+str(epoch)+"   "+str(i)+"       $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      modules_params = weight_step(modules_params, gate_params, masks, (X,Y), module_key)
      er_params = er_step(er_params, gate_params, modules_params, masks, (X,Y), er_key) #(X_val,Y_val)
      gate_params = gate_step(gate_params, er_params, modules_params, masks, (X,Y), gate_reg_rate, gating_key)
      i = ((i + 1) % 20)
      
      print("############################################ Training Metrics #################################################")
      losses[epoch], er_losses[epoch], expect_reward, exp_losses, used_gates, module_enorms[epoch], chosen_enorms[epoch] =\
              statistics(modules_params, er_params, gate_params, masks, (X,Y), stats_key) 
      print('Epoch: ',epoch, 'i:',i, ', Loss: ',losses[epoch],', ER_Loss: ',er_losses[epoch], '\nExpected Losses\n', exp_losses,\
            ',\nModules Chosen:\n',chosen_enorms[epoch], ',\nModules Used:\n',module_enorms[epoch])
      #os.system('clear')
      if (epoch % mds_sample_rate) == 0:
          gate_probs = gates_predict(gate_params, X)
          gates = jnp.where(gate_probs > 0.5, 1, 0)
          preds_hidden = predict_hidden(modules_params, masks, X, gates, stats_key)
          mds[int(epoch/mds_sample_rate)] = preds_hidden
      if (epoch == 500) or (epoch == 1000) or (epoch == 1990):
          plot_outputs(predict(modules_params, masks, X, gates, stats_key), new_cmap, 'plots/'+str(epoch)+'_bin_out.pdf', vmin=-1, vmax=1)

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

  norm_names = [r'Common', r'C1 and C2', r'C2 and C3', r'C1 and C3', r'C1', r'C2','C3']
  for m in range(module_enorms.shape[1]):
      plt.plot(module_enorms[:,m], label=norm_names[m])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.grid()
  plt.legend(loc='upper left')
  plt.savefig('train_norms/'+str(run_idx)+'_binomial.pdf')
  plt.close()

  np.savetxt('n_runs/'+str(run_idx)+'_binomial_train_losses.txt', losses)
  np.savetxt('n_runs/'+str(run_idx)+'_binomial_train_module_enorms.txt', chosen_enorms)
  np.savetxt('n_runs/'+str(run_idx)+'_binomial_mds.txt',mds.transpose(0,2,1).reshape((int(num_epochs/mds_sample_rate)+1)*X.shape[1],-1))
