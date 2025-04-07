import numpy as np
import numpy.random as npr
import matplotlib
import matplotlib.pyplot as plt
from jax import jit, grad, jacrev, random
import jax
import jax.numpy as jnp
import sys
import os

np.set_printoptions(threshold=np.inf, suppress=True, linewidth=200)
matplotlib.rcParams.update({'font.size': 14})
plt.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']="\\usepackage{amsmath}"

@jit
def sigmoid(x):
    return 1.0/(1+jnp.exp(-x))

def gen_binary_patterns(num_features):
    # Generates every unique binary digit possible with num_feature bits
    data = np.ones((2**num_features, num_features))*-1.0
    for j in np.arange(0, num_features, 1):
        step = 2**(j+1)
        idx = [list(range(i,i+int(step/2))) for i in np.arange(int(step/2),2**num_features,step)]
        idx = np.concatenate(idx)
        data[idx,j] = 1
    data = np.flip(data, axis=1)
    return data

def init_random_params(scale, layer_sizes, seed):
  # Returns a list of tuples where each tuple is the weight matrix and bias vector for a layer
  np.random.seed(seed)
  return [np.random.normal(0.0, scale, (n, m)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

def init_random_params_bias(scale, layer_sizes, rng):
  return [(np.random.normal(0,scale,(n, m)), np.random.normal(0,scale,(n,1))) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]

@jit
def module_predict(params, inputs):
  # Propagate data forward through the network
  return jnp.dot(params[1], jnp.dot(params[0], inputs))

@jit
def er_predict(params, inputs):
  return jnp.dot(params[1], jnp.maximum(0,jnp.dot(params[0], inputs)))

@jit
def gates_predict(gate_params, inputs):
  # Propagate data forward through the network
  net_out = sigmoid(jnp.dot(gate_params[1][0], jnp.dot(gate_params[0][0], inputs) + gate_params[0][1])) # + gate_params[1][1])
  return net_out

def predict(modules_params, modules_ranges, inputs, gates, key):
  output = jnp.zeros((n2+k2*(2**n1), inputs.shape[1]))
  for m in range(len(modules_params)):
      module_in_range, module_out_range = modules_ranges[m]
      output = output.at[module_out_range[0]:module_out_range[1]].set(\
               output[module_out_range[0]:module_out_range[1]] +\
               gates[m]*module_predict(modules_params[m], inputs[module_in_range[0]:module_in_range[1]])) 
  return output #gates[0]*Y + gates[1]*-0.1

def loss(modules_params, modules_ranges, gate_params, batch, rand_prob, lkey):
  # Loss over a batch of data
  lkey1, lkey2 = random.split(lkey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(lkey1, 1.0 - rand_prob, gates.shape)
  rand_action = jax.random.bernoulli(lkey2, 0.5, gates.shape).astype(int)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, modules_ranges, inputs, actions, lkey)
  return jnp.mean(jnp.sum(jnp.power(preds - targets,2), axis=0))

def er_loss(er_params, modules_params, modules_ranges, gate_params, batch, rand_prob, lkey):
  # Loss over a batch of data
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  actions = gates
  preds = predict(modules_params, modules_ranges, inputs, actions, lkey)
  actual_loss =  jnp.sum(jnp.power(preds - targets,2), axis=0)
  est_loss = er_predict(er_params, inputs)
  #print("Predicted Loss")
  #print(est_loss)
  #print("Actual Loss")
  #print(actual_loss)
  return jnp.mean(jnp.power((est_loss - actual_loss),2))

def expected_reward(gate_params, er_params, modules_params, modules_ranges, batch, rand_prob, ekey):
  ekey1, ekey2 = random.split(ekey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(ekey1, 1.0 - rand_prob, gates.shape)
  rand_action = jax.random.bernoulli(ekey2, 0.5, gates.shape)
  actions = gates*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, modules_ranges, inputs, actions, ekey)
  indiv_loss = ( er_predict(er_params, inputs) - jnp.sum(jnp.power(preds - targets,2), axis=0) )*actions*gate_probs
  return jnp.mean(indiv_loss, axis=0) - 0.2*jnp.sum(actions) # consider using gates instead of actions?

@jit
def mean_batch_grads(batch_grads):
  return [ [jnp.mean(batch_grads[i][j], axis=0) for j in range(len(batch_grads[0]))] for i in range(len(batch_grads)) ]

def statistics(modules_params, modules_ranges, er_params, gate_params, batch, rand_prob, skey):
  skey1, skey2 = random.split(skey, num=2)
  inputs, targets = batch
  gate_probs = gates_predict(gate_params, inputs)
  gates = jnp.where(gate_probs > 0.5, 1, 0)
  keep_action = jax.random.bernoulli(skey1, 1.0 - rand_prob, gates.shape)
  rand_action = jax.random.bernoulli(skey2, 0.5, gates.shape)
  actions = gates #*keep_action + rand_action*jnp.logical_not(keep_action).astype(int)
  preds = predict(modules_params, modules_ranges, inputs, actions, skey)
  actual_indiv_loss = jnp.sum(jnp.power((preds - targets),2), axis=0)
  actual_sys_loss = jnp.sum(jnp.power((preds[:n2] - targets[:n2]),2), axis=0)
  actual_non_loss = jnp.sum(jnp.power((preds[n2:] - targets[n2:]),2), axis=0)
  est_loss = er_predict(er_params, inputs)
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
  return jnp.mean(actual_indiv_loss), jnp.mean(actual_sys_loss), jnp.mean(actual_non_loss),\
                  er_loss, expect_reward, est_loss, gate_probs, norms/data_norms, actual_norms/data_norms

if __name__ == "__main__":
  
  def weight_step(modules_params, modules_ranges, gate_params, batch, rand_prob, key):
    weight_grads = grad(loss)(modules_params, modules_ranges, gate_params, batch, rand_prob, key)
    return [[w - weight_step_size * dw for w,dw in zip(modules_params[m], weight_grads[m])] for m in range(len(modules_params))]
 
  def er_step(er_params, gate_params, modules_params, modules_ranges, batch, rand_prob, key):
    er_grads = grad(er_loss)(er_params, modules_params, modules_ranges, gate_params, batch, rand_prob, key)
    return [w - er_step_size * dw for w, dw in zip(er_params, er_grads)]
 
  def gate_step(gate_params, er_params, modules_params, modules_ranges, batch, rand_prob, key):
    indiv_gate_grads = jacrev(expected_reward)(gate_params, er_params, modules_params, modules_ranges, batch, rand_prob, key)
    gate_grads = mean_batch_grads(indiv_gate_grads)
    return [(w + gate_step_size * dw, b + gate_step_size * db) for (w, b), (dw, db) in zip(gate_params, gate_grads)]

  # Data Hyper-parameters
  n1 = 6 #n1 num sys inputs #6
  n2 = 3 #n2 num sys outputs #1
  k1 = 3 #k1 num nonsys reps input #3
  k2 = 1  #k2 num nonsys reps output #1
  r = 2 #r scale #1

  # Training hyper-parameters
  num_training = 16
  num_validation = 16
  num_hidden = 60
  gate_layer_sizes = [n1+k1*(2**n1), 100, 9]
  er_layer_sizes = [n1+k1*(2**n1), 100, 1]
  weight_step_size = 1e-1
  gate_step_size = 1e-0 #1e-0
  er_step_size = 2e-3 #1e-2
  param_scale = 0.001/float(num_hidden) #0.02/float(num_hidden) (for GDLN work)
  num_epochs = 8000
  warm_up_epochs = 2000
  rand_prob = 0.7
  rand_drop_epoch = 1000
  seed = np.random.randint(0,100000) # can set seed here, for now it is random. The only randomness is in the network init
  key = random.PRNGKey(seed)
  run_idx = sys.argv[1]
  print("Run Index: ", run_idx)

  # Holds the SV trajectories and loss values to be plotted
  losses = np.zeros( num_epochs )
  sys_losses = np.zeros( num_epochs )
  non_losses = np.zeros( num_epochs )
  test_losses = np.zeros( num_epochs )
  sys_test_losses = np.zeros( num_epochs )
  non_test_losses = np.zeros( num_epochs )
  er_losses = np.zeros( num_epochs )
  test_er_losses = np.zeros( num_epochs )
  module_enorms = np.zeros( (num_epochs, 9) )
  chosen_enorms = np.zeros( (num_epochs, 9) )
  test_module_enorms = np.zeros( (num_epochs, 9) )
  test_chosen_enorms = np.zeros( (num_epochs, 9) )

  # Create Dataset training data
  X = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k1):
      X = np.vstack([X, r*np.eye(2**n1)])

  # Create Dataset labels
  Y = np.flip(gen_binary_patterns(n1).T, axis=1)
  for i in range(k2):
      Y = np.vstack([Y, r*np.eye(2**n1)])
  Y_keep_feat = np.arange(Y.shape[0])
  Y_delete = np.random.choice(n1, n1-n2, replace=False)
  Y_keep_feat = np.delete(Y_keep_feat, Y_delete)
  Y = Y[Y_keep_feat]
  #print("Input Data: \n", X)
  #print("Initial Labels: \n", Y)

  data_idxs = np.arange(0, X.shape[1])
  np.random.shuffle(data_idxs)
  X_train = X[:,data_idxs[:num_training]]
  Y_train = Y[:,data_idxs[:num_training]]
  X_val = X[:,data_idxs[num_training:num_training+num_validation]]
  Y_val = Y[:,data_idxs[num_training:num_training+num_validation]]
  X_seen = X[:,data_idxs[:num_training+num_validation]]
  Y_seen = Y[:,data_idxs[:num_training+num_validation]]
  X_test = X[:,data_idxs[num_training+num_validation:]]
  Y_test = Y[:,data_idxs[num_training+num_validation:]]

  # Create Modules
  XG = [(0,n1),(n1,n1+k1*(2**n1)),(0,n1+k1*(2**n1))]
  YG = [(0,n2),(n2,n2+k2*(2**n1)),(0,n2+k2*(2**n1))]
  modules_ranges = [ [xg, yg] for xg in XG for yg in YG]
  modules_params = [init_random_params(param_scale, [xg[1]-xg[0], num_hidden, yg[1]-yg[0]], seed) for xg in XG for yg in YG]
  data_norms = jnp.array([jnp.linalg.norm( jnp.dot( (1/X.shape[0])*jnp.dot(Y[yg[0]:yg[1]], X[xg[0]:xg[1]].T),\
               np.linalg.pinv((1/X.shape[0])*jnp.dot(X[xg[0]:xg[1]], X[xg[0]:xg[1]].T)) ), 'fro') for xg in XG for yg in YG])

  batch_size = X.shape[1]

  gate_params = init_random_params_bias(param_scale, gate_layer_sizes, seed)
  er_params = init_random_params(param_scale, er_layer_sizes, seed)
  start_gate_probs = gates_predict(gate_params, X)
  plt.imshow(start_gate_probs, cmap='BuPu', vmin=0, vmax=1)
  plt.colorbar()
  plt.savefig('start_probs.png',dpi=400)
  plt.close()

  for epoch in range(warm_up_epochs):
      key, module_key, er_key, gating_key, stats_key = random.split(key, num=5)
      er_params = er_step(er_params, gate_params, modules_params, modules_ranges, (X_seen,Y_seen), rand_prob, er_key)
      print('Epoch: ', epoch, ' ', str(er_loss(er_params, modules_params, modules_ranges, gate_params, (X_seen,Y_seen), rand_prob, stats_key)))
  
  i = 0
  for epoch in range(num_epochs):
      #os.system('clear')
      print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$        "+str(i)+"       $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
      if np.random.rand() < 0.95:
          key, module_key, er_key, gating_key, stats_key = random.split(key, num=5)
          modules_params = weight_step(modules_params, modules_ranges, gate_params, (X_train,Y_train), rand_prob, module_key)
          er_params = er_step(er_params, gate_params, modules_params, modules_ranges, (X_seen,Y_seen), rand_prob, er_key) #(X_val,Y_val)
          gate_params = gate_step(gate_params, er_params, modules_params, modules_ranges, (X_val,Y_val), rand_prob, gating_key)
          i = ((i + 1) % 20)
      else:
          key, module_key, er_key, gating_key, stats_key = random.split(key, num=5)
          modules_params = weight_step(modules_params, modules_ranges, gate_params, (X_train,Y_train), rand_prob, module_key)
          er_params = er_step(er_params, gate_params, modules_params, modules_ranges, (X_seen,Y_seen), rand_prob, er_key) #(X_val,Y_val)
          gate_params = gate_step(gate_params, er_params, modules_params, modules_ranges, (X_seen,Y_seen), rand_prob, gating_key)
          i = ((i + 1) % 20)
      
      print("############################################ Training Metrics #################################################")
      losses[epoch], sys_losses[epoch], non_losses[epoch], er_losses[epoch], expect_reward, exp_losses,\
              used_gates, module_enorms[epoch], chosen_enorms[epoch] =\
              statistics(modules_params, modules_ranges, er_params, gate_params, (X_seen,Y_seen), rand_prob, stats_key) 
      print('Epoch: ',epoch, 'i:',i, ', Loss: ',losses[epoch], ', Sys Loss: ',sys_losses[epoch], ', Non Loss: ',non_losses[epoch],\
            ', ER_Loss: ',er_losses[epoch],', Rand Prob: ',rand_prob, '\nExpected Losses\n', exp_losses,\
            ',\nModules Chosen:\n',chosen_enorms[epoch], ',\nModules Used:\n',module_enorms[epoch])

      print("############################################ Test Metrics #################################################")
      test_losses[epoch], sys_test_losses[epoch], non_test_losses[epoch], test_er_losses[epoch], expect_reward, exp_losses,\
              used_gates, test_module_enorms[epoch],test_chosen_enorms[epoch]=\
              statistics(modules_params, modules_ranges, er_params, gate_params, (X_test,Y_test), rand_prob, stats_key) 
      print('Epoch: ',epoch, 'i:',i, ', Loss: ',test_losses[epoch], ', Sys Loss: ',sys_test_losses[epoch],\
            ', Non Loss: ',non_test_losses[epoch], ', ER_Loss: ',test_er_losses[epoch],', Rand Prob: ',rand_prob,\
            '\nExpected Losses\n',exp_losses,',\nModules Chosen:\n',test_chosen_enorms[epoch],',\nModules Used:\n',test_module_enorms[epoch])
      
      os.system('clear') 
      if rand_prob > 0.0 and ( (epoch+1) % rand_drop_epoch) == 0:#1500
          rand_prob = rand_prob - 0.1
          rand_prob = np.round(rand_prob, 5)

  end_gate_probs = gates_predict(gate_params, X)
  plt.imshow(end_gate_probs, cmap='BuPu', vmin=0, vmax=1)
  plt.colorbar()
  plt.savefig('end_gates/'+str(run_idx)+'_uniform.pdf',dpi=400)
  plt.close()
 
  # Plot losses
  plt.plot(losses, color='red', label='Train Losses')
  plt.plot(test_losses, color='green', label='Test Losses')
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper right')
  plt.grid()
  plt.savefig('losses/'+str(run_idx)+'_uniform.pdf')
  plt.close()

  # Plot losses
  plt.plot(sys_losses, color='red', label='Compositional Train Losses')
  plt.plot(sys_test_losses, color='green', label='Compositional Test Losses')
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper right')
  plt.grid()
  plt.savefig('sys_losses/'+str(run_idx)+'_uniform.pdf')
  plt.close()

  # Plot losses
  plt.plot(non_losses, color='red', label='Non-compositional Train Losses')
  plt.plot(non_test_losses, color='green', label='Non-compositional Test Losses')
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Quadratic Loss")
  plt.xlabel("Epoch number")
  plt.legend(loc='upper right')
  plt.grid()
  plt.savefig('non_losses/'+str(run_idx)+'_uniform.pdf')
  plt.close()

  norm_names = [r'$\Omega_x \Omega_y$', r'$\Omega_x \Gamma_y$', r'$\Omega_x \textit{Full}_y$',
                r'$\Gamma_x \Omega_y$', r'$\Gamma_x \Gamma_y$', r'$\Gamma_x \textit{Full}_y$',
                r'$\textit{Full}_x \Omega_y$', r'$\textit{Full}_x \Gamma_y$', r'$\textit{Full}_x \textit{Full}_y$']
  # Plot Gated Norms
  for m in range(module_enorms.shape[1]):
      plt.plot(module_enorms[:,m], label=norm_names[m])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.grid()
  plt.legend(loc='upper left')
  plt.savefig('train_norms/'+str(run_idx)+'_uniform.pdf')
  plt.close() 

  for m in range(module_enorms.shape[1]):
      plt.plot(test_module_enorms[:,m], label=norm_names[m])
  plt.axhline(0, color='black')
  plt.axvline(0, color='black')
  plt.ylabel("Frobenius Norm")
  plt.xlabel("Epoch number")
  plt.grid()
  plt.legend(loc='upper left')
  plt.savefig('test_norms/'+str(run_idx)+'_uniform.pdf')
  plt.close() 

  np.savetxt('n_runs/'+str(run_idx)+'_uniform_train_losses.txt', losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_sys_train_losses.txt', sys_losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_non_train_losses.txt', non_losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_train_module_enorms.txt', chosen_enorms)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_test_losses.txt', test_losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_sys_test_losses.txt', sys_test_losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_non_test_losses.txt', non_test_losses)
  np.savetxt('n_runs/'+str(run_idx)+'_uniform_test_module_enorms.txt', test_chosen_enorms)
