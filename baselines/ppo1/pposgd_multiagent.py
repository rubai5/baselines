from baselines.common import Dataset, explained_variance, fmt_row, zipsame
from baselines import logger
import baselines.common.tf_util as U
import tensorflow as tf, numpy as np
import time
from baselines.common.mpi_adam import MpiAdam
from baselines.common.mpi_moments import mpi_moments
from baselines.ppo1 import pposgd_simple, pposgd_simple_attacker
from mpi4py import MPI
from collections import deque
from copy import deepcopy

def traj_segment_generator_att(pi_att, pi_def, env_att, horizon, stochastic):
    t = 0
    ac = env_att.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    ob = env_att.reset()
    label = env_att.env.labels()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    labels = np.array([label for _ in range(horizon)])
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()

    while True:
        prevac = ac
        ac, vpred = pi_att.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"label" : labels, "ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "def_ob" : def_obs, 
                    "def_ac" : def_ac}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        labels[i] = label
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # Editing update step to make work with defender
        #print("intermediate results")
        #print("state", env_att.env.state)
        def_obs = env_att.env.get_def_obs(ac)
        def_ac, _ = pi_def.act(stochastic=True, ob=def_obs)
        ob, rew, new, _ = env_att.env.def_state_update(def_obs, def_ac)
        label = env_att.env.labels()
        rews[i] = rew
        #print("attacker action", ac)
        #print("def obs", def_obs)
        #print("defender action", def_ac)
        #print("new state", env_att.env.state)        

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            #print("new round")
            ob = env_att.reset()
            #print("state", ob)
            label = env_att.env.labels()
        t += 1


def traj_segment_generator_def(pi_att, pi_def, env_att, env_def, horizon, stochastic):
    t = 0
    ac = env_def.action_space.sample() # not used, just so we have the datatype
    new = True # marks if we're on first timestep of an episode
    
    # get start state from env_att and add to def env
    att_ob = env_att.reset()
    _ = env_def.reset()
    env_def.env.game_state = deepcopy(att_ob)

    # get defender observation from attacker
    att_ac, _ = pi_att.act(stochastic=True, ob=att_ob)
    ob = env_att.env.get_def_obs(att_ac)
    env_def.env.state = deepcopy(ob) # sync def state
    label = env_def.env.labels()

    cur_ep_ret = 0 # return in current episode
    cur_ep_len = 0 # len of current episode
    ep_rets = [] # returns of completed episodes in this segment
    ep_lens = [] # lengths of ...

    # Initialize history arrays
    labels = np.array([label for _ in range(horizon)])
    obs = np.array([ob for _ in range(horizon)])
    rews = np.zeros(horizon, 'float32')
    vpreds = np.zeros(horizon, 'float32')
    news = np.zeros(horizon, 'int32')
    acs = np.array([ac for _ in range(horizon)])
    prevacs = acs.copy()
    # adding in quantities related to attacker
    att_obs = np.array([att_ob for _ in range(horizon)])
    att_acs = np.array([att_ac for _ in range(horizon)])
    

    while True:
        prevac = ac
        ac, vpred = pi_def.act(stochastic, ob)
        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % horizon == 0:
            yield {"label" : labels, "ob" : obs, "rew" : rews, "vpred" : vpreds, "new" : news,
                    "ac" : acs, "prevac" : prevacs, "nextvpred": vpred * (1 - new),
                    "ep_rets" : ep_rets, "ep_lens" : ep_lens, "att_ob" : att_ob, "att_ac" : att_ac}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets = []
            ep_lens = []
        i = t % horizon
        labels[i] = label
        obs[i] = ob
        vpreds[i] = vpred
        news[i] = new
        acs[i] = ac
        prevacs[i] = prevac

        # Editing update step to make work with attacker
        assert np.all(env_att.env.state == env_def.env.game_state), "Environments not synced!"
        assert np.all(env_def.env.state == ob), "Observations not synced"
        
        #print("state", env_def.env.game_state)
        #print("attacker action", att_ac)
        #print("defender observation", ob)
        #print("defender action", ac) 
        # take a step and sync
        _, rew, new, _ = env_def.step(ac)
        #print("reward, done", rew, new)
        _ = env_att.env.def_state_update(ob, ac)
        
        assert np.all(env_def.env.game_state == env_att.env.state), "state sync failed!"
        # if not done (new), attacker gets next observation
        if not new:
            att_ac, _ = pi_att.act(True, env_att.env.state)
            ob = env_att.env.get_def_obs(att_ac)
            env_def.env.state = deepcopy(ob)

        label = env_def.env.labels()
        rews[i] = rew

        cur_ep_ret += rew
        cur_ep_len += 1
        if new:
            ep_rets.append(cur_ep_ret)
            ep_lens.append(cur_ep_len)
            cur_ep_ret = 0
            cur_ep_len = 0
            
            # get start state from env_att and add to def env
            att_ob = env_att.reset()
            _ = env_def.reset()
            env_def.env.game_state = deepcopy(att_ob)

            # get defender observation from attacker
            att_ac, _ = pi_att.act(stochastic=True, ob=att_ob)
            ob = env_att.env.get_def_obs(att_ac)
            env_def.env.state = deepcopy(ob) # sync def state
            label = env_def.env.labels()
        t += 1



def add_vtarg_and_adv(seg, gamma, lam):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def create_graph(env, pi_name, policy_func, *,
                   clip_param, entcoeff,
                   adam_epsilon=1e-5):
    # Setup losses and stuff
    # ----------------------------------------
    with tf.name_scope(pi_name):
        ob_space = env.observation_space
        ac_space = env.action_space
        pi = policy_func("pi", pi_name, ob_space, ac_space) # Construct network for new policy
        oldpi = policy_func("oldpi", pi_name, ob_space, ac_space) # Network for old policy
        atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
        ret = tf.placeholder(dtype=tf.float32, shape=[None]) # Empirical return

        lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
        clip_param = clip_param * lrmult # Annealed cliping parameter epislon

        ob = U.get_placeholder_cached(name="ob"+pi_name)
        ac = pi.pdtype.sample_placeholder([None])

        kloldnew = oldpi.pd.kl(pi.pd)
        ent = pi.pd.entropy()
        meankl = U.mean(kloldnew)
        meanent = U.mean(ent)
        pol_entpen = (-entcoeff) * meanent

        ratio = tf.exp(pi.pd.logp(ac) - oldpi.pd.logp(ac)) # pnew / pold
        surr1 = ratio * atarg # surrogate from conservative policy iteration
        surr2 = U.clip(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg #
        pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)
        vf_loss = U.mean(tf.square(pi.vpred - ret))
        total_loss = pol_surr + pol_entpen + vf_loss
        losses = [pol_surr, pol_entpen, vf_loss, meankl, meanent]
        loss_names = ["pol_surr", "pol_entpen", "vf_loss", "kl", "ent"]

        var_list = pi.get_trainable_variables()
        lossandgrad = U.function([ob, ac, atarg, ret, lrmult], losses + [U.flatgrad(total_loss, var_list)])

        adam = MpiAdam(var_list, epsilon=adam_epsilon)

        assign_old_eq_new = U.function([],[], updates=[tf.assign(oldv, newv)
            for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
        compute_losses = U.function([ob, ac, atarg, ret, lrmult], losses)
    
    return pi, oldpi, loss_names, lossandgrad, adam, assign_old_eq_new, compute_losses


def initialize_graph(adam, name):
    with tf.name_scope(name):
        U.initialize()
        adam.sync()

def core_train_att(env, pi, oldpi, pi_def, loss_names,
              lossandgrad, adam, assign_old_eq_new,
               compute_losses,
              timesteps_per_batch, optim_epochs,
              optim_stepsize, optim_batchsize,
              gamma, lam,
              max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
              callback=None,
              schedule='constant',
              test_envs=[]):
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_att(pi, pi_def, env, timesteps_per_batch, stochastic=True)
    if test_envs:
       test_gens = [pposgd_simple_attacker.traj_segment_generator(pi, attenv, timesteps_per_batch,
                    stochastic=True) for attenv in test_envs]


    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=50) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=50) # rolling buffer for episode rewards
    testbuffers = [deque(maxlen=50) for attenv in test_envs]
    
    # Maithra edits: add lists to return logs
    ep_lengths = []
    ep_rewards = []
    ep_labels = []
    ep_actions = []
    ep_correct_actions = []
    ep_obs = []
    ep_def_obs = []
    ep_def_actions = []
    ep_test_rewards = [[] for attenv in test_envs]

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Attacker Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)
        if test_envs:
           test_segs = [test_gen.__next__() for test_gen in test_gens]

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, label, def_ob, def_ac  = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["label"], \
                                          seg["def_ob"], seg["def_ac"]
        
        if test_envs:
            for i, test_seg in enumerate(test_segs):
                test_rews = test_seg["ep_rets"]
                testbuffers[i].extend(test_rews)
                ep_test_rewards[i].append(np.mean(testbuffers[i]))        

        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))

        # Maithra edit: append intermediate results onto returned logs
        ep_lengths.append(np.mean(lenbuffer))
        ep_rewards.append(np.mean(rewbuffer))
        ep_labels.append(deepcopy(label))
        ep_actions.append(deepcopy(ac))
        ep_obs.append(deepcopy(ob))
        ep_def_obs.append(deepcopy(def_ob))
        ep_def_actions.append(deepcopy(def_ac))

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()

    #Maithra edit
    info_dict = {"lengths": ep_lengths, "rewards" : ep_rewards, "labels" : ep_labels,
                 "actions" : ep_actions, "correct_actions" : ep_correct_actions, "obs": ep_obs,
                 "def_obs" : ep_def_obs, "def_actions" : ep_def_actions, "test_rews" : ep_test_rewards} 
    return pi, oldpi, lossandgrad, adam, assign_old_eq_new, compute_losses, info_dict

def core_train_def(env, pi, oldpi, env_att, pi_att, loss_names,
              lossandgrad, adam, assign_old_eq_new,
               compute_losses,
              timesteps_per_batch, optim_epochs,
              optim_stepsize, optim_batchsize,
              gamma, lam,
              max_timesteps=0, max_episodes=0, max_iters=0, max_seconds=0,  # time constraint
              callback=None,
              schedule='constant', test_envs=[]):
    # Prepare for rollouts
    # ----------------------------------------
    seg_gen = traj_segment_generator_def(pi_att, pi, env_att, env, timesteps_per_batch, stochastic=True)

    if test_envs:
       test_gens = [pposgd_simple.traj_segment_generator(pi, attenv, timesteps_per_batch,
                    stochastic=True) for attenv in test_envs]
    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=50) # rolling buffer for episode lengths
    rewbuffer = deque(maxlen=50) # rolling buffer for episode rewards
    testbuffers = [deque(maxlen=50) for test_env in test_envs]

    # Maithra edits: add lists to return logs
    ep_lengths = []
    ep_rewards = []
    ep_labels = []
    ep_actions = []
    ep_correct_actions = []
    ep_obs = []
    ep_att_obs = []
    ep_att_actions = []
    ep_test_rewards = [[] for test_env in test_envs]

    assert sum([max_iters>0, max_timesteps>0, max_episodes>0, max_seconds>0])==1, "Only one time constraint permitted"

    while True:
        if callback: callback(locals(), globals())
        if max_timesteps and timesteps_so_far >= max_timesteps:
            break
        elif max_episodes and episodes_so_far >= max_episodes:
            break
        elif max_iters and iters_so_far >= max_iters:
            break
        elif max_seconds and time.time() - tstart >= max_seconds:
            break

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult =  max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("********** Defender Iteration %i ************"%iters_so_far)

        seg = seg_gen.__next__()
        add_vtarg_and_adv(seg, gamma, lam)

        if test_envs:
           test_segs = [test_gen.__next__() for test_gen in test_gens]

        # ob, ac, atarg, ret, td1ret = map(np.concatenate, (obs, acs, atargs, rets, td1rets))
        ob, ac, atarg, tdlamret, label, att_ob, att_ac  = seg["ob"], seg["ac"], seg["adv"], seg["tdlamret"], seg["label"], \
                                          seg["att_ob"], seg["att_ac"]
        
        if test_envs:
            for i, test_seg in enumerate(test_segs):
                test_rews = test_seg["ep_rets"]
                testbuffers[i].extend(test_rews)
                ep_test_rewards[i].append(np.mean(testbuffers[i])) 
       
        vpredbefore = seg["vpred"] # predicted value function before udpate
        atarg = (atarg - atarg.mean()) / atarg.std() # standardized advantage function estimate
        d = Dataset(dict(ob=ob, ac=ac, atarg=atarg, vtarg=tdlamret), shuffle=not pi.recurrent)
        optim_batchsize = optim_batchsize or ob.shape[0]

        if hasattr(pi, "ob_rms"): pi.ob_rms.update(ob) # update running mean/std for policy

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))
        # Here we do a bunch of optimization epochs over the data
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):
                *newlosses, g = lossandgrad(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
                adam.update(g, optim_stepsize * cur_lrmult) 
                losses.append(newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            newlosses = compute_losses(batch["ob"], batch["ac"], batch["atarg"], batch["vtarg"], cur_lrmult)
            losses.append(newlosses)            
        meanlosses,_,_ = mpi_moments(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)
        logger.record_tabular("ev_tdlam_before", explained_variance(vpredbefore, tdlamret))
        lrlocal = (seg["ep_lens"], seg["ep_rets"]) # local values
        listoflrpairs = MPI.COMM_WORLD.allgather(lrlocal) # list of tuples
        lens, rews = map(flatten_lists, zip(*listoflrpairs))
        lenbuffer.extend(lens)
        rewbuffer.extend(rews)
        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpRewMean", np.mean(rewbuffer))
        logger.record_tabular("EpThisIter", len(lens))

        # Maithra edit: append intermediate results onto returned logs
        ep_lengths.append(np.mean(lenbuffer))
        ep_rewards.append(np.mean(rewbuffer))
        ep_labels.append(deepcopy(label))
        ep_actions.append(deepcopy(ac))
        ep_obs.append(deepcopy(ob))
        ep_att_obs.append(deepcopy(att_ob))
        ep_att_actions.append(deepcopy(att_ac))
        # compute mean of correct actions and append, ignoring actions
        # where either choice could be right
        count = 0
        idxs = np.all((label == [1,1]), axis=1)
        # removing for now: count += np.sum(idxs)
        new_label = label[np.invert(idxs)]
        new_ac = ac[np.invert(idxs)]
        count += np.sum((new_ac == np.argmax(new_label, axis=1)))
        # changing ep_correct_actions.append(count/len(label))
        ep_correct_actions.append(count/(len(label) - np.sum(idxs)))

        episodes_so_far += len(lens)
        timesteps_so_far += sum(lens)
        iters_so_far += 1
        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        if MPI.COMM_WORLD.Get_rank()==0:
            logger.dump_tabular()
    
    info_dict = {"lengths": ep_lengths, "rewards" : ep_rewards, "labels" : ep_labels,
                 "actions" : ep_actions, "correct_actions" : ep_correct_actions, "obs": ep_obs,
                  "att_actions" : ep_att_actions, "att_obs" : ep_att_obs, "test_rews" : ep_test_rewards}   
    #Maithra edit
    return pi, oldpi, lossandgrad, adam, assign_old_eq_new, compute_losses, info_dict



def flatten_lists(listoflists):
    return [el for list_ in listoflists for el in list_]
