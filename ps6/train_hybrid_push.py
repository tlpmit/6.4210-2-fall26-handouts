"""Policy gradients with a push face, continuous distance, and a curriculum.

Rollouts use gamma=1 and bootstrap at computational cutoffs. PPO clips the ratio of
the joint face/distance probability; a separate critic supplies the baseline.
The Gaussian raw distance is transformed by a sigmoid to a bounded command.
Its Jacobian cancels in the old/new probability ratio because the transform is
fixed. The default trains from scratch; optional checkpoint initialization and
a separate, decaying demonstration loss support behavior-cloned policies.
Demonstrations never enter the PPO importance-ratio objective.
"""
from dataclasses import asdict
from copy import deepcopy
import argparse
import json
from pathlib import Path
import hashlib
import time
import numpy as np
import torch
from torch import nn
from push_env import LearningPush, FACES, summary, OBSERVATION_DIM, OBSERVATION_VERSION
from push_reward import reward_from_legacy
from push_math import joint_log_prob, ppo_actor_loss, critic_loss
from reinforce import reward_to_go

ROOT = Path(__file__).parent/'results/hybrid_push_pilot'
from push_policy import (HybridActorCritic, ResidualHybridActorCritic,
                         distance_from_raw, DISTANCE_MIN, DISTANCE_MAX)


def hybrid_step(env, face, raw, *, free_time_cost=0., total_time_cost=0., push_penalty=.02):
    axis,sign = FACES[int(face)]
    distance = float(distance_from_raw(torch.tensor(raw)))
    free_before=env.task.motion_seconds['free']
    _,reward,term,trunc,info = env.task.step(env.task.face_push(axis,sign,distance))
    free_seconds=env.task.motion_seconds['free']-free_before
    info.update(unpenalized_reward=reward,free_motion_seconds=free_seconds)
    reward=reward_from_legacy(reward,int(info['valid']),free_seconds,info['simulated_seconds'],
        free_time_cost=free_time_cost,total_time_cost=total_time_cost,push_penalty=push_penalty)
    info.update(is_success=info['success'],distance=float(np.linalg.norm(
        env.task.box_pose()[:2]-env.task.goal)),macros=env.task.count,command_distance=distance,
        terminated=bool(term),truncated=bool(trunc))
    return env.observation(),reward,term or trunc,info


def rollout_returns(model, rewards, final_observation, *, terminated):
    """Fixed return targets: observed reward-to-go plus V at a cutoff, gamma=1.

    Success and physical failure have zero continuation. The supplied caller
    evaluates and detaches the critic before any optimization of this batch.
    """
    continuation = 0.
    if not terminated:
        with torch.no_grad():
            continuation = float(model.critic(torch.as_tensor(
                final_observation, dtype=torch.float32)[None]).squeeze())
    return reward_to_go(rewards) + continuation


def curriculum_scale(steps, budget, enabled=True):
    """Scale distant goals from 20% to 100% over the first 75% of training."""
    return .2+.8*min(1.,steps/(.75*budget)) if enabled else 1.


def training_reset(env, rng, steps, budget, curriculum=True, curriculum_duration=None):
    scale = (curriculum_scale(steps,budget,curriculum) if curriculum_duration is None else
             .2+.8*min(1.,steps/curriculum_duration) if curriculum else 1.)
    goal = rng.uniform(.22,.35,2)*rng.choice([-1,1],2)*scale
    obs,_ = env.reset(options=dict(goal=goal,box_pose=(0.,0.,rng.uniform(-.5,.5))))
    return obs,scale


def ppo_loss(model, obs, faces, raw, old_log_probs, returns, advantages, *, clip=.2, entropy_coef=.01):
    """Combine student losses, keeping recorded targets fixed during updates."""
    # Supplied gradient bookkeeping: only current predictions are optimized.
    old_log_probs = old_log_probs.detach()
    returns = returns.detach()
    advantages = advantages.detach()
    log_probs,values,entropy = model.evaluate(obs,faces,raw)
    policy_loss = ppo_actor_loss(log_probs, old_log_probs, advantages, clip)
    value_loss = critic_loss(values, returns)
    return policy_loss + .25 * value_loss - entropy_coef * entropy.mean()


def reinforce_loss(model, obs, faces, raw, advantages):
    """One on-policy score-function update; the baseline is held fixed.

    Keep the same small face-entropy bonus as the PPO comparison. The fixed
    distance transform has no parameter-dependent Jacobian, so raw Gaussian
    log probabilities give the same policy gradient as physical distances.
    """
    log_probs,_,entropy = model.evaluate(obs,faces,raw)
    return -(log_probs*advantages.detach()).mean()-.01*entropy.mean()


def evaluate(model, seeds, *, sampled=False, record=False, horizon=None,
             box_com_x=None,mass_model=None,timestep=None,physics=None):
    if getattr(model, 'observation_version', None) != OBSERVATION_VERSION:
        raise ValueError('This checkpoint predates resting-state inputs. Use data/zero_minimum_push/bc_seed130.pt or retrain; archived scores remain available.')
    trained_horizon=getattr(model,'training_horizon',8)
    horizon=trained_horizon if horizon is None else horizon
    box_com_x=getattr(model,'box_com_x',0.) if box_com_x is None else box_com_x
    env = LearningPush(record=record,horizon=horizon,
                       box_com_x=box_com_x,orientation_features=getattr(model,'orientation_features','none'),
                       mass_model=getattr(model,'mass_model','halves') if mass_model is None else mass_model,
                       timestep=getattr(model,'timestep',.005) if timestep is None else timestep,
                       physics=getattr(model,'physics',None) if physics is None else physics); rows=[]
    for seed in seeds:
        obs,_ = env.reset(seed=int(seed)); total=0.;original_total=0.;invalid=0
        rng = np.random.default_rng(seed+2_000_000)
        while not env.task.done:
            face,raw,_,_ = model.act(obs,greedy=not sampled,rng=rng)
            obs,reward,done,info = hybrid_step(env,face,raw,free_time_cost=getattr(model,'free_time_cost',0.),
                total_time_cost=getattr(model,'total_time_cost',0.),push_penalty=getattr(model,'push_penalty',.02))
            total += reward; invalid += int(not info['valid'])
            original_total+=info['unpenalized_reward']
        rows.append(dict(seed=int(seed),success=info['success'],distance=info['distance'],
                         macros=info['macros'],return_=total,invalid=invalid,
                         unpenalized_return=original_total,execution_seconds=float(env.task.context.get_time()),
                         free_motion_seconds=env.task.motion_seconds['free'],
                         settling_seconds=env.task.motion_seconds['settle'],
                         cutoff=info['truncated'],observation_version=OBSERVATION_VERSION,
                         finger_travel_m=sum(env.task.motion_length.values())))
    return rows,env


def load(path):
    saved = torch.load(path,map_location='cpu',weights_only=True)
    if saved['plan'].get('observation_version') != OBSERVATION_VERSION:
        raise ValueError('Archived checkpoint has velocity/countdown inputs. Use the current supplied BC checkpoint or a new training run.')
    if saved['plan'].get('distance_bounds') != [DISTANCE_MIN,DISTANCE_MAX]:
        raise ValueError('Checkpoint uses an older push-distance mapping. Use data/zero_minimum_push/bc_seed130.pt or retrain; old weights cannot be silently reinterpreted.')
    model = HybridActorCritic(saved['obs_dim'])
    if saved['plan'].get('residual',False):model=ResidualHybridActorCritic(model)
    model.load_state_dict(saved['model'])
    model.observation_version=OBSERVATION_VERSION
    model.training_horizon=saved['plan'].get('horizon',8)
    model.free_time_cost=saved['plan'].get('free_time_cost',0.)
    model.total_time_cost=saved['plan'].get('total_time_cost',0.)
    model.push_penalty=saved['plan'].get('push_penalty',.02)
    model.box_com_x=saved['plan'].get('box_com_x',0.)
    model.orientation_features=saved['plan'].get('orientation_features','none')
    model.mass_model=saved['plan'].get('mass_model','halves')
    model.timestep=saved['plan'].get('timestep',.005)
    from push_physics import PushPhysics
    specification=saved['plan'].get('physics')
    model.physics=None if specification is None else PushPhysics(**specification)
    return model


def adapt_bc_checkpoint(source, destination):
    """Restrict the old BC actor to zero velocities and its initial countdown.

    Retained weights are copied, and the old countdown=1 contribution becomes
    a constant first-layer bias. No discarded input is read at deployment.
    The critic is reset: old finite-deadline returns are not new value targets.
    This archival adapter preserves the old distance mapping; load() rejects
    it for the current zero-minimum action space. Refit BC for current runs.
    """
    saved=torch.load(source,map_location='cpu',weights_only=True)
    if saved['obs_dim']!=22 or saved['plan'].get('residual',False):
        raise ValueError('Expected the original non-residual 22-input BC checkpoint')
    model=HybridActorCritic(OBSERVATION_DIM)
    retained=np.r_[0:4,5:14]
    state=model.state_dict()
    for key,value in saved['model'].items():
        if key.startswith('critic.'):
            state[key]=torch.zeros_like(state[key])
        elif key=='actor.0.weight':state[key]=value[:,retained].clone()
        elif key=='actor.0.bias':state[key]=value+saved['model']['actor.0.weight'][:,4]
        else:state[key]=value.clone()
    plan=dict(saved['plan'],observation_version=OBSERVATION_VERSION,observation_dim=OBSERVATION_DIM,
              distance_bounds=saved['plan'].get('distance_bounds',[.025,.125]),
              adaptation='Original BC actor restricted to zero velocities and countdown=1; critic reset',
              source_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              total_time_cost=.02,free_time_cost=0.,push_penalty=0.,
              cutoff_semantics='Computational rollout cutoff, not a task deadline')
    torch.save(dict(model=state,obs_dim=OBSERVATION_DIM,plan=plan,steps=0),destination)
    return plan


def add_orientation_inputs(model):
    """Preserve the complete BC actor/critic function when adding two features.

    Zero initial incoming weights make the warm-start outputs exactly the same;
    PPO can subsequently learn nonzero weights on the heavy-axis orientation.
    """
    old_dim=model.actor[0].in_features
    if old_dim==15:return model
    if old_dim!=13:raise ValueError('Orientation expansion expects 13 or 15 input features')
    expanded=HybridActorCritic(15);state=model.state_dict()
    for key in ['actor.0.weight','critic.0.weight']:
        state[key]=torch.cat([state[key],torch.zeros(state[key].shape[0],2)],dim=1)
    expanded.load_state_dict(state)
    for key in ['training_horizon','free_time_cost','total_time_cost','push_penalty','box_com_x','orientation_features','mass_model','timestep','physics']:
        if hasattr(model,key):setattr(expanded,key,getattr(model,key))
    expanded.orientation_features='heavy_axis'
    expanded.observation_version=OBSERVATION_VERSION
    return expanded


def execution_summary(rows):
    return {f'mean_{key}':float(np.mean([r[key] for r in rows]))
            for key in ['unpenalized_return','execution_seconds','free_motion_seconds','finger_travel_m']}


def train(seed=5, budget=2048, curriculum=True, out=ROOT, monitoring_start=11000,test_start=12000,
          algorithm='ppo', horizon=8, *, initial_checkpoint=None, learning_rate=.003,
          epochs=10, entropy_coef=.01, clip=.2, target_kl=None, bc_files=None, bc_weight=0.,
          curriculum_duration=None, test_count=40, monitoring_count=8, save_checkpoints=False,
          free_time_cost=0., initial_face_temperature=1., initial_raw_std=None, critic_warmup_batches=0,
          evaluate_best=False, box_com_x=0., orientation_features='none',mass_model='halves',timestep=.005,physics=None,
          residual=False, training_seed_start=None, fresh_critic=False, total_time_cost=0., push_penalty=.02):
    reward_from_legacy(0.,0,0.,0.,free_time_cost=free_time_cost,total_time_cost=total_time_cost,push_penalty=push_penalty)
    if algorithm not in ('ppo','reinforce'):
        raise ValueError('algorithm must be ppo or reinforce')
    if free_time_cost<0 or initial_face_temperature<=0 or (initial_raw_std is not None and initial_raw_std<=0):
        raise ValueError('Costs must be nonnegative and exploration scales positive')
    if critic_warmup_batches<0:raise ValueError('Critic warmup batches must be nonnegative')
    if evaluate_best and not save_checkpoints:raise ValueError('Best-policy evaluation requires saved checkpoints')
    if residual and (initial_checkpoint is None or bc_files):raise ValueError('Residual needs a BC checkpoint and no continuing imitation loss')
    if training_seed_start is not None and curriculum:raise ValueError('Seeded goal sequences require curriculum=False')
    torch.set_num_threads(1);torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    env = LearningPush(horizon=horizon,box_com_x=box_com_x,orientation_features=orientation_features,mass_model=mass_model,timestep=timestep,physics=physics); obs_dim = env.observation_space.shape[0]
    model = load(initial_checkpoint) if initial_checkpoint is not None else HybridActorCritic(obs_dim)
    if orientation_features=='heavy_axis':model=add_orientation_inputs(model)
    if model.actor[0].in_features!=obs_dim:raise ValueError('Checkpoint and requested observation dimensions differ')
    if orientation_features!='none' and bc_files:raise ValueError('Legacy demonstration files lack heavy-axis features')
    model.training_horizon=horizon
    model.observation_version=OBSERVATION_VERSION
    model.free_time_cost=free_time_cost
    model.total_time_cost=total_time_cost;model.push_penalty=push_penalty
    model.box_com_x=float(box_com_x)
    model.orientation_features=orientation_features
    model.mass_model=mass_model
    model.physics=physics;model.timestep=timestep
    with torch.no_grad():
        model.face.weight.div_(initial_face_temperature);model.face.bias.div_(initial_face_temperature)
        if initial_raw_std is not None:model.log_std.fill_(np.log(initial_raw_std))
    if residual:model=ResidualHybridActorCritic(model)
    if fresh_critic:
        # Same independently initialized critic for all PPO variants of a seed.
        with torch.random.fork_rng():
            torch.manual_seed(seed+100_000)
            model.critic=HybridActorCritic(obs_dim).critic
    if training_seed_start is not None:torch.manual_seed(seed+200_000)
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
    demonstrations=None
    if bc_files:
        from clone_multi_push import dataset,imitation_loss
        demos=dataset(bc_files)
        demonstrations=[torch.from_numpy(demos[k]) for k in ['observations','faces','distances']]
    if bc_weight and demonstrations is None:raise ValueError('BC regularization needs demonstration files')
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    base = out/f'hybrid_seed{seed}'
    plan = dict(seed=seed,algorithm=algorithm,residual=residual,training_seed_start=training_seed_start,fresh_critic=fresh_critic,
        advantage='Observed reward-to-go plus fixed critic continuation at rollout cutoffs, minus learned state baseline',
        observation_version=OBSERVATION_VERSION,observation_dim=obs_dim,
        cutoff_semantics='Computational limit; bootstrap training targets, report only observed evaluation rewards',
        settling=dict(linear_speed_m_s=env.task.settle_linear_speed,
                      angular_speed_rad_s=env.task.settle_angular_speed,
                      hold_seconds=env.task.settle_hold_seconds),
        budget=budget,curriculum=curriculum,horizon=horizon,
        box_com_x=float(box_com_x),mass_properties=env.task.mass_properties,
        orientation_features=orientation_features,
        mass_model=mass_model,timestep=timestep,
        physics=None if physics is None else asdict(physics),
        reward_definition='(distance_before-distance_after)/.15 + success - push_penalty - time cost; invalid -.2; workspace exit additional -1',
        legacy_return_definition='Progress + success - .02 per valid push; invalid -.2; workspace exit additional -1',
        free_time_cost=free_time_cost,total_time_cost=total_time_cost,
        additional_reward='Subtract total_time_cost times all simulated macro-action seconds OR free_time_cost times free-space seconds; all settling included',
        initial_face_temperature=initial_face_temperature,critic_warmup_batches=critic_warmup_batches,
        push_penalty=push_penalty,monitoring_seeds=list(range(monitoring_start,monitoring_start+monitoring_count)),
        test_seeds=list(range(test_start,test_start+test_count)),gamma=1.,batch_episodes=16,epochs=epochs,
        learning_rate=learning_rate,clip=clip if algorithm=='ppo' else None,target_kl=target_kl,
        actor_updates=f'{epochs} shuffled minibatch epochs' if algorithm=='ppo' else 'one full-batch update',
        critic_updates=f'{epochs} shuffled minibatch epochs' if algorithm=='ppo' else '10 full-batch updates after actor update',
        face_entropy_coefficient=entropy_coef,advantage_normalization='batch mean and standard deviation',
        distance_bounds=[DISTANCE_MIN,DISTANCE_MAX],initial_std_per_face=model.log_std.detach().exp().tolist(),
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        bc_files=[str(p) for p in bc_files] if bc_files else [],bc_weight=bc_weight,
        bc_schedule='Weight decays linearly to zero over the requested action budget',
        curriculum_duration=curriculum_duration,save_checkpoints=save_checkpoints,
        evaluate_best=evaluate_best,best_selection='Maximum monitoring success, then mean reward; earliest tie',
        selection='Final iterate, no test-based selection',
        curriculum_rule=(f'Goal scale increases .2 to 1 over {curriculum_duration or .75*budget:g} macro actions'
                         if curriculum else 'Full distant training goals from the start')
                         + '; evaluation always uses full distant goals')
    base.with_suffix('.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    steps=0; rows=[]; start=time.perf_counter(); next_monitor=512
    recent_success=[];episode_count=0;update_count=0;best_monitor=(-1.,-float('inf'));kl_stops=0;training_invalid=0
    def monitor():
        nonlocal best_monitor
        tests,_ = evaluate(model,plan['monitoring_seeds'])
        row = dict(steps=steps,seconds=time.perf_counter()-start,
                   horizon=horizon,training_episodes=episode_count,policy_batches=update_count,
                   goal_scale=(curriculum_scale(steps,budget,curriculum) if curriculum_duration is None else
                               .2+.8*min(1.,steps/curriculum_duration) if curriculum else 1.),
                   kl_early_stops=kl_stops,training_invalid=training_invalid,
                   recent_training_success=float(np.mean(recent_success[-32:])) if recent_success else None,
                   **summary(tests),**execution_summary(tests),episodes=tests)
        rows.append(row)
        base.with_suffix('.progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        saved=dict(model=model.state_dict(),obs_dim=obs_dim,plan=plan,steps=steps)
        torch.save(saved,base.with_suffix('.pt'))
        if save_checkpoints:
            torch.save(saved,out/f'{base.name}_step{steps}.pt')
            score=(row['success'],row['mean_return'])
            if score>best_monitor:
                best_monitor=score;torch.save(saved,out/f'{base.name}_best.pt')
        print('HYBRID',seed,{k:v for k,v in row.items() if k!='episodes'},flush=True)
    monitor()
    while steps < budget:
        states=[];faces=[];raw_actions=[];old_logs=[];old_values=[];returns=[]
        for episode in range(16):
            if training_seed_start is None:
                obs,scale = training_reset(env,rng,steps,budget,curriculum,curriculum_duration)
            else:
                obs,_=env.reset(seed=training_seed_start+episode_count);scale=1.
            rewards=[]
            while not env.task.done:
                face,raw,logp,value = model.act(obs)
                states.append(obs); faces.append(face); raw_actions.append(raw)
                old_logs.append(logp);old_values.append(value)
                obs,reward,done,info = hybrid_step(env,face,raw,free_time_cost=free_time_cost,
                    total_time_cost=total_time_cost,push_penalty=push_penalty)
                training_invalid+=int(not info['valid'])
                rewards.append(reward); steps += 1
            returns.extend(rollout_returns(model,rewards,obs,terminated=info['terminated']).tolist())
            recent_success.append(info['success'])
            episode_count+=1
            if steps >= budget: break
        tensors=[torch.tensor(np.asarray(x),dtype=dtype) for x,dtype in
                 [(states,torch.float32),(faces,torch.long),(raw_actions,torch.float32),
                  (old_logs,torch.float32),(returns,torch.float32)]]
        # Fixed targets for the critic-only phases, which bypass ppo_loss.
        fixed_returns = tensors[-1].detach()
        advantages = tensors[-1]-torch.tensor(old_values)
        # Optional PPO convention: center and scale once per collected batch,
        # keeping actor-loss magnitudes comparable. Centering can change signs
        # and clipping behavior. Critic targets stay in original reward units.
        # unbiased=False uses the batch variance with denominator N.
        advantages = (advantages-advantages.mean())/(advantages.std(unbiased=False)+1e-8)
        if update_count<critic_warmup_batches:
            # Fit a baseline on the new objective and exploration distribution
            # before the first actor update. This uses online training data only.
            for epoch in range(20):
                loss=.25*critic_loss(model.critic(tensors[0]).squeeze(-1),fixed_returns)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.critic.parameters(),.5);optimizer.step()
        elif algorithm=='reinforce':
            # Baseline predictions were recorded before fitting this batch.
            loss = reinforce_loss(model,*tensors[:3],advantages)
            optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
            # The critic is a separate network: fitting it cannot change the actor.
            for epoch in range(10):
                loss = .25*critic_loss(model.critic(tensors[0]).squeeze(-1),fixed_returns)
                optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
        else:
            stop_update=False
            for epoch in range(epochs):
                for indices in torch.randperm(len(states)).split(64):
                    if target_kl is not None:
                        with torch.no_grad():
                            current,_,_=model.evaluate(*[t[indices] for t in tensors[:3]])
                            log_ratio=current-tensors[3][indices]
                            approximate_kl=((log_ratio.exp()-1)-log_ratio).mean()
                        if approximate_kl>target_kl:
                            stop_update=True;kl_stops+=1;break
                    loss = ppo_loss(model,*(t[indices] for t in tensors),advantages[indices],
                                    clip=clip,entropy_coef=entropy_coef)
                    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite PPO loss')
                    if demonstrations is not None and bc_weight:
                        demo_indices=torch.randint(len(demonstrations[0]),(64,))
                        weight=bc_weight*max(0.,1-steps/budget)
                        loss=loss+weight*imitation_loss(model,*(t[demo_indices] for t in demonstrations))
                    optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),.5);optimizer.step()
                if stop_update:break
        update_count+=1
        if steps >= next_monitor or steps >= budget:
            monitor();next_monitor = (steps//512+1)*512
    training_seconds = time.perf_counter()-start
    tests,_ = evaluate(model,plan['test_seeds'])
    sampled,_ = evaluate(model,plan['test_seeds'],sampled=True)
    result = dict(plan=plan,steps=steps,learning=rows,test=tests,sampled_test=sampled,
                  training_episodes=episode_count,policy_batches=update_count,training_invalid=training_invalid,
                  **summary(tests),**execution_summary(tests),sampled={**summary(sampled),**execution_summary(sampled)},training_and_monitoring_seconds=training_seconds,
                  total_seconds=time.perf_counter()-start)
    if evaluate_best:
        best_path=out/f'{base.name}_best.pt'
        selected=torch.load(best_path,map_location='cpu',weights_only=True)
        best_tests=tests if selected['steps']==steps else evaluate(load(best_path),plan['test_seeds'])[0]
        result['best_monitoring']=dict(checkpoint=str(best_path),steps=selected['steps'],test=best_tests,
            **summary(best_tests),**execution_summary(best_tests))
        result['total_seconds']=time.perf_counter()-start
    base.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HYBRID RESULT',seed,{k:v for k,v in result.items() if k not in ('plan','learning','test','sampled_test','best_monitoring')},flush=True)
    if evaluate_best:print('BEST MONITORING RESULT',seed,{k:v for k,v in result['best_monitoring'].items() if k!='test'},flush=True)
    return result
