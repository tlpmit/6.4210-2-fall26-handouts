"""Clone analytic face/distance commands from actual Drake demonstrations."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from push_math import bc_loss
from push_env import (LearningPush, FACES, summary, configuration_observations,
                      OBSERVATION_DIM, OBSERVATION_VERSION)
from train_hybrid_push import HybridActorCritic, distance_from_raw, evaluate, DISTANCE_MIN, DISTANCE_MAX

ROOT=Path(__file__).parent/'results/push_imitation'


def expert_action(observation):
    """The existing analytic controller expressed in the hybrid action space.

    Only the observation is used, never a goal ID or simulator hidden state.
    The 2 cm compensation is the existing scripted controller's tracking
    correction. Cloning learns these labels; no analytic calls occur at deploy.
    """
    error=np.asarray(observation)[:2]*.3
    axis=int(np.argmax(np.abs(error)))
    face=2*axis+int(error[axis]<0)
    distance=float(np.clip(abs(error[axis])+.02,DISTANCE_MIN,DISTANCE_MAX))
    return face,distance


def dataset(paths):
    arrays=[dict(np.load(path,allow_pickle=False)) for path in paths]
    keys=['observations','faces','distances','returns','episode_ids']
    data = {key:np.concatenate([a[key] for a in arrays]) for key in keys}
    data['observations'] = configuration_observations(data['observations'])
    return data


def imitation_loss(model, observations, faces, distances):
    """Call the student BC loss with predictions and fixed demonstration labels."""
    distances = distances.detach()
    hidden=model.actor(observations)
    logits=model.face(hidden)
    raw=model.mean(hidden).gather(-1,faces[:,None]).squeeze(-1)
    # Regress physical distances: max-length expert labels need no infinite
    # inverse sigmoid, and centimeter errors keep their physical meaning.
    return bc_loss(logits, distance_from_raw(raw), faces, distances)


@torch.no_grad()
def prediction_metrics(model,data):
    obs=torch.from_numpy(data['observations']);faces=torch.from_numpy(data['faces'])
    distance=torch.from_numpy(data['distances'])
    hidden=model.actor(obs);pred=model.face(hidden).argmax(-1)
    raw=model.mean(hidden).gather(-1,faces[:,None]).squeeze(-1)
    return dict(loss=float(imitation_loss(model,obs,faces,distance)),
        face_accuracy=float((pred==faces).float().mean()),
        distance_mae_cm=float(100*(distance_from_raw(raw)-distance).abs().mean()))


def clone(train_paths, validation_paths, seed=30, epochs=400, out=ROOT, *,
          physics=None,timestep=.005,evaluate_rollouts=True):
    torch.set_num_threads(1);torch.manual_seed(seed)
    train=dataset(train_paths);validation=dataset(validation_paths)
    if set(train['episode_ids']) & set(validation['episode_ids']):
        raise ValueError('Demonstration training and validation episodes overlap')
    model=HybridActorCritic();model.training_horizon=16
    model.observation_version=OBSERVATION_VERSION
    # This controls online exploration, not the deterministic imitation loss.
    with torch.no_grad():model.log_std.fill_(np.log(.15))
    actor_parameters=[*model.actor.parameters(),*model.face.parameters(),*model.mean.parameters()]
    actor_optimizer=torch.optim.Adam(actor_parameters,lr=.001)
    obs=torch.from_numpy(train['observations']);faces=torch.from_numpy(train['faces'])
    distances=torch.from_numpy(train['distances'])
    out=Path(out);out.mkdir(parents=True,exist_ok=True);base=out/f'bc_seed{seed}'
    plan=dict(seed=seed,algorithm='Behavior cloning of analytic face/distance actions',horizon=16,
        distance_bounds=[DISTANCE_MIN,DISTANCE_MAX],
        observation_version=OBSERVATION_VERSION,observation_dim=OBSERVATION_DIM,
        data_note='Archived heuristic action labels projected onto configuration inputs; no velocity/countdown inputs. No archived return targets used.',
        physics=None if physics is None else asdict(physics),timestep=timestep,
        epochs=epochs,learning_rate=.001,batch_size=128,distance_loss_weight=1.,
        distance_normalization_m=.1,raw_distance_std=.15,
        train_files=[str(p) for p in train_paths],validation_files=[str(p) for p in validation_paths],
        selection='Lowest held-out demonstration imitation loss, evaluated every 20 epochs',
        critic='Not pretrained; PPO initializes a fresh critic for the current objective',
        deployment='Neural actor only: no analytic action, correction, or fallback')
    base.with_suffix('.plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    best=float('inf');rows=[];started=time.perf_counter()
    for epoch in range(epochs+1):
        if epoch%20==0 or epoch==epochs:
            metrics=prediction_metrics(model,validation)
            row=dict(epoch=epoch,seconds=time.perf_counter()-started,**metrics);rows.append(row)
            if metrics['loss']<best:
                best=metrics['loss']
                torch.save(dict(model=model.state_dict(),obs_dim=OBSERVATION_DIM,plan=plan,steps=0,epoch=epoch),base.with_suffix('.pt'))
            base.with_suffix('.progress.json').write_text(json.dumps(rows,indent=2)+'\n')
            if epoch%100==0:print('BC',seed,row,flush=True)
        if epoch==epochs:break
        for batch in torch.randperm(len(obs)).split(128):
            loss=imitation_loss(model,obs[batch],faces[batch],distances[batch])
            actor_optimizer.zero_grad();loss.backward();actor_optimizer.step()
    from train_hybrid_push import load
    selected=load(base.with_suffix('.pt'))
    result=dict(plan=plan,validation=prediction_metrics(selected,validation),
                seconds=time.perf_counter()-started)
    if evaluate_rollouts:
        monitors,_=evaluate(selected,range(33000,33016))
        sampled,_=evaluate(selected,range(33000,33016),sampled=True)
        result.update(monitoring=monitors,sampled_monitoring=sampled,
                      **summary(monitors),sampled=summary(sampled))
    result['seconds']=time.perf_counter()-started
    base.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    print('BC RESULT',seed,{k:v for k,v in result.items() if k not in ('plan','monitoring','sampled_monitoring')},flush=True)
    return result
