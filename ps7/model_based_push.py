"""The learned push-transition model and the MPPI planner that uses it.

Dynamics predicts the block and finger configuration after one complete push
command (approach, push, retract, settle), and how long the command takes.
MPPI tries every sequence of H faces and, for each, improves the push
distances by cost-weighted averaging of random perturbations (Williams et al.,
https://arxiv.org/abs/1509.01149). The planner's rollouts run on the learned
model, and the chosen command is then executed in Drake.
"""
from itertools import product
import time
import numpy as np
import torch
from torch import nn
from scipy.special import expit, logit
from model_based_math import (fit_displacement, calibrated_length, dynamics_loss,
                              rollout_cost, cost_weights, update_nominal)

FACE_ANGLES=np.array([0.,np.pi,np.pi/2,-np.pi/2])
STATE_NAMES=['x','y','yaw','finger_x','finger_y','z','roll','pitch']


def wrap(x):return np.arctan2(np.sin(x),np.cos(x))


def state_of(env):
    from pydrake.all import RollPitchYaw
    t=env.task;X=t.plant.EvalBodyPoseInWorld(t.pc,t.box)
    angles=RollPitchYaw(X.rotation()).vector()
    return np.r_[X.translation()[:2],angles[2],t.finger_position(),
        X.translation()[2],angles[:2]]


def local(v,angle):
    c,s=np.cos(angle),np.sin(angle)
    return np.stack([c*v[...,0]+s*v[...,1],-s*v[...,0]+c*v[...,1]],axis=-1)


def world(v,angle):return local(v,-angle)


def inputs(states,faces,lengths):
    a=states[:,2]+FACE_ANGLES[np.asarray(faces,dtype=int)]
    finger=local(states[:,3:5]-states[:,:2],a)
    # The goal and the command count do not affect the physics, so they are not inputs.
    return np.c_[states[:,:2]/.65,np.cos(a),np.sin(a),finger/.5,
        np.abs(finger)/.5,np.linalg.norm(finger+[.185,0.],axis=1)/.5,
        np.asarray(lengths)/.125,states[:,5]/.075,states[:,6:8]]


def targets(states,faces,next_states,seconds):
    a=states[:,2]+FACE_ANGLES[np.asarray(faces,dtype=int)]
    return np.c_[local(next_states[:,:2]-states[:,:2],a),
        wrap(next_states[:,2]-states[:,2]),
        local(next_states[:,3:5]-next_states[:,:2],a),next_states[:,5:8],
        np.log(np.maximum(seconds,.02))]


def decode(states,faces,prediction):
    a=states[:,2]+FACE_ANGLES[np.asarray(faces,dtype=int)]
    out=np.empty((len(states),8),dtype=float)
    out[:,:2]=states[:,:2]+world(prediction[:,:2],a)
    out[:,2]=wrap(states[:,2]+prediction[:,2])
    out[:,3:5]=out[:,:2]+world(prediction[:,3:5],a)
    out[:,5:8]=prediction[:,5:8]
    seconds=np.exp(np.clip(prediction[:,-1],np.log(.02),np.log(60.)))
    return out,seconds


def geometric_validity(states,faces,lengths):
    a=states[:,2]+FACE_ANGLES[np.asarray(faces,dtype=int)]
    direction=np.c_[np.cos(a),np.sin(a)]
    start=states[:,:2]-.185*direction
    end=start+(np.asarray(lengths)+.035)[:,None]*direction
    finger_body=local(states[:,3:5]-states[:,:2],states[:,2])
    # A conservative planar approximation of the executor's exact validity check.
    inside=(np.abs(finger_body)<.165).all(axis=1)
    return (~inside)&(np.abs(states[:,3:5])<=.85).all(axis=1)&(
        np.abs(start)<=.85).all(axis=1)&(np.abs(end)<=.85).all(axis=1)


def heuristic(state,goal,calibration=None):
    error=local(np.asarray(goal)-state[:2],state[2])
    axis=int(np.argmax(np.abs(error)));face=2*axis+int(error[axis]<0)
    alpha,beta=(1.,-.02) if calibration is None else (
        calibration['alpha'],calibration['beta'])
    if alpha<=0:raise ValueError('Calibration slope must be positive')
    return face,float(calibrated_length(abs(error[axis]),alpha,beta))


def calibrate(records):
    valid=[r for r in records if r['valid']]
    x=np.array([r['length'] for r in valid])
    y=np.array([local(np.array(r['next_state'][:2])-r['state'][:2],
        r['state'][2]+FACE_ANGLES[r['face']])[0] for r in valid])
    coefficient=fit_displacement(x,y)
    if coefficient[0]<=0:raise ValueError('Nonpositive calibration slope')
    return dict(alpha=float(coefficient[0]),beta=float(coefficient[1]),
                n=len(x),rmse_m=float(np.sqrt(np.mean((coefficient[0]*x+coefficient[1]-y)**2))))


class Dynamics(nn.Module):
    def __init__(self,xdim,ydim=9):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(xdim,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,ydim))
        for key,n in [('xmean',xdim),('xscale',xdim),('ymean',ydim),('yscale',ydim)]:
            self.register_buffer(key,torch.zeros(n) if key.endswith('mean') else torch.ones(n))

    def forward(self,x):return self.net((x-self.xmean)/self.xscale)

    @torch.no_grad()
    def predict(self,states,faces,lengths):
        x=torch.as_tensor(inputs(states,faces,lengths),dtype=torch.float32)
        pred=(self(x)*self.yscale+self.ymean).numpy().astype(float)
        return decode(states,faces,pred)


def array_data(records):
    rows=[r for r in records if r['valid']]
    states=np.array([r['state'] for r in rows]);faces=np.array([r['face'] for r in rows])
    lengths=np.array([r['length'] for r in rows]);nxt=np.array([r['next_state'] for r in rows])
    seconds=np.array([r['seconds'] for r in rows])
    return inputs(states,faces,lengths),targets(states,faces,nxt,seconds)


def fit_dynamics(train,validation,path,seed=200,epochs=400):
    started=time.perf_counter();torch.set_num_threads(1);torch.manual_seed(seed)
    x,y=array_data(train);vx,vy=array_data(validation)
    model=Dynamics(x.shape[1]);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=.0001)
    # Minimum output scales, so nearly constant targets are not blown up by normalization.
    floors=np.array([.01,.01,.03,.02,.02,.005,.01,.01,.1])
    with torch.no_grad():
        model.xmean.copy_(torch.tensor(x.mean(0)));model.xscale.copy_(torch.tensor(np.maximum(x.std(0),.05)))
        model.ymean.copy_(torch.tensor(y.mean(0)));model.yscale.copy_(torch.tensor(np.maximum(y.std(0),floors)))
    xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y,dtype=torch.float32)
    vxt=torch.tensor(vx,dtype=torch.float32);vyt=torch.tensor(vy,dtype=torch.float32)
    best=float('inf');selected=0;history=[]
    for epoch in range(epochs):
        for indices in torch.randperm(len(x)).split(128):
            loss=dynamics_loss(model(xt[indices]),yt[indices],model.ymean,model.yscale)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite model loss')
            optimizer.zero_grad();loss.backward();optimizer.step()
        if epoch%10==0 or epoch==epochs-1:
            with torch.no_grad():v=float(dynamics_loss(model(vxt),vyt,model.ymean,model.yscale))
            history.append(dict(epoch=epoch+1,validation_loss=v))
            if v<best:
                best=v;selected=epoch+1
                torch.save(dict(model=model.state_dict(),xdim=x.shape[1],seed=seed,epoch=selected,
                                state_version='resting_configuration_v1'),path)
    loaded=load_dynamics(path)
    return loaded,dict(seed=seed,epochs=epochs,selected_epoch=selected,validation_loss=best,
        train_transitions=len(x),validation_transitions=len(vx),seconds=time.perf_counter()-started,history=history)


def load_dynamics(path):
    data=torch.load(path,map_location='cpu',weights_only=True)
    m=Dynamics(data['xdim'],len(data['model']['ymean']))
    m.load_state_dict(data['model']);m.eval();return m


def diagnostics(model,records):
    rows=[r for r in records if r['valid']]
    s=np.array([r['state'] for r in rows]);n=np.array([r['next_state'] for r in rows])
    f=np.array([r['face'] for r in rows]);l=np.array([r['length'] for r in rows]);t=np.array([r['seconds'] for r in rows])
    pred,sec=model.predict(s,f,l)
    delta=pred-n;delta[:,2]=wrap(delta[:,2])
    result=dict(n=len(rows),position_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:,:2]**2,axis=1)))),
        yaw_rmse_rad=float(np.sqrt(np.mean(delta[:,2]**2))),
        finger_rmse_m=float(np.sqrt(np.mean(np.sum(delta[:,3:5]**2,axis=1)))),
        time_rmse_s=float(np.sqrt(np.mean((sec-t)**2))),
        max_observed_tilt_rad=float(np.abs(s[:,6:8]).max()),
        finite=bool(np.isfinite(pred).all() and np.isfinite(sec).all()))
    multi=[]
    for episode in sorted(set(r['episode'] for r in rows)):
        ep=[r for r in rows if r['episode']==episode]
        for start in range(len(ep)-2):
            z=np.array(ep[start]['state'])[None]
            for r in ep[start:start+3]:z,_=model.predict(z,np.array([r['face']]),np.array([r['length']]))
            multi.append(float(np.linalg.norm(z[0,:2]-np.array(ep[start+2]['next_state'])[:2])))
    result['three_step_position_rmse_m']=float(np.sqrt(np.mean(np.square(multi)))) if multi else None
    return result


class MPPI:
    def __init__(self,model,calibration,*,horizon=2,samples=24,iterations=3,temperature=.05,noise=1.0):
        self.model,self.calibration=model,calibration
        self.horizon,self.samples,self.iterations=horizon,samples,iterations
        self.temperature,self.noise=temperature,noise

    def rollout(self,state,goal,faces,raw):
        n,h=faces.shape;s=np.repeat(state[None],n,axis=0);alive=np.ones(n,dtype=bool)
        success=np.zeros(n,dtype=bool);failure=np.zeros(n,dtype=bool);elapsed=np.zeros(n);invalid=np.zeros(n)
        lengths=.125*expit(raw)
        for t in range(h):
            valid=geometric_validity(s,faces[:,t],lengths[:,t]);active=alive&valid
            pred,seconds=self.model.predict(s,faces[:,t],lengths[:,t])
            invalid+=(alive&~valid);elapsed+=seconds*active
            s[active]=pred[active]
            # Success: block within 2.5 cm of the goal. Failure (fall or escape):
            # block center beyond 0.65 m in x or y, or block height below 0.04 m.
            hit=(np.linalg.norm(s[:,:2]-goal,axis=1)<.025)&active
            bad=((np.abs(s[:,:2])>.65).any(axis=1)|(s[:,5]<.04))&active
            success|=hit;failure|=bad;alive&=~(hit|bad)
        return rollout_cost(np.linalg.norm(s[:,:2]-goal,axis=1),success,elapsed,invalid,failure)

    def act(self,state,goal,rng):
        # The horizon stays H even near the 32-command cutoff.
        begin=time.perf_counter();h=self.horizon
        faces=np.array(list(product(range(4),repeat=h)),dtype=int);m=len(faces)
        nominal=np.zeros((m,h));s=np.repeat(state[None],m,axis=0)
        # The initial guess for each face sequence inverts the affine fit toward
        # the goal at each push, with the learned model predicting the state
        # between pushes.
        for t in range(h):
            a=s[:,2]+FACE_ANGLES[faces[:,t]]
            along=local(np.repeat(np.asarray(goal)[None],m,axis=0)-s[:,:2],a)[:,0]
            length=calibrated_length(along,self.calibration['alpha'],self.calibration['beta'])
            # Clip away from 0 and 1 so the inverse sigmoid stays finite.
            eps=np.finfo(float).eps
            nominal[:,t]=logit(np.clip(length/.125,eps,1-eps))
            s,_=self.model.predict(s,faces[:,t],length)
        calls=m*h
        for _ in range(self.iterations):
            # Candidate 0 of each face sequence is the unperturbed nominal, so the
            # current guess is always scored.
            epsilon=rng.normal(0,self.noise,size=(m,self.samples,h));epsilon[:,0]=0
            samples=nominal[:,None,:]+epsilon
            costs=self.rollout(state,goal,np.repeat(faces,self.samples,axis=0),samples.reshape(-1,h)).reshape(m,self.samples)
            nominal=update_nominal(nominal,epsilon,cost_weights(costs,self.temperature))
            calls+=m*self.samples*h
        costs=self.rollout(state,goal,faces,nominal);calls+=m*h
        best=int(costs.argmin());face=int(faces[best,0]);length=float(.125*expit(nominal[best,0]))
        return face,length,dict(planning_seconds=time.perf_counter()-begin,model_calls=calls,
            predicted_cost=float(costs[best]),face_sequence=faces[best].tolist(),
            distance_sequence=(.125*expit(nominal[best])).tolist())
