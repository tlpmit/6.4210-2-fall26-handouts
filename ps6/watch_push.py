"""Drake/Meshcat replay of the learned straight-push manipulation pilot."""
from pathlib import Path
import html
import numpy as np
from push_primitive import PushPrimitive

def watch_analytic_push(seed=122000, output_dir=None, show=True):
    """Run the supplied analytic controller in the homework world and replay it.

    Inputs: integer starting-state seed, optional output directory, and a bool
    controlling notebook display. No student loss or model-fitting function is
    called. Returns the usual Meshcat replay dict plus a 'rollout' summary dict
    (seed, success, pushes, distance in meters, ending reason, and invalid count).
    Every call runs one fresh episode, with at most 16 pushes.
    """
    import json
    from clone_multi_push import expert_action
    from push_env import LearningPush, FACES
    from push_physics import PushPhysics

    root = Path(__file__).resolve().parent
    plan = json.loads((root / 'data/push/plan.json').read_text())
    env = LearningPush(horizon=plan['horizon'], timestep=plan['timestep'],
                       physics=PushPhysics(**plan['physics']), record=True)
    env.reset(seed=int(seed))
    invalid = 0
    while not env.task.done:
        face, distance = expert_action(env.observation())
        axis, sign = FACES[face]
        _, _, _, _, info = env.task.step(env.task.face_push(axis, sign, distance))
        invalid += int(not info['valid'])
    summary = dict(seed=int(seed), success=bool(info['success']),
                   pushes=env.task.count, distance=float(np.linalg.norm(
                       env.task.box_pose()[:2] - env.task.goal)),
                   reason=info['reason'], invalid=invalid)
    print(f"Analytic controller: {summary['pushes']} pushes; "
          f"ending reason: {summary['reason']}; success: {summary['success']}")
    print(f"Final distance: {summary['distance']:.4f} m; "
          f"goal tolerance: {env.task.success_tolerance:.4f} m")
    folder = Path(output_dir or root / 'results/replays') / f'analytic_seed{seed}'
    replay = watch_multi_push(env.task, output_dir=folder, show=show)
    replay['rollout'] = summary
    return replay

def watch_push(travel=.25,goal=.2,output_dir=None,show=True):
    from pydrake.all import (StartMeshcat,MeshcatVisualizer,MeshcatVisualizerParams,
                            Box,Rgba,RigidTransform)
    # Physics always runs without rendering. Rebuild only for state playback.
    recorded=PushPrimitive(timestep=.005,record=True)
    final=recorded.execute(travel)
    meshcat=StartMeshcat()
    meshcat.SetCameraPose([.6,-.85,.8],[.12,0,.07])
    # A stand-alone visualizer consumes the already-built diagram's query output.
    # Use its API directly; no new simulator or contact solve is needed.
    params=MeshcatVisualizerParams();params.publish_period=.02
    visualizer=MeshcatVisualizer(meshcat,params)
    vc=visualizer.CreateDefaultContext()
    meshcat.SetObject('goal',Box(.2,.2,.05),Rgba(.2,.8,.3,.35))
    meshcat.SetTransform('goal',RigidTransform([goal,0,.075]))
    visualizer.StartRecording()
    for k,state in enumerate(recorded.states):
        recorded.context.SetTime((k+1)*.02)
        recorded.plant.SetPositionsAndVelocities(recorded.pc,state)
        scene_context=recorded.scene_graph.GetMyContextFromRoot(recorded.context)
        query=recorded.scene_graph.get_query_output_port().Eval(scene_context)
        visualizer.query_object_input_port().FixValue(vc,query)
        vc.SetTime((k+1)*.02)
        visualizer.ForcedPublish(vc)
    visualizer.StopRecording();visualizer.PublishRecording()
    output_dir=Path(output_dir or Path(__file__).parent/'results/push_primitive')
    output_dir.mkdir(parents=True,exist_ok=True)
    path=output_dir/f'push_travel{travel:.3f}_goal{goal:.3f}.html'
    contents=meshcat.StaticHtml();path.write_text(contents)
    if show:
        from IPython.display import HTML,display
        display(HTML(f'<iframe srcdoc="{html.escape(contents,quote=True)}" width="100%" height="500"></iframe>'))
    print('Final box position:',final,'goal:',goal,'Meshcat:',meshcat.web_url())
    return dict(meshcat=meshcat,path=path,final_position=final,states=recorded.states)

def watch_multi_push(recorded=None, output_dir=None, show=True):
    """Replay recorded multi-push physics; never rerun contact while rendering."""
    from multi_push import MultiPush, scripted_episode
    from pydrake.all import (StartMeshcat, MeshcatVisualizer, MeshcatVisualizerParams,
                            Box, Sphere, Rgba, RigidTransform)
    if recorded is None:
        recorded = MultiPush(record=True)
        scripted_episode(recorded)
    if not recorded.states:
        raise ValueError('Run an episode with record=True before playback')
    final = recorded.box_pose().copy()
    # Use a second plant context so replay does not mutate the completed episode.
    replay = MultiPush(record=False,box_com_x=recorded.box_com_x,mass_model=recorded.mass_model,
                       timestep=recorded.timestep,physics=getattr(recorded,'physics',None))
    meshcat = StartMeshcat()
    meshcat.SetCameraPose([.85,-1.15,1.5],[.05,.05,.05])
    params = MeshcatVisualizerParams(); params.publish_period = .02
    visualizer = MeshcatVisualizer(meshcat, params)
    vc = visualizer.CreateDefaultContext()
    meshcat.SetObject('goal', Box(.2,.2,.05), Rgba(.2,.8,.3,.3))
    meshcat.SetTransform('goal', RigidTransform([*recorded.goal,.075]))
    if recorded.box_com_x:
        # Marker projects the COM onto the top face so it remains visible.
        meshcat.SetObject('mass_center_projection',Sphere(.012),Rgba(1.,.7,.1,1.))
    visualizer.StartRecording()
    for t, state in zip(recorded.times, recorded.states):
        replay.context.SetTime(t)
        replay.plant.SetPositionsAndVelocities(replay.pc, state)
        if recorded.box_com_x:
            pose=replay.plant.EvalBodyPoseInWorld(replay.pc,replay.box)
            meshcat.SetTransform('mass_center_projection',pose @ RigidTransform([recorded.box_com_x,0.,.03]),time_in_recording=t)
        sc = replay.scene_graph.GetMyContextFromRoot(replay.context)
        visualizer.query_object_input_port().FixValue(vc, replay.scene_graph.get_query_output_port().Eval(sc))
        vc.SetTime(t); visualizer.ForcedPublish(vc)
    visualizer.StopRecording(); visualizer.PublishRecording()
    output_dir = Path(output_dir or Path(__file__).parent/'results/multi_push')
    output_dir.mkdir(parents=True,exist_ok=True)
    path = output_dir/'multi_push.html'
    contents = meshcat.StaticHtml(); path.write_text(contents)
    if show:
        from IPython.display import HTML,display
        display(HTML(f'<iframe srcdoc="{html.escape(contents,quote=True)}" width="100%" height="550"></iframe>'))
    print('Final pose:', final, 'goal:', recorded.goal, 'pushes:', len(recorded.history))
    return dict(meshcat=meshcat,path=path,final_pose=final)
