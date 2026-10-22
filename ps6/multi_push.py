"""Sequential Drake pushes with geometric free-space planning between contacts.

Push.start is the *pusher center* in world xy before a straight stroke;
Push.vector is its commanded displacement, in meters. Physics is continuous
across actions. Only reset() changes object/finger positions directly.
The planner treats the stationary box as an inflated oriented rectangle and
finds a shortest visibility-graph route in the two-DOF finger's configuration
space. This is planar motion planning, not an articulated-arm planner.
"""
from dataclasses import dataclass
import heapq
import numpy as np
from pydrake.all import RigidTransform, RollPitchYaw, SpatialVelocity
from push_primitive import PushPrimitive


@dataclass(frozen=True)
class Push:
    start: object
    vector: object


def segment_hits_rectangle(a, b, half):
    """Exact slab intersection, including the rectangle boundary."""
    a, b, half = map(np.asarray, (a, b, half))
    lower, upper = 0., 1.
    for axis in range(2):
        delta = b[axis] - a[axis]
        if abs(delta) < 1e-12:
            if abs(a[axis]) > half[axis]:
                return False
        else:
            bounds = sorted(((-half[axis]-a[axis])/delta,
                             (half[axis]-a[axis])/delta))
            lower, upper = max(lower, bounds[0]), min(upper, bounds[1])
            if lower > upper:
                return False
    return True


class FreeSpacePlanner:
    """Conservative collision geometry: box projection + finger radius + margin."""
    def __init__(self, center, yaw, half=(.1, .1), radius=.05, margin=.015,
                 workspace=.85):
        self.center = np.asarray(center, dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        self.rotation = np.array([[c, -s], [s, c]])
        self.half = np.asarray(half) + radius + margin
        self.workspace = workspace

    def local(self, p):
        return (np.asarray(p)-self.center) @ self.rotation

    def world(self, p):
        return np.asarray(p) @ self.rotation.T + self.center

    def edge_free(self, a, b):
        return (np.max(np.abs([a, b])) <= self.workspace and
                not segment_hits_rectangle(self.local(a), self.local(b), self.half))

    def plan(self, start, goal):
        start, goal = map(lambda x: np.asarray(x, dtype=float), (start, goal))
        if not self.edge_free(start, start) or not self.edge_free(goal, goal):
            raise ValueError('Start or goal is outside free configuration space')
        # Offset vertices avoid routing exactly on an inclusive boundary.
        corners = self.world(np.array([[-1,-1], [-1,1], [1,-1], [1,1]]) *
                             (self.half + .003))
        nodes = [start, goal, *corners]
        edges = [[] for _ in nodes]
        for i, a in enumerate(nodes):
            for j in range(i):
                if self.edge_free(a, nodes[j]):
                    length = float(np.linalg.norm(a-nodes[j]))
                    edges[i].append((j, length)); edges[j].append((i, length))
        queue = [(0., 0)]; costs = {0: 0.}; parents = {}
        while queue:
            cost, i = heapq.heappop(queue)
            if cost > costs[i]:
                continue
            if i == 1:
                route = [i]
                while route[-1] != 0:
                    route.append(parents[route[-1]])
                return np.array([nodes[k] for k in route[::-1]])
            for j, length in edges[i]:
                candidate = cost + length
                if candidate < costs.get(j, np.inf):
                    costs[j], parents[j] = candidate, i
                    heapq.heappush(queue, (candidate, j))
        raise ValueError('No collision-free route in the workspace')


class MultiPush(PushPrimitive):
    max_stroke = .16
    control_dt = .02
    success_tolerance = .025
    settle_linear_speed = 1e-4       # m/s, both the block and finger
    settle_angular_speed = 1e-3      # rad/s, block
    settle_hold_seconds = .10       # continuously below thresholds
    settle_timeout_seconds = 3.0

    def __init__(self, timestep=.005, horizon=8, reward_kind='dense', record=False, box_com_x=0.,mass_model='halves',physics=None):
        super().__init__(timestep=timestep, record=record, box_com_x=box_com_x,mass_model=mass_model,physics=physics)
        if reward_kind not in ('dense', 'sparse'):
            raise ValueError('reward_kind must be dense or sparse')
        self.horizon, self.reward_kind = horizon, reward_kind
        self.reset()

    def reset(self, *, goal=(.30, .25), box_pose=(0., 0., 0.), pusher=(-.5, -.4)):
        goal, box_pose, pusher = map(lambda x: np.asarray(x, dtype=float),
                                     (goal, box_pose, pusher))
        if (goal.shape != (2,) or box_pose.shape != (3,) or pusher.shape != (2,) or
            not np.all(np.isfinite(np.r_[goal, box_pose, pusher])) or
            np.max(np.abs(goal)) > .5 or np.max(np.abs(box_pose[:2])) > .5):
            raise ValueError('Invalid reset pose or goal')
        planner = FreeSpacePlanner(box_pose[:2], box_pose[2])
        if not planner.edge_free(pusher, pusher):
            raise ValueError('Initial pusher must be in free space')
        self.context.SetTime(0.)
        self.plant.SetFreeBodyPose(self.pc, self.box,
            RigidTransform(RollPitchYaw(0., 0., box_pose[2]), [*box_pose[:2], self.initial_height]))
        self.plant.SetFreeBodySpatialVelocity(self.pc, self.box, SpatialVelocity(np.zeros(6)))
        for joint, pos in zip(self.joints, pusher):
            joint.set_translation(self.pc, pos); joint.set_translation_rate(self.pc, 0.)
        self.target.FixValue(self.context, pusher)
        self.sim.Initialize()
        self.goal, self.count, self.done = goal.copy(), 0, False
        self.states, self.times, self.phases, self.history = [], [], [], []
        self.motion_length = {phase: 0. for phase in ('free', 'push', 'retract', 'settle')}
        self.motion_seconds = {phase: 0. for phase in self.motion_length}
        self._capture('reset')
        return self.observation(), {}

    def box_pose(self):
        pose = self.plant.EvalBodyPoseInWorld(self.pc, self.box)
        return np.r_[pose.translation()[:2], RollPitchYaw(pose.rotation()).yaw_angle()]

    def finger_position(self):
        return np.array([j.get_translation(self.pc) for j in self.joints])

    def planner(self):
        pose = self.plant.EvalBodyPoseInWorld(self.pc, self.box)
        yaw = RollPitchYaw(pose.rotation()).yaw_angle()
        # Bound all 8 projected box vertices, including small contact-induced tilt.
        corners = np.array([[x,y,z] for x in [-.1,.1] for y in [-.1,.1]
                            for z in [-.025,.025]]) @ pose.rotation().matrix().T
        c, s = np.cos(yaw), np.sin(yaw)
        half = np.max(np.abs(corners[:,:2] @ np.array([[c,-s],[s,c]])), axis=0)
        return FreeSpacePlanner(pose.translation()[:2], yaw, half=half)

    def observation(self):
        # Keep full simulator state for replay and settling diagnostics. The
        # learning wrapper selects configuration only, without velocities/time.
        return dict(state=self.plant.GetPositionsAndVelocities(self.pc).copy(),
                    box_pose=self.box_pose(), pusher=self.finger_position(),
                    goal=self.goal.copy(), remaining=self.horizon-self.count)

    def _capture(self, phase):
        if self.record:
            self.states.append(self.plant.GetPositionsAndVelocities(self.pc).copy())
            self.times.append(self.context.get_time()); self.phases.append(phase)

    def _tick(self, target, phase, guard=None):
        previous = self.finger_position()
        self.target.FixValue(self.context, target)
        self.sim.AdvanceTo(self.context.get_time()+self.control_dt)
        self._capture(phase)
        current = self.finger_position()
        # Execution cost in simulated physical time and realized finger travel,
        # independent of recording, rendering, and machine training throughput.
        self.motion_length[phase] += float(np.linalg.norm(current-previous))
        self.motion_seconds[phase] += self.control_dt
        if guard is not None and not guard.edge_free(previous, current):
            raise RuntimeError('Executed free-space motion violated its clearance margin')

    def _move(self, target, phase, speed=.20, guard=None):
        start = self.finger_position(); target = np.asarray(target)
        ticks = max(1, int(np.ceil(np.linalg.norm(target-start)/speed/self.control_dt)))
        for k in range(1, ticks+1):
            self._tick(start+(target-start)*k/ticks, phase, guard)
        # Stop at every corner, rather than allowing the controller to cut it.
        for _ in range(60):
            self._tick(target, phase, guard)
            velocity = np.array([j.get_translation_rate(self.pc) for j in self.joints])
            if np.linalg.norm(self.finger_position()-target) < .001 and np.linalg.norm(velocity) < .005:
                break

    def step(self, action):
        if self.done:
            raise RuntimeError('Episode is finished; call reset()')
        before = self.box_pose(); initial_time = self.context.get_time()
        info = dict(valid=False, success=False, reason='', path=[],
                    before=before.tolist(), free_box_motion=0.)
        self.count += 1
        try:
            start, vector = np.asarray(action.start, dtype=float), np.asarray(action.vector, dtype=float)
            if (start.shape != (2,) or vector.shape != (2,) or
                not np.all(np.isfinite(np.r_[start, vector])) or
                not 0. < np.linalg.norm(vector) <= self.max_stroke+1e-10 or
                np.max(np.abs([start, start+vector])) > .85):
                raise ValueError('Push needs a finite start and a nonzero vector of at most 0.16 m inside the workspace')
            planner = self.planner()
            route = planner.plan(self.finger_position(), start)
            info['path'] = route.tolist()
        except (ValueError, AttributeError) as error:
            info['reason'] = str(error)
            truncated = self.count >= self.horizon
            self.done = truncated
            info.update(after=before.tolist(), simulated_seconds=0.)
            self.history.append(info)
            return self.observation(), -.2, False, truncated, info
        info['valid'] = True
        for point in route[1:]:
            self._move(point, 'free', guard=planner)
        info['free_box_motion'] = float(np.linalg.norm(self.box_pose()[:2]-before[:2]))
        self._move(start+vector, 'push', speed=.12)
        # Retract along the stroke before asking the free-space planner to route
        # around the object. This separation motion intentionally starts at contact.
        retreat = self.finger_position() - .08*vector/np.linalg.norm(vector)
        self._move(retreat, 'retract', speed=.15)
        settling = self._settle(retreat)
        after = self.box_pose()
        success = bool(np.linalg.norm(after[:2]-self.goal) < self.success_tolerance)
        fallen = bool(self.plant.EvalBodyPoseInWorld(self.pc,self.box).translation()[2] < .04)
        escaped = bool(np.max(np.abs(after[:2])) > .65)
        terminated = success or fallen or escaped
        truncated = self.count >= self.horizon and not terminated
        self.done = terminated or truncated
        progress = (np.linalg.norm(before[:2]-self.goal)-np.linalg.norm(after[:2]-self.goal))/.15
        reward = (progress if self.reward_kind == 'dense' else 0.) + float(success) - .02
        if fallen or escaped:
            reward -= 1.
        info.update(success=success, after=after.tolist(),
                    reason='success' if success else 'workspace' if fallen or escaped else 'cutoff' if truncated else '',
                    simulated_seconds=self.context.get_time()-initial_time,
                    start=start.tolist(), vector=vector.tolist(), **settling)
        self.history.append(info)
        return self.observation(), float(reward), terminated, truncated, info

    def _settle(self, target):
        """Hold the retracted finger until both bodies rest; charge every tick.

        Linear speeds must stay below 0.1 mm/s and block angular speed below
        0.001 rad/s for 0.10 s. A physical failure ends waiting immediately.
        A 3 s timeout raises instead of returning an unsettled learning state.
        """
        quiet = 0
        required = int(np.ceil(self.settle_hold_seconds / self.control_dt))
        start = self.context.get_time()
        for _ in range(int(np.ceil(self.settle_timeout_seconds / self.control_dt))):
            self._tick(target, 'settle')
            velocity = self.plant.EvalBodySpatialVelocityInWorld(self.pc, self.box)
            speeds = dict(block_speed=float(np.linalg.norm(velocity.translational())),
                          block_angular_speed=float(np.linalg.norm(velocity.rotational())),
                          finger_speed=float(np.linalg.norm([
                              j.get_translation_rate(self.pc) for j in self.joints])))
            pose = self.plant.EvalBodyPoseInWorld(self.pc, self.box).translation()
            failed = pose[2] < .04 or np.max(np.abs(pose[:2])) > .65
            resting = (speeds['block_speed'] < self.settle_linear_speed and
                       speeds['finger_speed'] < self.settle_linear_speed and
                       speeds['block_angular_speed'] < self.settle_angular_speed)
            quiet = quiet + 1 if resting else 0
            if failed or quiet >= required:
                return dict(settled=not failed, settling_seconds=self.context.get_time()-start,
                            **speeds)
        raise RuntimeError(f'Bodies did not settle within {self.settle_timeout_seconds:g} s: {speeds}')

    def face_push(self, axis, sign, displacement=.1, offset=0.):
        """Convenience action generator; the policy can instead supply any Push.

        Four faces, selectable stroke length and tangential contact offset.
        Commands refer to the CURRENT box pose, so later actions are feedback.
        displacement is the nonnegative stroke after the 3.5 cm approach
        to contact. Zero is allowed and still incurs approach/retraction cost.
        """
        if axis not in (0,1) or sign not in (-1,1) or abs(offset) > .07:
            raise ValueError('Invalid face/contact offset')
        pose = self.box_pose(); c,s = np.cos(pose[2]), np.sin(pose[2])
        rotation = np.array([[c,-s],[s,c]])
        direction = rotation[:,axis]*sign
        tangent = rotation[:,1-axis]
        start = pose[:2] - .185*direction + offset*tangent
        distance = np.clip(displacement, 0., self.max_stroke-.035)
        return Push(start, direction*(distance+.035))

    def greedy_action(self):
        """Geometric controller used ONLY to validate the environment, not learned RL."""
        pose = self.box_pose(); c,s = np.cos(pose[2]), np.sin(pose[2])
        error = (self.goal-pose[:2]) @ np.array([[c,-s],[s,c]])
        axis = int(np.argmax(np.abs(error)))
        # Approximate 2 cm loaded-controller offset; this is a scripted
        # feasibility controller, not a fitted policy or an exact push model.
        return self.face_push(axis, 1 if error[axis] >= 0 else -1, abs(error[axis])+.02)


def scripted_episode(env, goal=(.30,.25), **reset_kwargs):
    env.reset(goal=goal, **reset_kwargs)
    while not env.done:
        env.step(env.greedy_action())
    return env.history
