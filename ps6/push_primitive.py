"""Small Drake contact-manipulation pilot: learn a straight push's travel.

This is a contextual bandit (one macro action), NOT a full pick-and-place
or general planar pushing task. The supplied position controller executes
the chosen travel; Drake simulates the free box and contact with the table.
"""
from pathlib import Path
import sys
import time
import numpy as np
from push_geometry import finger_sdf, table_sdf, small_box_sdf
from push_mass import mass_properties, with_mass_distribution
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, Parser,
                        ContactModel, Simulator, RigidTransform, SpatialVelocity, MatrixGain, Adder)

class PushPrimitive:
    def __init__(self, timestep=.002, duration=1.6, record=False, box_com_x=0.,mass_model='halves',physics=None):
        self.timestep,self.duration,self.record=timestep,duration,record
        self.box_com_x=float(box_com_x)
        self.mass_model=mass_model
        self.mass_properties=mass_properties(self.box_com_x,mass_model)
        self.physics=physics
        self.initial_height=.075 if physics is None else physics.initial_height
        builder=DiagramBuilder()
        self.plant,self.scene_graph=AddMultibodyPlantSceneGraph(builder,timestep)
        self.plant.set_contact_model(ContactModel.kHydroelasticWithFallback)
        parser=Parser(self.plant)
        box_sdf=with_mass_distribution(small_box_sdf,self.box_com_x,mass_model)
        surface_sdf=table_sdf
        if physics is not None:
            box_sdf,surface_sdf=physics.sdf(box_sdf,surface_sdf)
            self.plant.mutable_gravity_field().set_gravity_vector(physics.gravity())
            self.mass_properties=dict(self.mass_properties)
            for key in self.mass_properties:
                if key=='mass_kg' or key.endswith('_mass_kg'):
                    self.mass_properties[key]*=physics.mass_scale
            self.mass_properties['central_inertia_diagonal_kg_m2']=[v*physics.mass_scale for v in self.mass_properties['central_inertia_diagonal_kg_m2']]
        for sdf in [surface_sdf,box_sdf,finger_sdf]: parser.AddModelsFromString(sdf,'sdf')
        self.plant.Finalize()
        self.box=self.plant.GetBodyByName('small_box_link')
        self.joints=[self.plant.GetJointByName(n) for n in ['finger_x','finger_y']]
        # Same P control as misc/pusher.py, expressed with C++ Drake systems so
        # Python is not called during every contact-solver step. Joint damping
        # supplies the derivative term, as in that original controller.
        feedback=np.zeros((self.plant.num_velocities(),self.plant.num_multibody_states()))
        feedforward=np.zeros((self.plant.num_velocities(),2))
        for axis,joint in enumerate(self.joints):
            feedback[joint.velocity_start(),joint.position_start()]=-100.
            feedforward[joint.velocity_start(),axis]=100.
        fb=builder.AddSystem(MatrixGain(feedback)); ff=builder.AddSystem(MatrixGain(feedforward))
        add=builder.AddSystem(Adder(2,self.plant.num_velocities()))
        builder.Connect(self.plant.get_state_output_port(),fb.get_input_port())
        builder.Connect(fb.get_output_port(),add.get_input_port(0))
        builder.Connect(ff.get_output_port(),add.get_input_port(1))
        builder.Connect(add.get_output_port(),self.plant.get_applied_generalized_force_input_port())
        builder.ExportInput(ff.get_input_port(),'target')
        self.diagram=builder.Build()
        self.sim=Simulator(self.diagram)
        self.context=self.sim.get_mutable_context()
        self.pc=self.plant.GetMyMutableContextFromRoot(self.context)
        self.target=self.diagram.GetInputPort('target')
        self.states=[]

    def execute(self, travel):
        """Travel in meters. Start aligned behind a box, push, and settle."""
        travel=float(np.clip(travel,0.,.5))
        self.context.SetTime(0.)
        self.plant.SetFreeBodyPose(self.pc,self.box,RigidTransform([0.,0.,self.initial_height]))
        self.plant.SetFreeBodySpatialVelocity(self.pc,self.box,SpatialVelocity(np.zeros(6)))
        for joint,pos in zip(self.joints,[-.18,0.]):
            joint.set_translation(self.pc,pos); joint.set_translation_rate(self.pc,0.)
        self.target.FixValue(self.context,[-.18,0.])
        self.sim.Initialize()
        self.states=[]
        for k in range(round(self.duration/.02)):
            t=(k+1)*.02
            # Ramp for 1 second, then hold to dissipate velocity.
            x=-.18+travel*min(1.,t/1.)
            self.target.FixValue(self.context,[x,0.])
            self.sim.AdvanceTo(t)
            if self.record: self.states.append(self.plant.GetPositionsAndVelocities(self.pc).copy())
        pose=self.plant.EvalBodyPoseInWorld(self.pc,self.box)
        return pose.translation().copy()
