"""Explicit, optional physics variants for bounded instructor pilots.

Inclination is expressed in table coordinates by rotating gravity, including
its effect on the finger. A rounded support cap is a sphere intersecting the
box bottom; its exposed lower half/cap changes contact geometry. Its mass is
neglected (a thin contact covering), preserving the supplied body inertia.
"""
from dataclasses import dataclass,asdict
import re
import numpy as np

@dataclass(frozen=True)
class PushPhysics:
    mass_scale: float=1.
    friction: float|None=None
    tilt_deg: float=0.
    bump_x: float=0.
    bump_y: float=0.
    bump_radius: float=0.
    bump_height: float=0.
    bump_friction: float=.8
    bump_resolution: float=.003

    def __post_init__(self):
        values=[v for v in asdict(self).values() if v is not None]
        if not all(np.isfinite(v) for v in values):raise ValueError('Finite physics parameters required')
        if self.mass_scale<=0 or abs(self.tilt_deg)>20:raise ValueError('Invalid mass scale or inclination')
        if self.friction is not None and self.friction<0:raise ValueError('Friction must be nonnegative')
        if self.bump_friction<0:raise ValueError('Bump friction must be nonnegative')
        if self.bump_radius<0 or not 0<=self.bump_height<=self.bump_radius:raise ValueError('Expose at most a hemisphere')
        if self.bump_radius and (max(abs(self.bump_x),abs(self.bump_y))+self.bump_radius>.1 or self.bump_resolution<=0):raise ValueError('Bump must fit inside box footprint')

    @property
    def initial_height(self):return .075+self.bump_height

    def gravity(self):
        angle=np.radians(self.tilt_deg)
        return [9.81*np.sin(angle),0.,-9.81*np.cos(angle)]

    def sdf(self,box,table):
        def scale(match):return f'<{match[1]}>{float(match[2])*self.mass_scale:.17g}</{match[1]}>'
        if self.mass_scale!=1:
            box=re.sub(r'<(mass|ixx|iyy|izz)>([^<]+)</\1>',scale,box)
        def surface(mu):return f'<surface><friction><ode><mu>{mu}</mu><mu2>{mu}</mu2></ode></friction></surface>'
        if self.friction is not None:
            box=box.replace('</collision>',surface(self.friction)+'</collision>')
            table=table.replace('</collision>',surface(self.friction)+'</collision>')
        if self.bump_radius:
            z=-.025+self.bump_radius-self.bump_height
            pose=f'<pose>{self.bump_x} {self.bump_y} {z} 0 0 0</pose>'
            geometry=f'<geometry><sphere><radius>{self.bump_radius}</radius></sphere></geometry>'
            collision=f'''<collision name="support_bump">{pose}{geometry}{surface(self.bump_friction)}
              <drake:proximity_properties><drake:compliant_hydroelastic/>
              <drake:hydroelastic_modulus>1e7</drake:hydroelastic_modulus>
              <drake:mesh_resolution_hint>{self.bump_resolution}</drake:mesh_resolution_hint>
              <drake:point_contact_stiffness>1e6</drake:point_contact_stiffness>
              </drake:proximity_properties></collision>'''
            visual=f'<visual name="support_bump">{pose}{geometry}<material><diffuse>0.8 0.25 0.1 1</diffuse></material></visual>'
            box=box.replace('</link>',collision+visual+'</link>')
        return box,table
