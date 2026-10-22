"""A fixed, physically realizable mass asymmetry, without changing contact geometry.

Split the 20 x 20 x 5 cm box into two uniform-density halves along body x.
Their centers are at +/-5 cm. Keeping total mass at 0.2 kg gives COM c when
the positive half has mass fraction (1 + c/.05)/2. Second moments about the
geometric origin stay unchanged; central moments follow by parallel axes.
The body origin, observations, goals, and collision geometry remain geometric.
"""
import xml.etree.ElementTree as ET
import re
import numpy as np


def mass_properties(box_com_x=0.,mass_model='halves'):
    c=float(box_com_x)
    if mass_model not in ('halves','ballast','edge_ballast'):raise ValueError('Unknown mass model')
    ballast_fraction=.99 if mass_model=='edge_ballast' else .9
    bound=.05 if mass_model=='halves' else ballast_fraction*.095
    if not np.isfinite(c) or abs(c)>=bound:
        raise ValueError(f'box_com_x must be finite and strictly between {-bound} and {bound} m')
    mass=.2
    origin=np.array([mass*(.2**2+.05**2)/12,
                     mass*(.2**2+.05**2)/12,mass*(.2**2+.2**2)/12])
    central=origin-mass*np.array([0.,c*c,c*c])
    if mass_model in ('ballast','edge_ballast'):
        # Uniform body plus a uniform 1 cm cube inside it. The ballast
        # center is c/f, so the complete body's COM is exactly c. Assemble
        # the two central inertias about their common COM by parallel axes.
        f=ballast_fraction;body_fraction=.01 if mass_model=='edge_ballast' else .1
        central=body_fraction*origin+f*mass*(.01**2)/6*np.ones(3)
        central+=mass*(body_fraction*c*c+f*(c/f-c)**2)*np.array([0.,1.,1.])
        return dict(model=f'{100*body_fraction:.0f}% uniform box plus {100*f:.0f}% internal 1 cm cubic ballast',mass_model=mass_model,
                    mass_kg=mass,com_body_m=[c,0.,0.],central_inertia_diagonal_kg_m2=central.tolist(),
                    ballast_mass_kg=f*mass,ballast_center_body_m=[c/f,0.,0.],ballast_size_m=.01)
    fraction=(1+c/.05)/2
    return dict(model='two uniform half-boxes along body x',mass_kg=mass,
                com_body_m=[c,0.,0.],central_inertia_diagonal_kg_m2=central.tolist(),
                negative_half_mass_kg=mass*(1-fraction),positive_half_mass_kg=mass*fraction)


def with_mass_distribution(sdf,box_com_x=0.,mass_model='halves'):
    properties=mass_properties(box_com_x,mass_model)
    if box_com_x==0. and mass_model=='halves':return sdf  # Preserve the existing nominal simulator exactly.
    # The existing SDF uses Drake extension prefixes without XML namespace
    # declarations. Parse only the standard inertial fragment, preserving the
    # contact and visual XML byte for byte.
    match=re.search(r'<inertial>.*?</inertial>',sdf,flags=re.DOTALL)
    if match is None:raise ValueError('Box SDF has no inertial element')
    inertial=ET.fromstring(match.group())
    pose=inertial.find('pose')
    if pose is None:pose=ET.SubElement(inertial,'pose')
    pose.text=f'{box_com_x:.17g} 0 0 0 0 0'
    inertial.find('mass').text=str(properties['mass_kg'])
    for key,value in zip(['ixx','iyy','izz'],properties['central_inertia_diagonal_kg_m2']):
        inertial.find(f'inertia/{key}').text=f'{value:.17g}'
    return sdf[:match.start()]+ET.tostring(inertial,encoding='unicode')+sdf[match.end():]
