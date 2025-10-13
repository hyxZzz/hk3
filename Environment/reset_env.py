import numpy as np
import math as m
from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles


def generate_missile_positions(a_x, a_y, a_z, num_missiles):
    positions = []
    semi_major = 20000.0
    semi_minor = 18000.0
    vertical_span = 3000.0
    for _ in range(num_missiles):
        angle = np.random.uniform(0, 2 * m.pi)
        radius_scale = np.random.uniform(0.9, 1.1)
        x_offset = semi_major * radius_scale * m.cos(angle)
        z_offset = semi_minor * radius_scale * m.sin(angle)
        y_offset = np.random.uniform(-vertical_span, vertical_span)
        missile_x = a_x + x_offset
        missile_y = max(0.0, a_z + y_offset)
        missile_z = a_y + z_offset
        positions.append([missile_x, missile_y, missile_z])
    return positions


def reset_para(num_missiles=3, StepNum=1200):
    num_missiles = 3 if num_missiles is None else num_missiles

    # 飞行高度8~12Km
    a_x = np.random.uniform(-10000, 10000)
    a_y = np.random.uniform(-10000, 10000)
    a_z = np.random.uniform(8000, 12000)

    # 飞行速度0.5~1.2马赫
    a_v = np.random.uniform(0.5, 1.2)
    a_v = a_v * 340

    # 飞机的俯仰角和偏转角
    aPitch = 0
    aHeading = np.random.uniform(-1, 1)
    aHeading = aHeading * m.pi

    missile_positions = generate_missile_positions(a_x, a_y, a_z, num_missiles)
    mposList = []
    mHeadingList = []
    mPitchList = []
    for missile_pos in missile_positions:
        mposList.append(missile_pos)
        mHeadingList.append(ComputeHeading([a_x, a_z, a_y], missile_pos))
        mPitchList.append(ComputePitch([a_x, a_z, a_y], missile_pos))

    # 导弹速度不低于2马赫
    m_v = np.random.uniform(2, 3)
    m_v = m_v * 340

    aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
    missiles_list = []
    for missile_pos, missile_pitch, missile_heading in zip(mposList, mPitchList, mHeadingList):
        missiles_list.append(Missiles(missile_pos, V=m_v, Pitch=missile_pitch, Heading=missile_heading))

    return missiles_list, aircraft_agent[0], a_v, num_missiles, StepNum, m_v
