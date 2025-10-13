from Environment.env import ManeuverEnv
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


# 环境的来袭导弹数num_missiles，最大步数StepNum

def init_env(num_missiles=3, StepNum=1000, interceptor_num=8):
    # 飞机的初始位置，x和y为[-10000, 10000], z为服从2000为均值，300为标准差的正态分布

    num_missiles = 3 if num_missiles is None else num_missiles

    # 飞行高度8~12Km
    a_x = np.random.uniform(-10000, 10000)
    a_y = np.random.uniform(-10000, 10000)
    a_z = np.random.uniform(8000, 12000)

    # print(a_x, a_y, a_z)

    # 飞行速度0.5~1.2马赫
    a_v = np.random.uniform(0.5, 1.2)
    a_v = a_v * 340

    # 飞机的俯仰角和偏转角
    aPitch = 0
    aHeading = np.random.uniform(-1, 1)
    # aHeading = np.random.uniform(0, 2)
    aHeading = aHeading * m.pi

    missile_positions = generate_missile_positions(a_x, a_y, a_z, num_missiles)
    mposList = []
    mHeadingList = []
    mPitchList = []
    for missile_pos in missile_positions:
        mposList.append(missile_pos)
        # mHeadingList.append(ComputeHeading([a_x, a_y, a_z], [xi_r[i], yi_r[i], zi_r[i]]))
        # mPitchList.append(ComputePitch([a_x, a_y, a_z], [xi_r[i], yi_r[i], zi_r[i]]))
        mHeadingList.append(ComputeHeading([a_x, a_z, a_y], missile_pos))
        mPitchList.append(ComputePitch([a_x, a_z, a_y], missile_pos))
    # print(mposList)

    # 导弹速度不低于2马赫
    m_v = np.random.uniform(2, 3)
    m_v = m_v * 340

    aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
    missiles_list = []
    for i in range(len(mposList)):
        missiles_list.append(Missiles(mposList[i], V=m_v, Pitch=mPitchList[i], Heading=mHeadingList[i]))

    # 初始化环境
    env = ManeuverEnv(
        missiles_list,
        aircraft_agent[0],
        planeSpeed=a_v,
        missilesNum=num_missiles,
        spaceSize=StepNum,
        InterceptorNum=interceptor_num,
    )

    # """
    #     一致性测试
    # """
    #
    # aPos = [2000, 1000, 2000]
    # # 测试aircraft类
    # plane = Aircraft(aPos, 340, 0, 180 * m.pi / 180, )
    # m1Pos = [1000, 2000, 1000]
    # m2Pos = [1000, 2000, 500]
    # heading1 = ComputeHeading(aPos, m1Pos)
    # ms = Missiles(m1Pos, 680, ComputePitch(aPos, m1Pos), heading1)
    # ms2 = Missiles(m2Pos, 680, ComputePitch(aPos, m2Pos), ComputeHeading(aPos, m2Pos))
    # missiles_list = [ms, ms2]
    # aircraftList = plane
    # env = ManeuverEnv(missiles_list, aircraftList, missilesNum=2)

    return env, aircraft_agent, missiles_list
# init_env(2)
