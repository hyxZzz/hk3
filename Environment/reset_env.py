import numpy as np
import math as m
from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles


def reset_para(num_missiles=3, StepNum=1200):
    # 飞机的初始位置，x和y为[-10000, 10000]

    horizontal_radius_major = 20000
    horizontal_radius_minor = 15000
    vertical_span = 3000

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


    # 在飞机周围的水平椭圆区域生成导弹位置
    angles = np.random.uniform(0, 2 * m.pi, num_missiles)
    radial_scale = np.random.uniform(0.7, 1.0, num_missiles)
    xi_r = a_x + horizontal_radius_major * radial_scale * np.cos(angles)
    yi_r = a_y + horizontal_radius_minor * radial_scale * np.sin(angles)
    zi_r = a_z + np.random.uniform(-vertical_span, vertical_span, num_missiles)
    zi_r = np.clip(zi_r, 0, None)
    mposList = []
    mHeadingList = []
    mPitchList = []
    for i in range(len(xi_r)):
        mposList.append([xi_r[i], zi_r[i], yi_r[i]])
        # mHeadingList.append(ComputeHeading([a_x, a_y, a_z], [xi_r[i], yi_r[i], zi_r[i]]))
        # mPitchList.append(ComputePitch([a_x, a_y, a_z], [xi_r[i], yi_r[i], zi_r[i]]))
        mHeadingList.append(ComputeHeading([a_x, a_z, a_y], [xi_r[i], zi_r[i], yi_r[i]]))
        mPitchList.append(ComputePitch([a_x, a_z, a_y], [xi_r[i], zi_r[i], yi_r[i]]))
    # print(mposList)

    # 初始导弹速度在0.7~0.9马赫，加速段提升至目标速度
    m_v = np.random.uniform(0.7, 0.9) * 340

    aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
    missiles_list = []
    for i in range(len(mposList)):
        missiles_list.append(
            Missiles(
                mposList[i],
                V=m_v,
                Pitch=mPitchList[i],
                Heading=mHeadingList[i],
            )
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
    # aircraft_agent = [plane]
    # m_v = 680
    # a_v = 340

    return missiles_list, aircraft_agent[0], a_v, num_missiles, StepNum, m_v

