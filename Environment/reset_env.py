import numpy as np
import math as m
from utils.common import ComputeHeading, ComputePitch
from flat_models.trajectory import Aircraft, Missiles


def sample_spherical(npoints, ndim=3):
    vec = np.random.randn(ndim, npoints)
    vec /= np.linalg.norm(vec, axis=0)
    return vec


def reset_para(num_missiles=3, StepNum=1200):
    # 飞机的初始位置，x和y为[-10000, 10000], z为服从2000为均值，300为标准差的正态分布

    # 雷达探测范围10km

    r = 10000

    r_h = 5000
    # 飞行高度0.5~15Km
    a_x = np.random.uniform(-10000, 10000)
    a_y = np.random.uniform(-10000, 10000)
    a_z = np.random.uniform(550, 15000)

    # print(a_x, a_y, a_z)

    # 飞行速度0.5~1.2马赫
    a_v = np.random.uniform(0.5, 1.2)
    a_v = a_v * 340

    # 飞机的俯仰角和偏转角
    aPitch = 0
    aHeading = np.random.uniform(-1, 1)
    # aHeading = np.random.uniform(0, 2)
    aHeading = aHeading * m.pi


    # 以a_x, a_y, a_z为球心, 均匀生成导弹位置
    xi, yi, zi = sample_spherical(num_missiles)  # 单位球上取点

    # 导弹的发射高度不能为负数
    zi = zi
    # for i in range(len(xi)):
    #     if zi[i] < 0:
    #         zi[i] = 0

    # print(xi, yi, zi)
    # 扩张半径与平移坐标

    xi_r = xi * r + a_x
    yi_r = yi * r + a_y
    zi_r = zi * r_h + a_z  # 导弹的竖直高度不能太高，在飞机的10km范围内
    for i in range(len(zi_r)):
        if zi_r[i] < 0:
            zi_r[i] = 0
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

    # 导弹速度不低于2马赫
    m_v = np.random.uniform(2, 3)
    m_v = m_v * 340

    aircraft_agent = [Aircraft([a_x, a_z, a_y], V=a_v, Pitch=aPitch, Heading=aHeading)]
    missiles_list = []
    for i in range(len(mposList)):
        missiles_list.append(Missiles(mposList[i], V=m_v, Pitch=mPitchList[i], Heading=mHeadingList[i]))


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

