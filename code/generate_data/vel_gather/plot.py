# -*- coding: utf-8 -*-
"""
@Time ： 2022/3/11 9:24 AM
@Auth ： YanJinXiang
@File ：plot.py
@IDE ：PyCharm
"""

import numpy as np
import matplotlib.pyplot as plt

# # plot velocity  色卡不带标注
# vel = np.load("./velocity/fault/velocity_00000000.npy")  # 加载速度
# plt.imshow(vel.T)
# plt.xlabel("Distance (m)")    # 横轴名称
# plt.ylabel("Depth (m)")       # 纵轴名称
# plt.title("Velocity")         # 图名称
# # plt.xticks([])                # 去除x坐标
# # plt.yticks([])                # 去除y坐标
# plt.colorbar()       # 色卡
# plt.savefig("plot_vel.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)
# plt.savefig("plot_vel.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
# plt.show()           # 展示

# ## plot gather  色卡不带标注
# f = plt.figure(figsize=(15, 5))
# VLIM = 0.6
# gather = np.load("./gather/fault_2ms_r/gather_00000000_00000000.npy")  #加载gather
# plt.imshow(gather.T,aspect=0.12, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
# # plt.xticks([])                # 去除x坐标
# # plt.yticks([])                # 去除y坐标
# plt.colorbar()       # 色卡
# plt.savefig("plot_gather.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)
# plt.savefig("plot_gather.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
# plt.show()


# ## plot velocity  色卡带标注
# vel = np.load("./velocity/fault/velocity_00000000.npy")
# f = plt.figure(figsize=(6, 5))
# im0 = plt.imshow(vel.T)
# plt.xlabel("Distance (m)")    # 横轴名称
# plt.ylabel("Depth (m)")       # 纵轴名称
# plt.title("Velocity")         # 图名称
# # plt.xticks([])                # 去除x坐标
# # plt.yticks([])                # 去除y坐标
# ax = f.add_axes([0.85, 0.68, 0.01, 0.20])  #设置坐标
# cb = f.colorbar(im0, cax=ax)               #colorbar
# cb.ax.set_ylabel('Velocity ($\mathrm{ms}^{-1}$)')     #colorbar标注
# plt.savefig("plot_vel.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)   #保存
# plt.savefig("plot_vel.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
# plt.show()

## plot gather  色卡不带标注
VLIM = 0.6
gather = np.load("./gather/fault_2ms_r/gather_00000000_00000000.npy")
f = plt.figure(figsize=(6, 5))
im1 = plt.imshow(gather.T,aspect=0.12, cmap="gray_r", vmin=-VLIM, vmax=VLIM)
# # plt.xticks([])                # 去除x坐标
# # plt.yticks([])                # 去除y坐标
ax = f.add_axes([0.70, 0.68, 0.01, 0.20])
cb = f.colorbar(im1, cax=ax, aspect=0.01)
cb.ax.set_ylabel('Pressure (arb)')
plt.savefig("plot_gather.pdf", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.savefig("plot_gather.png", bbox_inches='tight', pad_inches=0.01, dpi=300)
plt.show()