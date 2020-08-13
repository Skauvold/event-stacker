# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:06:14 2019

@author: jas
"""
import numpy as np

# Define facies trend function in terms of relative horizontal and vertical position
def ftrend(rho, tau, cpts, coefs):
    if np.any(rho > 1.0) or np.any(rho < 0.0) or np.any(tau > 1.0) or np.any(tau < 0.0):
        return np.nan
    else:
        rhoc, tauc = cpts
        a1, a2, a3 = coefs
        c = - a1 * tauc - a2 * np.power(tauc, 2.0) - a3 * np.power(tauc, 3.0)
        b1 = a1 * tauc / rhoc
        b2 = a2 * np.power(tauc / rhoc, 2.0)
        b3 = a3 * np.power(tauc / rhoc, 3.0)
        ftau_xy = a1 * tau + a2 * np.power(tau, 2.0) + a3 * np.power(tau, 3.0)
        frho_xy = b1 * rho + b2 * np.power(rho, 2.0) + b3 * np.power(rho, 3.0)
        return c + frho_xy + ftau_xy

def psand(rho, tau):
    critical_points = (0.8, 0.8)
    coefficients = (0.0, 5.0, 3.0)
    ft = ftrend(rho, tau, critical_points, coefficients)
    eft = np.exp(ft)
    return 1 / (1 + eft)

#x_range = np.linspace(0.0, 1.0, 101)
#y_range = np.linspace(0.0, 1.0, 101)
#x_mesh, y_mesh = np.meshgrid(x_range, y_range)
#mesh_corners = (x_range.min(), x_range.max(), y_range.min(), y_range.max())
#
#ft_ms = psand(x_mesh, y_mesh)
#
## Plot function values and contours
#import matplotlib.pyplot as plt
#import matplotlib.cm as cm
#plt.rc("text", usetex=True)
#plt.rc("font", family="serif")
#plt.rc("font", size=16)
#
#cnt_list_small = [0.10, 0.50, 0.90, 0.98]
#cnt_list_big = [k for k in np.arange(0.05, 0.95, 0.10)]
#
#plt.figure()
#img_handle = plt.imshow(ft_ms, extent=mesh_corners, origin="lower", cmap=cm.Wistia_r, vmin=0.0, vmax=1.0)
#cnt_handle = plt.contour(x_mesh, y_mesh, ft_ms, cnt_list_big, colors="black")
#plt.clabel(cnt_handle, fmt="%1.2f")
#plt.xlabel(r"$\rho$")
#plt.ylabel(r"$\tau$")
#cba = plt.colorbar(img_handle, orientation="vertical", pad=0.1)
#cba.ax.set_ylabel(r"$p_{\mathrm{sand}}$")
#plt.show()
#
##%% Access contours
#
#plt.figure()
#
#upper_right = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
#cnt_combo = np.vstack((cnt_handle.allsegs[0][0], upper_right))    
#plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=cm.Wistia_r(0.0))
#
#for k in range(len(cnt_handle.allsegs)-1):
#    cnt_combo = np.vstack((cnt_handle.allsegs[k][0], np.flipud(cnt_handle.allsegs[k+1][0])))    
#    plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=cm.Wistia_r(cnt_handle.labelLevelList[k]))
#
#lower_left = np.array([[0.0, 0.0]])
#cnt_combo = np.vstack((cnt_handle.allsegs[-1][0], lower_left))    
#plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=cm.Wistia_r(1.0))
#
#plt.xlabel(r"$\rho$")
#plt.ylabel(r"$\tau$")
#
#plt.show()
#
##%% Map contours to event geometry
#
#def rho2t(rho):
#    return 1.0 * np.sqrt(1.0 - rho**2)
#
#plt.figure(figsize=(6,3))
#
#top_edge = np.vstack((np.linspace(0.0, 1.0, 101), np.ones(101))).T
#cnt_combo = np.vstack((cnt_handle.allsegs[0][0], top_edge))
#rho = cnt_combo[:,0]
#tau = cnt_combo[:,1]
#plt.fill(rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(0.0))
#plt.fill(-rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(0.0))
#
#for k in range(len(cnt_handle.allsegs)-1):
#    cnt_combo = np.vstack((cnt_handle.allsegs[k][0], np.flipud(cnt_handle.allsegs[k+1][0])))    
#    rho = cnt_combo[:,0]
#    tau = cnt_combo[:,1]
#    plt.fill(rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(cnt_handle.labelLevelList[k]))
#    plt.fill(-rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(cnt_handle.labelLevelList[k]))
#
#bottom_edge = np.vstack((np.linspace(0.0, 1.0, 101), np.zeros(101))).T
#cnt_combo = np.vstack((cnt_handle.allsegs[-1][0], bottom_edge))    
#rho = cnt_combo[:,0]
#tau = cnt_combo[:,1]
#plt.fill(rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(1.0))
#plt.fill(-rho, rho2t(rho) * tau, facecolor=cm.Wistia_r(1.0))
#
#plt.xlim([-1.0, 1.0])
#plt.ylim([0.0, 1.0])
#
#plt.axis("off")
#
#plt.show()
#
##%% facies_contour_llokup implementation
#import matplotlib._cntr as cntr
#
## Get contours without plotting:
## https://stackoverflow.com/a/55420612
## Then interpolate directly from rho to tau
#
#def get_single_contour(c=0.0):
#    cnt = cntr.Cntr(x_mesh, y_mesh, ft_ms)
#    nlist = c.trace(level, level, c)
#    segs = nlist[:len(nlist)//2]
#    return segs[0][0]
    