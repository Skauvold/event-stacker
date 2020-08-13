# -*- coding: utf-8 -*-
"""
Created on Wed May  8 15:04:49 2019

@author: jas
"""
import numpy as np
from facies import ftrend, psand

x_range = np.linspace(0.0, 1.0, 101)
y_range = np.linspace(0.0, 1.0, 101)
x_mesh, y_mesh = np.meshgrid(x_range, y_range)
mesh_corners = (x_range.min(), x_range.max(), y_range.min(), y_range.max())

ft_ms = psand(x_mesh, y_mesh)

# Plot function values and contours
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

#colormap = cm.Wistia_r
raw_colormap = cm.YlOrBr_r
colormap = truncate_colormap(raw_colormap, 0.3, 0.8)

plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rc("font", size=16)

cnt_list_small = [0.10, 0.50, 0.90, 0.98]
cnt_list_big = [k for k in np.arange(0.05, 0.95, 0.10)]

plt.figure()
img_handle = plt.imshow(ft_ms, extent=mesh_corners, origin="lower", cmap=colormap, vmin=0.0, vmax=1.0)
cnt_handle = plt.contour(x_mesh, y_mesh, ft_ms, cnt_list_small, colors="black")
plt.clabel(cnt_handle, fmt="%1.2f")
plt.xlabel(r"$\rho$")
plt.ylabel(r"$\tau$")
cba = plt.colorbar(img_handle, orientation="vertical", pad=0.1)
cba.ax.set_ylabel(r"$p_{\mathrm{sand}}$")
plt.show()

#%% Access contours

from scipy.interpolate import interp1d

plt.figure()

upper_right = np.array([[0.0, 1.0], [1.0, 1.0], [1.0, 0.0]])
cnt_combo = np.vstack((cnt_handle.allsegs[0][0], upper_right))    
plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=colormap(0.0))

for k in range(len(cnt_handle.allsegs)-1):
    cnt_combo = np.vstack((cnt_handle.allsegs[k][0], np.flipud(cnt_handle.allsegs[k+1][0])))    
    plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=colormap(cnt_handle.labelLevelList[k]))

lower_left = np.array([[0.0, 0.0]])
cnt_combo = np.vstack((cnt_handle.allsegs[-1][0], lower_left))    
plt.fill(cnt_combo[:,0], cnt_combo[:,1], facecolor=colormap(1.0))

plt.xlabel(r"$\rho$")
plt.ylabel(r"$\tau$")

plt.show()

# Set up interpolating functions
rho2tau = []

for cnt_segs in cnt_handle.allsegs:
    f_cnt = interp1d(cnt_segs[0][:,0], cnt_segs[0][:,1], bounds_error=False, fill_value=0.0)
    rho2tau.append(f_cnt)

#%%

import eventmodel

# Set up and run process
# Compute and draw intra-event facies trend

# Define model grid
g = eventmodel.Grid(801, 401, 0.0, 2000.0, 0.0, 1000.0)

# Create an initial bathymetry trend surface
bathy = eventmodel.Bathymetry(type="exponential", level=0.0, azimuth=0.0, dip=1.0, decay=0.0025, extent=g.extent)
#bathy.cut_top(cutoff=-37.5)
#bathy.set_top_level(0.0)
#bathy.rescale(2.0)
#bathy.set_top_level(10.0)

# Create a variogram for the initial bathymetry perturbation
vario = eventmodel.Variogram(azimuth=0.0, dip=0.0, variance=0.1**2, range=(300.0, 300.0, 5.0), power=2.0)

# Create process (standard choice is gam=2.20, lam=50.0)
pr = eventmodel.Process(grid=g, bathymetry=bathy, init_surf_variogram=vario, gam=2.2, lam=50.0, sea_level=-16.0)

for event_index in range(50):
    pr.generate_next_event()
    
    
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

#%% Plot multiple sections in 3D

fig = plt.figure()
ax = Axes3D(fig)



verts = []

verts.append([[(0, 0, 0), (1, 0, 0), (1, 0, 0.33), (0, 0, 0.67)]])

verts.append([[(0, 0, 0.67), (0, 1, 0.4), (0, 1, 0), (0, 0, 0)]])
    
verts.append([[(0, 0, 1), (1, 0, 1), (1, 0, 0.33), (0, 0, 0.67)]])

verts.append([[(0, 0, 0.67), (0, 1, 0.4), (0, 1, 1), (0, 0, 1)]])
    
face_color_list = [[1.0, 0.4, 0.0], [1.0, 0.67, 0.0]]

face_color_inds = [0, 1, 1, 0]

for vrt, ck in zip(verts, face_color_inds):
    collection = Poly3DCollection(vrt)
    collection.set_facecolor(face_color_list[ck])
    collection.set_edgecolor([0.0, 0.0, 0.0])
    ax.add_collection3d(collection)

plt.show()

#%% Test: find intersection between polygon and line

from shapely.geometry import Polygon, LineString
#
#line = [(pr.grid.xmin, pr.grid.ymin), (pr.grid.xmax, pr.grid.ymax)]
#shapely_line = LineString(line)
#
#plt.figure()
#
#num_touched = 0
#num_untouched = 0
#
#for ev in pr.events:
#    footprint = ev.planform
#    x_footprint = ev.center[0] + footprint["r"] * np.cos(footprint["theta"] + ev.azimuth)
#    y_footprint = ev.center[1] + footprint["r"] * np.sin(footprint["theta"] + ev.azimuth)
#    shapely_poly = Polygon(list(zip(x_footprint, y_footprint)))
#    if shapely_poly.intersects(shapely_line):
#        num_touched += 1
#        plt.plot(x_footprint, y_footprint, "-", color="gray")
#        intersection_coords = list(shapely_poly.intersection(shapely_line).coords)
#        p1 = intersection_coords[0]
#        p2 = intersection_coords[1]
#        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], "k.-", markersize=15, linewidth=2)
#    else:
#        num_untouched += 1
#        plt.plot(x_footprint, y_footprint, "-", color="lightgray")
#
#plt.title("{}/{} events intersected\n".format(num_touched, len(pr.events)))
#plt.show()


#%%

import matplotlib.cm as cm

sec_start = (pr.grid.xmin + 0.33 * pr.grid.xrange, pr.grid.ymin + 0.75 * pr.grid.yrange)
sec_end = (pr.grid.xmin + 1.0 * pr.grid.xrange, pr.grid.ymin + 0.75 * pr.grid.yrange)
x_s, y_s, d_s, z_s = pr.slice_stack(sec_start, sec_end, 201)
shapely_line = LineString([sec_start, sec_end])

plt.figure()

for ind, ev in enumerate(pr.events):
    footprint = ev.planform
    x_footprint = ev.center[0] + footprint["r"] * np.cos(footprint["theta"] + ev.azimuth)
    y_footprint = ev.center[1] + footprint["r"] * np.sin(footprint["theta"] + ev.azimuth)
    shapely_poly = Polygon(list(zip(x_footprint, y_footprint)))
    if shapely_poly.intersects(shapely_line):
        intersection_coords = list(shapely_poly.intersection(shapely_line).coords)
    else:
        continue
    
    p1 = intersection_coords[0]
    p2 = intersection_coords[1]
    d1 = np.sqrt((p1[0] - sec_start[0])**2 + (p1[1] - sec_start[1])**2)
    d2 = np.sqrt((p2[0] - sec_start[0])**2 + (p2[1] - sec_start[1])**2)
    l_event = np.minimum(d1, d2)
    r_event = np.maximum(d1, d2)
    
    # Identify origin (left endpoint of event in section)
    if d1 < d2:
        p_l = p1
        p_r = p2
    else:
        p_l = p2
        p_r = p1
    
    f_B = interp1d(d_s, z_s[ind])
    f_T = interp1d(d_s, z_s[ind+1])
    
    dd_event = np.linspace(l_event, r_event, 101)
    
    plt.plot(dd_event, f_B(dd_event), color="black", linewidth=0.8)
    plt.plot(dd_event, f_T(dd_event), color="black", linewidth=0.8)
    
    # Get information needed to compute contours
    xx_event = np.linspace(p_l[0], p_r[0], 101)
    yy_event = np.linspace(p_l[1], p_r[1], 101)
    rr_event = np.sqrt((xx_event - ev.center[0])**2 + (yy_event - ev.center[1])**2)
    theta_event = np.arctan2(yy_event - ev.center[1], xx_event - ev.center[0])
    theta_event = (theta_event + 2.0 * np.pi) % (2.0 * np.pi)
    theta2R_interp = interp1d(footprint["theta"], footprint["r"])
    RR_event = theta2R_interp(theta_event)
    rho_event = np.minimum(rr_event / RR_event, 1.0)
    
    # Fill patch between upper contour and event top
    tau_cnt = rho2tau[0](rho_event)
    
    bot_event = f_B(dd_event)
    top_event = f_T(dd_event)
    thick_event = top_event - bot_event
    plt.fill_between(dd_event, top_event, bot_event + tau_cnt * thick_event,
                     facecolor=colormap(0.0))
    
    # Fill patches between contour pairs
    for k in range(len(cnt_handle.labelLevelList)-1):
        tau_low = rho2tau[k](rho_event)
        tau_high = rho2tau[k+1](rho_event)
        plt.fill_between(dd_event, bot_event + tau_low * thick_event,
                         bot_event + tau_high * thick_event,
                         facecolor=colormap(cnt_handle.labelLevelList[k]))
    
    # Fill patch between lower contour and event base
    tau_cnt = rho2tau[-1](rho_event)
    plt.fill_between(dd_event, bot_event, bot_event + tau_cnt * thick_event,
                     facecolor=colormap(1.0))

plt.plot(d_s, z_s[0], "k-", linewidth=1.6)
plt.plot(d_s, z_s[-1], "k-", linewidth=1.6)
plt.axis("off")
plt.show()

#import time
#import os.path
#
#if os.path.exists("facies_coloring.png"):
#    savefilename = "facies_coloring_{}.png".format(int(time.time()))
#else:
#    savefilename = "facies_coloring.png"
#
#plt.savefig(savefilename, dpi=300, transparent=True, pad_inches=0.1)

#%% Define function to plot section in 3D

def plot_section_3d(process, sec_start, sec_end, ax):
    x_s, y_s, d_s, z_s = pr.slice_stack(sec_start, sec_end, 201)
    shapely_line = LineString([sec_start, sec_end])
        
    for ind, ev in enumerate(process.events):
        footprint = ev.planform
        x_footprint = ev.center[0] + footprint["r"] * np.cos(footprint["theta"] + ev.azimuth)
        y_footprint = ev.center[1] + footprint["r"] * np.sin(footprint["theta"] + ev.azimuth)
        shapely_poly = Polygon(list(zip(x_footprint, y_footprint)))
        if shapely_poly.intersects(shapely_line):
            intersection_coords = list(shapely_poly.intersection(shapely_line).coords)
        else:
            continue
        
        p1 = intersection_coords[0]
        p2 = intersection_coords[1]
        d1 = np.sqrt((p1[0] - sec_start[0])**2 + (p1[1] - sec_start[1])**2)
        d2 = np.sqrt((p2[0] - sec_start[0])**2 + (p2[1] - sec_start[1])**2)
        l_event = np.minimum(d1, d2)
        r_event = np.maximum(d1, d2)
        
        # Identify origin (left endpoint of event in section)
        if d1 < d2:
            p_l = p1
            p_r = p2
        else:
            p_l = p2
            p_r = p1
        
        f_B = interp1d(d_s, z_s[ind])
        f_T = interp1d(d_s, z_s[ind+1])
        
        dd_event = np.linspace(l_event, r_event, 101)
                
        # Get information needed to compute contours
        xx_event = np.linspace(p_l[0], p_r[0], 101)
        yy_event = np.linspace(p_l[1], p_r[1], 101)
        rr_event = np.sqrt((xx_event - ev.center[0])**2 + (yy_event - ev.center[1])**2)
        theta_event = np.arctan2(yy_event - ev.center[1], xx_event - ev.center[0])
        theta_event = (theta_event + 2.0 * np.pi) % (2.0 * np.pi)
        theta2R_interp = interp1d(footprint["theta"], footprint["r"])
        RR_event = theta2R_interp(theta_event)
        rho_event = np.minimum(rr_event / RR_event, 1.0)
        
        # Plot event outline
        ax.plot(xx_event, yy_event, f_B(dd_event), color="black", linewidth=2.0)
        ax.plot(xx_event, yy_event, f_T(dd_event), color="black", linewidth=2.0)
        
        # Fill patch between upper contour and event top
        tau_cnt = rho2tau[0](rho_event)
        
        bot_event = f_B(dd_event)
        top_event = f_T(dd_event)
        thick_event = top_event - bot_event
        
        xx_event_combined = np.hstack((xx_event, np.flip(xx_event)))
        yy_event_combined = np.hstack((yy_event, np.flip(yy_event)))
        
        cnt_edge = bot_event + tau_cnt * thick_event
        zz_event_combined = np.hstack((top_event, np.flip(cnt_edge)))        
        event_collection = Poly3DCollection([list(zip(xx_event_combined,
                                                yy_event_combined,
                                                zz_event_combined))])
        event_collection.set_facecolor(colormap(0.0))
        ax.add_collection3d(event_collection)
        
        # Fill patches between contour pairs
        for k in range(len(cnt_handle.labelLevelList)-1):
            tau_low = rho2tau[k](rho_event)
            tau_high = rho2tau[k+1](rho_event)
            lower_edge = bot_event + tau_low * thick_event
            upper_edge = bot_event + tau_high * thick_event
            zz_event_combined = np.hstack((lower_edge, np.flip(upper_edge)))
            event_collection = Poly3DCollection([list(zip(xx_event_combined,
                                                    yy_event_combined,
                                                    zz_event_combined))])
            event_collection.set_facecolor(colormap(cnt_handle.labelLevelList[k]))
            ax.add_collection3d(event_collection)
        
        # Fill patch between lower contour and event base
        tau_cnt = rho2tau[-1](rho_event)
        
        cnt_edge = bot_event + tau_cnt * thick_event
        
        zz_event_combined = np.hstack((bot_event, np.flip(cnt_edge)))
        event_collection = Poly3DCollection([list(zip(xx_event_combined,
                                                yy_event_combined,
                                                zz_event_combined))])
        event_collection.set_facecolor(colormap(1.0))
        ax.add_collection3d(event_collection)
            

#%% Call 3D plotting function
from scipy.interpolate import griddata
fig = plt.figure()
ax = Axes3D(fig)

sec_start_a = (pr.grid.xmin + 0.45 * pr.grid.xrange, pr.grid.ymin + 0.33 * pr.grid.yrange)
sec_end_a = (pr.grid.xmin + 0.75 * pr.grid.xrange, pr.grid.ymin + 0.33 * pr.grid.yrange)
plot_section_3d(pr, sec_start_a, sec_end_a, ax)

sec_start_b = (pr.grid.xmin + 0.75 * pr.grid.xrange, pr.grid.ymin + 0.33 * pr.grid.yrange)
sec_end_b = (pr.grid.xmin + 0.75 * pr.grid.xrange, pr.grid.ymin + 0.85 * pr.grid.yrange)
plot_section_3d(pr, sec_start_b, sec_end_b, ax)

# Plot top surface
xx_rect = np.linspace(sec_start_a[0], sec_end_a[0], 101)
yy_rect = np.linspace(sec_start_b[1], sec_end_b[1], 101)
x_mesh_rect, y_mesh_rect = np.meshgrid(xx_rect, yy_rect)
z_mesh_rect = griddata((pr.grid.xmesh.flatten(), pr.grid.ymesh.flatten()), pr.top_surface.flatten(), xi=(x_mesh_rect, y_mesh_rect))

ax.plot_surface(x_mesh_rect, y_mesh_rect, z_mesh_rect, color=colormap(0.0), shade=True)

# Plot section edges

plt.axis("off")
plt.show()

import time
import os.path

if os.path.exists("facies_coloring_3d.png"):
    savefilename = "facies_coloring_3d_{}.png".format(int(time.time()))
else:
    savefilename = "facies_coloring_3d.png"

plt.savefig(savefilename, dpi=300, transparent=True, pad_inches=0.1)