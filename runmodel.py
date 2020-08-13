import matplotlib.pyplot as plt
from eventmodel import *

# A script for testing the functionality of the event stacker program

# TODO:
#   - Make this a notebook

# Define model grid
g = Grid(801, 401, 0.0, 2000.0, 0.0, 1000.0)

# Create an initial bathymetry trend surface
bathy = Bathymetry(type="exponential", level=0.0, azimuth=0.0, dip=1.0, decay=0.005, extent=g.extent)
#bathy.addfunc(Bathymetry(type="exponential", level=0.0, azimuth=45.0, dip=0.5, decay=0.005, extent=g.extent))
bathy.addfunc(Bathymetry(type="exponential", level=0.0, azimuth=90.0, dip=1.0, decay=0.005, extent=g.extent))
bathy.cut_top(cutoff=-37.5)
bathy.set_top_level(0.0)
bathy.rescale(2.0)
bathy.set_top_level(10.0)

# Create a variogram for the initial bathymetry perturbation
vario = Variogram(azimuth=0.0, dip=0.0, variance=0.75**2, range=(300.0, 300.0, 5.0), power=2.0)

# Create process (standard choice is gam=2.20, lam=50.0)
pr = Process(grid=g, bathymetry=bathy, init_surf_variogram=vario, gam=2.2, lam=50.0, sea_level=-16.0)

#for event_index in range(50):
#    pr.generate_next_event()
#    pr.heal_last_event(vc=False)
    
# Load data from earlier run
pr.load_data("process_data_1.npz")

max_num_iterations = 800
num_iterations = 0
num_picks = 9

while num_iterations < max_num_iterations and pr.num_assimilated < num_picks:
    pr.generate_next_event()
    pr.heal_last_event(vc=False)
    pr.condition_last_event()
    num_iterations += 1
    if np.mod(num_iterations, 10) == 0:
        print("Iteration {}/{}".format(num_iterations, max_num_iterations))

print("Generated {} events matching {} picks.".format(len(pr.events), pr.num_assimilated))

well_1 = Well([400.0, 400.0, 1800.0], [380.0, 380.0, 870.0], [20.0, -18.0, -20.0])

pr.set_well_picks(well_1)

# Save data
#pr.save_data("process_data_1")

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(g.xmesh, g.ymesh, pr.base_surface, alpha=0.5, label="Base")
ax.plot_surface(g.xmesh, g.ymesh, pr.top_surface, alpha=0.5, label="Top")
ax.plot(pr.well_picks[0, :], pr.well_picks[1, :], pr.well_picks[2, :], "k.")
ax.plot(xw, yw, zw, 'r-')
ax.set_aspect("equal")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.zaxis.label.set_size(14)
ax.view_init(elev=20.0, azim=-10.0)
#ax.xaxis.tick_labels.set_size(16)

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(g.xmesh, g.ymesh, pr.top_surface - pr.base_surface, alpha=0.6)
ax.set_aspect("equal")
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.xaxis.label.set_size(14)
ax.yaxis.label.set_size(14)
ax.zaxis.label.set_size(14)
ax.view_init(elev=20.0, azim=-10.0)


well_1 = Well([400.0, 400.0, 1800.0], [380.0, 380.0, 870.0], [20.0, -18.0, -20.0])
xw, yw, zw = well_1.trajectory(well_1.t)
dw = np.sqrt((xw-xw[0])**2 + (yw-yw[0])**2)

t_intersect = well_1.find_intersections(pr)
x_intersect, y_intersect, z_intersect = well_1.trajectory(t_intersect)
d_intersect = np.sqrt((x_intersect-xw[0])**2 + (y_intersect-yw[0])**2)

x_slice1, y_slice1, d_slice1, curves1 = pr.slice_stack((400.0, 380.0), (1800.0, 870.0), 101)

x_picks = pr.well_picks[0,:]
y_picks = pr.well_picks[1,:]
z_picks = pr.well_picks[2,:]
d_picks = np.sqrt((x_picks - xw[0])**2 + (y_picks - yw[0])**2)

fig = plt.figure(figsize=(8,4))
for c in curves1:
    plt.plot(d_slice1, c, "k-")
plt.plot(dw, zw, "r-")
plt.plot(d_intersect, z_intersect, "ko", markersize=8)
plt.plot(d_picks, z_picks, "ko", markersize=8)
plt.show()
plt.ylim((-20.0, -12.0))
#plt.axis("off")
#plt.title("comp. exp.: {}, healing exp.: {}, w/vol. ctrl.".format(pr.compositional_exponent, pr.healing_exponent), fontsize=18)

x_slice2, y_slice2, d_slice2, curves2 = pr.slice_stack((250.0, 950.0), (1950.0, 250.0), 101)

fig = plt.figure(figsize=(8,4))
for c in curves2:
    plt.plot(d_slice2, c, "k-")
plt.show()
plt.ylim((-20.0, -12.0))
