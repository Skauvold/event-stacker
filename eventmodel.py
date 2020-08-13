import numpy as np
from numpy import linspace
import math
import skimage.transform
from bisect import bisect_left
from scipy.interpolate import griddata, interp2d, RectBivariateSpline, RegularGridInterpolator
from scipy.spatial import Delaunay
from scipy.special import comb
from scipy import optimize
import matplotlib.path as pth

# TODO:
#   - Improve/debug Process.condition_last_event
#   - More robust well trajectory/event boundary intersect detection
#   - Variogram creation for event residuals on Process initialization
#   - Event placement based on sea level
#   - Healing based on global rather than local topographic relief
#   - Track interface width over simulation time and see if it agrees with the theory of Barabási and Stanley


class Grid:
    def __init__(self,
                 nx = 11,
                 ny = 11,
                 xmin=0.0,
                 xmax=1.0,
                 ymin=0.0,
                 ymax=1.0):
        self.nx, self.ny = nx, ny
        self.xmin, self.xmax = xmin, xmax
        self.ymin, self.ymax = ymin, ymax
        self.x_coords = linspace(xmin, xmax, nx)
        self.y_coords = linspace(ymin, ymax, ny)
        self.xrange = xmax - xmin
        self.yrange = ymax - ymin
        if nx > 1:
            self.dx = self.xrange/float(nx-1)
        if ny > 1:
            self.dy = self.yrange/float(ny-1)
        self.xmesh, self.ymesh = np.meshgrid(self.x_coords, self.y_coords)
        self.extent = self.xmin, self.xmax, self.ymin, self.ymax
    
    def ind2sub(self, k):
        assert (k in range(self.nx * self.ny)), "Index out of bounds"
        i = int(k / self.ny)
        j = int(k % self.ny)
        return i, j
    
    def sub2ind(self, i, j):
        assert (i in range(self.nx)) & (j in range(self.ny)), "Subscript indices out of bounds"
        k = int(i*self.ny + j)
        return k
    
    def evaluate(self, f):
        return f(self.xmesh, self.ymesh)
    
    def distance_matrix_xyz(self):
        x_points = self.xmesh.flatten()
        y_points = self.ymesh.flatten()
        
        dx_mat = x_points[..., np.newaxis] - x_points[np.newaxis, ...]
        dy_mat = y_points[..., np.newaxis] - y_points[np.newaxis, ...]
        
        return dx_mat, dy_mat, np.zeros_like(dx_mat)
    
    def distance_matrix_circular(self):
        # Wrap-around boundary condition
        x_points = self.xmesh.flatten()
        dx_mat = x_points[..., np.newaxis] - x_points[np.newaxis, ...]
        
        # Compute smallest angle difference
        dx_mat = np.minimum(abs(dx_mat), 2.0 * np.pi - abs(dx_mat))
        
        return dx_mat, np.zeros_like(dx_mat), np.zeros_like(dx_mat)


class Event:
    def __init__(self,
                 center=(0.0, 0.0),
                 length=2.0,
                 width=1.0,
                 azimuth=30,
                 max_thickness=0.5):
        self.center = center
        self.length = length
        self.width = width
        self.azimuth = math.radians(azimuth)
        self.max_thickness = max_thickness
        self.theta_grid = Grid(101, 1, 0.0, 2.0*math.pi, 0.0, 0.0)
        self.generate_planform()
        self.generate_thickness()
    
    def global2local(self, x_global, y_global):
        x_center, y_center = self.center
        xdiff = x_global - x_center
        ydiff = y_global - y_center
        radius = np.sqrt(xdiff**2 + ydiff**2)
        angle = np.arctan2(ydiff, xdiff) - self.azimuth
        x_local = radius * np.cos(angle)
        y_local = radius * np.sin(angle)
        return x_local, y_local

    def planform_trend(self):
        r0 = 0.5 * self.width
        r1 = 0.5 * (self.length - self.width)
        theta = self.theta_grid.x_coords
        r = r0 + r1 * 0.5 * (1.0 + np.cos(theta))        
        return r, theta
        
    def planform_noise(self):
        variogram_r = Variogram(range=(0.6, 0.5, 0.5), power=2.0, variance=(0.05*self.length)**2)
        # This covariance function has to wrap around
        r_noise = variogram_r.draw(self.theta_grid, circular=True)
        return r_noise.flatten()
        
    def generate_planform(self):
        r_trend, theta = self.planform_trend()
        r_noise = self.planform_noise()
        r_combined = r_trend + r_noise
        self.planform = {"r": r_combined, "theta": theta}
        
    def thickness_trend(self, r_rel):
        return self.max_thickness * (np.ones_like(r_rel) - np.power(r_rel, 2.0))
        
    def thickness_noise(self, x_full, y_full, num_int, num_bdy):
        num_tot = num_int + num_bdy
        variogram_t = Variogram(range=(0.50*self.length, 0.50*self.width, 1.0), power=2.0, variance=(0.2*self.max_thickness)**2)
        self.thickness_variogram = variogram_t
        self.thickness_point_count = {"internal": num_int, "boundary": num_bdy}
        grid_t = Grid(num_tot, 1, 0.0, 1.0, 0.0, 1.0)
        grid_t.xmesh = x_full
        grid_t.ymesh = y_full
        # Form covariance matrix of (T_I, T_B)
        sigma_full = variogram_t.cov_matrix(grid_t)        
        # Split covariance matrix
        sigma_ii = sigma_full[:num_int, :num_int]
        sigma_ib = sigma_full[:num_int, num_int:]
        sigma_bi = sigma_full[num_int:, :num_int]
        sigma_bb = sigma_full[num_int:, num_int:]        
        # Compute covariance matrix of T_I conditional on T_B = 0
        #sigma_igb = sigma_i - sigma_ib * sigma_bb \ sigma_bi
        sigma_bb_inv_sigma_bi = np.linalg.solve(sigma_bb, sigma_bi)
        sigma_igb = sigma_ii - np.matmul(sigma_ib, sigma_bb_inv_sigma_bi)         
        mu_igb = np.zeros(shape=(num_int,))        
        # Draw a realization from conditional distribution   
        t_noise_int = np.random.multivariate_normal(mu_igb, sigma_igb)
        t_noise_bdy = np.zeros(shape=(num_bdy,))
        t_noise_full = list(t_noise_int.flatten()) + list(t_noise_bdy.flatten())        
        return t_noise_full
        
    def generate_thickness(self):
        r_boundary = self.planform["r"]
        theta_boundary = self.planform["theta"]
        x_points, y_points, r_rel_points = self.distribute_points(r_boundary, theta_boundary)
        t_trend = self.thickness_trend(r_rel_points)
        
        # The last len(r_boundary) points are boundary points, the rest are interior points
        num_boundary = len(r_boundary)
        num_internal = len(x_points) - num_boundary
        
        t_noise = self.thickness_noise(x_points, y_points, num_internal, num_boundary)
        t_combined = t_trend + t_noise
        self.thickness_local = {"t": t_combined, "x": x_points, "y": y_points}
        
    def distribute_points(self, r_shape, theta_shape):
        r_list = []
        theta_list = []
        r_max = 1.0
        delta_r = 1.0/5.0
        num_r_levels = int(np.ceil(r_max/delta_r))
        r_levels = np.linspace(0.0, (num_r_levels/(1.0 + num_r_levels)) * r_max, num_r_levels)
        for r in r_levels:
            if r == 0:
                delta_theta = 0.0
                num_theta = 1
            else:
                num_theta = int(np.ceil(2.0 * np.pi * r / delta_r))
                delta_theta = 2.0 * np.pi / num_theta
            for k in range(num_theta):
                r_list.append(r)
                theta_list.append(k * delta_theta)
        r_mapped = []
        for (r, theta) in zip(r_list, theta_list):
            pos_closest = bisect_left(theta_shape, theta)
            r_closest = 0.5*(r_shape[pos_closest-1] + r_shape[pos_closest])
            r_mapped.append(r_closest * r / r_max)            
        for (r, theta) in zip(r_shape[0:-1:2], theta_shape[0:-1:2]):
            r_mapped.append(r)
            theta_list.append(theta)
            r_list.append(r_max)
        #r_mapped.pop()
        #theta_list.pop()
        #r_list.pop()
        x_mapped = r_mapped * np.cos(theta_list)
        y_mapped = r_mapped * np.sin(theta_list)
        return x_mapped, y_mapped, r_list
    
    def thickness(self, x_global, y_global):
        x_q, y_q = self.global2local(x_global, y_global)
        t_ev = self.thickness_local["t"]
        x_ev = self.thickness_local["x"]
        y_ev = self.thickness_local["y"]
        t_q = griddata((x_ev, y_ev), t_ev, (x_q, y_q), method="linear", fill_value=0.0)
        return t_q
    
    def trapezoidal_volume(self, xyz):
        """Calculate volume under a surface defined by irregularly spaced points
        using delaunay triangulation. "x,y,z" is a <numpoints x 3> shaped ndarray."""
        d = Delaunay(xyz[:,:2])
        tri = xyz[d.vertices]
    
        a = tri[:,0,:2] - tri[:,1,:2]
        b = tri[:,0,:2] - tri[:,2,:2]
        proj_area = np.abs(np.cross(a, b))
        zavg = tri[:,:,2].sum(axis=1)
        vol = zavg * np.abs(proj_area) / 6.0
        return vol.sum()

    def volume(self):
        x_ev = self.thickness_local["x"]
        y_ev = self.thickness_local["y"]
        t_ev = self.thickness_local["t"]
        xyt_ev = np.vstack((x_ev, y_ev, t_ev)).T
        return self.trapezoidal_volume(xyt_ev)


class Bathymetry:
    def __init__(self,
                 type="linear",
                 **kwargs):
        # Parse parameters, substitute default values where necessary
        self.azimuth = math.radians(kwargs.get("azimuth", 0.0))
        if "dip_angle" in kwargs:
            self.dip_angle = math.radians(kwargs.get("dip_angle"))
        else:
            self.dip_angle = math.radians(kwargs.get("dip", 30.0))
        self.extent = kwargs.get("extent", (0.0, 1.0, 0.0, 1.0))
        self.decay = kwargs.get("decay", 1.0)
        if "level" in kwargs:
            self.level = kwargs.get("level")
        else:
            self.level = kwargs.get("top", 0.0)
        self.top_level = self.level
        # Create surface function
        if type is "linear":
            self.func = self.linear_bathymetry()
        elif type is "exponential":
            if self.azimuth <= 0.25*math.pi:
                length_x = self.extent[1] - self.extent[0]
                u_max = length_x/math.cos(self.azimuth)
            else:
                length_y = self.extent[3] - self.extent[2]
                u_max = length_y/math.sin(self.azimuth)
            self.scale = self.dip_angle * u_max / (1.0 - math.exp(-1.0 * self.decay * u_max))
            self.func = self.exponential_bathymetry()
        else:
            raise ValueError("""type must be "linear" or "exponential".""")
            
    def linear_bathymetry(self):
        def bathy_func(x, y):
            return self.level - math.tan(self.dip_angle) * (x*np.cos(self.azimuth) + y*np.sin(self.azimuth))
        return bathy_func

    def exponential_bathymetry(self):
        def bathy_func(x, y):
            return (self.level - self.scale) + self.scale * np.exp(-self.decay * (x*np.cos(self.azimuth) + y*np.sin(self.azimuth)))
        return bathy_func

    def addfunc(self, other_bathymetry):
        # Should update attributes too, not just surface
        old_func = self.func
        other_func = other_bathymetry.func
        def new_func(x, y):
            return old_func(x, y) + other_func(x, y)
        self.func = new_func
        self.top_level = self.top_level + other_bathymetry.top_level

    def cut_top(self, cutoff=0.0):
        old_func = self.func
        def new_func(x, y):
            old_val = old_func(x, y)
            return np.minimum(cutoff * np.ones_like(old_val), old_val)
        self.func = new_func
        self.top_level = cutoff

    def set_top_level(self, new_top_level):
        old_func = self.func
        old_top_level = self.top_level
        difference = new_top_level - old_top_level
        def new_func(x, y):
            return old_func(x, y) + difference
        self.func = new_func
        self.top_level = new_top_level
            
    def rescale(self, scale_factor):
        old_func= self.func
        def new_func(x, y):
            return scale_factor * old_func(x, y)
        self.func = new_func
        self.top_level = scale_factor * self.top_level


class Variogram:
    def __init__(self,
                 **kwargs):
        self.azimuth = math.radians(kwargs.get("azimuth", 0.0))
        if "dip_angle" in kwargs:
            self.dip_angle = math.radians(kwargs.get("dip_angle"))
        else:
            self.dip_angle = math.radians(kwargs.get("dip", 0.0))
        self.range = kwargs.get("range", (1.0, 1.0, 1.0))
        self.power = kwargs.get("power", 1.5)
        self.variance = kwargs.get("variance", 1.0) # Constant variance
    
    def distance_pair(self, loc_1, loc_2):
        # Assume loc_1 and loc_2 are triples
        dx = loc_2[0] - loc_1[0]
        dy = loc_2[1] - loc_1[1]
        dz = loc_2[2] - loc_1[2]
            
        cosphi = math.cos(self.dip_angle)
        sinphi = math.sin(self.dip_angle)
        costheta = math.cos(self.azimuth)
        sintheta = math.sin(self.azimuth)
        
        # Rotation
        dx_rot =  cosphi*costheta*dx - cosphi*sintheta*dy + sinphi*dz
        dy_rot =         sintheta*dx +        costheta*dy
        dz_rot = -sinphi*costheta*dx + sinphi*sintheta*dy + cosphi*dz
            
        range_x, range_y, range_z = self.range
    
        return math.sqrt((dx_rot/range_x)**2 + (dy_rot/range_y)**2 + (dz_rot/range_z)**2)
    
    def corr_pair(self, loc_1, loc_2):
        return math.exp(-math.pow(self.distance_pair(loc_1, loc_2), self.power))

    def cov_pair(self, loc_1, loc_2):
        return self.variance * self.corr_pair(loc_1, loc_2)

    def corr_matrix(self, grid, circular=False):
        if circular:
            dx_mat, dy_mat, dz_mat = grid.distance_matrix_circular()
        else:
            dx_mat, dy_mat, dz_mat = grid.distance_matrix_xyz()
        
        cosphi = math.cos(self.dip_angle)
        sinphi = math.sin(self.dip_angle)
        costheta = math.cos(self.azimuth)
        sintheta = math.sin(self.azimuth)
        
        dx_mat_rot =  cosphi*costheta*dx_mat - cosphi*sintheta*dy_mat + sinphi*dz_mat
        dy_mat_rot =         sintheta*dx_mat +        costheta*dy_mat
        dz_mat_rot = -sinphi*costheta*dx_mat + sinphi*sintheta*dy_mat + cosphi*dz_mat
        
        range_x, range_y, range_z = self.range
        dist_mat = np.sqrt((dx_mat_rot/range_x)**2 + (dy_mat_rot/range_y)**2 + (dz_mat_rot/range_z)**2)
        
        return np.exp(-np.power(dist_mat, self.power))
                
    def cov_matrix(self, grid, circular=False):
        return self.variance * self.corr_matrix(grid, circular)
    
    def draw(self, grid, circular=False):
        cov_mat = self.cov_matrix(grid, circular)
        mean_vec = np.zeros(shape=cov_mat.shape[0])
        rf = np.random.multivariate_normal(mean_vec, cov_mat)
        return np.reshape(rf, newshape=(grid.ny, grid.nx))


class Process:
    def __init__(self,
                 **kwargs):
        self.grid = kwargs.get("grid")
        m_cf = math.ceil(0.5*math.log(self.grid.nx * self.grid.ny, 2.0) - 6.0)
        coarsening_factor = int(math.pow(2.0, m_cf))
        print("Using coarsening factor {0}.".format(coarsening_factor))
        nx_coarse = int(math.ceil(self.grid.nx / coarsening_factor))
        ny_coarse = int(math.ceil(self.grid.ny / coarsening_factor))
        xmin, xmax = self.grid.xmin, self.grid.xmax
        ymin, ymax = self.grid.ymin, self.grid.ymax
        self.coarse_grid = Grid(nx_coarse, ny_coarse, xmin, xmax, ymin, ymax)
        self.initial_bathymetry = kwargs.get("bathymetry")
        initial_surface_variogram = kwargs.get("init_surf_variogram")
        self.base_surface = self.grid.evaluate(self.initial_bathymetry.func)
        base_surf_noise_coarse = initial_surface_variogram.draw(self.coarse_grid)
        base_surf_noise_fine = skimage.transform.resize(base_surf_noise_coarse, self.base_surface.shape)
        self.base_surface += base_surf_noise_fine
        self.top_surface = self.base_surface.copy()
        self.previous_top_surface = self.base_surface.copy()
        self.events = []
        self.compositional_exponent = kwargs.get("lam", 1.0)
        self.healing_exponent = kwargs.get("gam", 1.2)
        self.sea_level = kwargs.get("sea_level", 0.0)

    def draw_placement(self):
        # When $f \propto \exp(-\lambda \log \Delta z)$, then $\lambda$
        # corresponds to the compositional exponent $m$ in Jo and Pyrcz (2019)
        lam = self.compositional_exponent
        x_grid = self.grid.xmesh
        y_grid = self.grid.ymesh
        x_list = x_grid.flatten()
        y_list = y_grid.flatten()
        z_grid = self.top_surface
        z_grid = z_grid + (0.001 - np.amin(z_grid)) * np.ones_like(z_grid)                        
        sl_grid = self.sea_level * np.ones_like(z_grid)
        f_grid = np.where(z_grid > sl_grid, np.exp(-lam * np.log(z_grid)), 0.0)
        f_list = f_grid.flatten()
        f_sums_list = np.cumsum(f_list)
        u = np.random.uniform(low=0.0, high=f_sums_list[-1])
        k_u = bisect_left(f_sums_list, u)
        x_u = x_list[k_u]
        y_u = y_list[k_u]
        # Consider perturbing these to avoid dependence on grid
        return x_u, y_u
    
    def generate_next_event(self):
        pos_x, pos_y = self.draw_placement()
        length = np.random.normal(1000.0, 80.0)
        width = np.random.normal(700.0, 60.0)
        angle = np.random.normal(0.0, 22.5)
        thickness = np.random.normal(1.75, 0.2)
        ev = Event((pos_x, pos_y), length, width, angle, thickness)
        self.events.append(ev)
        self.previous_top_surface = self.top_surface.copy()
        self.top_surface += self.grid.evaluate(ev.thickness)
        
    def slice_stack(self, start_coords, end_coords, num_points):
        x_start, y_start = start_coords
        x_end, y_end = end_coords
        x_slice = linspace(x_start, x_end, num_points)
        y_slice = linspace(y_start, y_end, num_points)
        d_x_slice = x_slice - x_start * np.ones_like(x_slice)
        d_y_slice = y_slice - y_start * np.ones_like(y_slice)
        d_slice = np.sqrt(d_x_slice**2 + d_y_slice**2)
        x_axis = self.grid.x_coords
        y_axis = self.grid.y_coords
        base_interp = RegularGridInterpolator((x_axis, y_axis), self.base_surface.transpose())
        base_slice = base_interp((x_slice, y_slice))
        slice_curves = []
        slice_curves.append(base_slice)
        current_top = base_slice.copy()
        for ev in self.events:
            current_top += ev.thickness(x_slice, y_slice)
            slice_curves.append(current_top.copy())      
        return x_slice, y_slice, d_slice, slice_curves
    
    def heal_last_event(self, vc=True):
        gam = self.healing_exponent
        ev = self.events[-1]
        volume_initial = ev.volume()
        x_ev = ev.thickness_local["x"]
        y_ev = ev.thickness_local["y"]
        t_ev = ev.thickness_local["t"]        
        x_axis = self.grid.x_coords
        y_axis = self.grid.y_coords        
        bot_interp = RegularGridInterpolator((x_axis, y_axis),
                                             self.previous_top_surface.transpose(),
                                             bounds_error=False,
                                             fill_value=None)
        ev_bot = bot_interp((x_ev, y_ev))
        ev_bot_max, ev_bot_min = max(ev_bot) * np.ones_like(ev_bot), min(ev_bot) * np.ones_like(ev_bot)
        f_heal = np.ones_like(ev_bot) - np.power((ev_bot - ev_bot_min) / (ev_bot_max - ev_bot_min), gam)
        #grid_bot_max, grid_bot_min = self.previous_top_surface.max(), self.previous_top_surface.min()
        #f_heal = np.ones_like(ev_bot) - np.power((ev_bot - grid_bot_min) / (ev_bot_max - grid_bot_min), gam)
        ev.thickness_local["t"] *= f_heal
        if vc:
            volume_final = ev.volume()
            ev.thickness_local["t"] *= (volume_initial / volume_final) * np.ones_like(ev_bot)
        self.top_surface = self.previous_top_surface + self.grid.evaluate(ev.thickness)
     
    def set_well_picks(self, well):
        t_picks = well.find_intersections(self)
        x_picks, y_picks, z_picks = well.trajectory(t_picks)
        self.well_picks = np.vstack((x_picks, y_picks, z_picks))
            
    def save_data(self, filename):
        # Save base surface and well picks to a file
        if not hasattr(self, "well_picks"):
            print("Process has no well data to save.")
            return -1
        np.savez(filename, base_surface=self.base_surface, well_picks=self.well_picks)

    def load_data(self, filename):
        # Load base surface and well picks from file and overwrite existing attributes
        container = np.load(filename)
        self.base_surface = container["base_surface"]
        self.well_picks = container["well_picks"]
        if len(self.events) == 0:
            self.top_surface = self.base_surface.copy()
            self.previous_top_surface = self.base_surface.copy()
            self.num_assimilated = 0
    
    def remove_last_event(self):
        self.top_surface = self.previous_top_surface.copy()
        self.previous_top_surface -= self.grid.evaluate(self.events[-1].thickness)
        self.events.pop()
    
    def condition_last_event(self):
        # Condition the latest event to the next well observation
        if not hasattr(pr, "well_picks"):
            print("Process has no well data to assimilate.")
            return -1
        
        # Get well observations (current and previous picks)
        previous_index = self.num_assimilated
        current_index = previous_index + 1
        
        current_pick = self.well_picks[:,current_index]
        previous_pick = self.well_picks[:,previous_index]
        
        # Get the latest event
        if len(self.events) == 0:
            print("Process has no events to condition.")
            return -1
        
        ev = self.events[-1]
        
        # Get planform polygon of event
        r_pf = ev.planform["r"]
        theta_pf = ev.planform["theta"]
        x_c, y_c = ev.center
        x_pf = x_c + r_pf * np.cos(theta_pf)
        y_pf = y_c + r_pf * np.sin(theta_pf)
        planform_polygon = [[x, y] for (x, y) in zip(x_pf, y_pf)]
        planform_polygon_path = pth.Path(planform_polygon)
        
        # Check if (x, y)-coordinates of well picks lie in polygon
        picks_xy = np.vstack((current_pick[0:2], previous_pick[0:2]))
        
        inside = planform_polygon_path.contains_points(picks_xy)

        if sum(inside) == 0:
            # If both picks are outside, keep the event (do nothing) and return
            #print("2 outside")
            return 
        elif sum(inside) == 1:
            # If exactly one pick is inside, delete the event and return
            # Remember to also change current and previous top surfaces
            self.remove_last_event()
            #print("1 outside")
            return
        
        # If both picks are inside, proceed        
        x_ib = ev.thickness_local["x"]
        y_ib = ev.thickness_local["y"]
        t_ib = ev.thickness_local["t"]
        
        x_pick_global, y_pick_global, z_pick = current_pick
        x_pick, y_pick = ev.global2local(x_pick_global, y_pick_global)
        
        # Current thickness at observed location
        t_o = griddata((x_ib, y_ib), t_ib, (x_pick, y_pick), method="linear", fill_value=0.0)
        
        # Event base elevation at observed location
        f_bot = RectBivariateSpline(self.grid.x_coords, self.grid.y_coords, self.previous_top_surface.T)
        z_bot = f_bot(x_pick_global, y_pick_global)
        z_bot = z_bot[0,0]
        
        # Target thickness at observed location
        # Observed residual = current pick z 
        # minus previous top surface at current pick (x, y)
        t_pick = z_pick - z_bot
        
        thickness_change = (t_pick - t_o) / t_o
        
        if thickness_change < -0.5 or thickness_change > 0.5:
            #print("rejected")
            self.remove_last_event()
            return
        
        x_ibo = np.append(x_ib, x_pick)        
        y_ibo = np.append(y_ib, y_pick)
        
        num_int = ev.thickness_point_count["internal"]
        num_bdy = ev.thickness_point_count["boundary"]
        num_tot = num_int + num_bdy + 1
        grid_t = Grid(num_tot, 1, 0.0, 1.0, 0.0, 1.0)
        grid_t.xmesh = x_ibo
        grid_t.ymesh = y_ibo
        
        # Form covariance matrix of (interior, boundary, observed)
        sigma_full = ev.thickness_variogram.cov_matrix(grid_t)
        
        # Split covariance matrix
        sigma_i = sigma_full[:num_int, :num_int]
        sigma_i_bo = sigma_full[:num_int, num_int:]
        sigma_bo = sigma_full[num_int:, num_int:]
        
        t_i_old = t_ib[:num_int]
        t_b = t_ib[num_int:]
        
        # Fifference between observed and predicted thickness
        delta_bo = np.append(np.zeros_like(t_b), t_pick - t_o)

        # Solve for updated thickness at interior points
        sigma_bo_inv_delta_bo = np.linalg.solve(sigma_bo, delta_bo)
        t_i_new = t_i_old + np.matmul(sigma_i_bo, sigma_bo_inv_delta_bo)
        
        t_i_new = np.where(t_i_new >= 0.0, t_i_new, 0.0)
        
        # Evaluate the residual thickness pdf at the fitted residual
        # (use t_i_new - t_i_old as a conventient substitute)
#        t_i_mean = np.mean(t_i_new) * np.ones_like(t_i_new)
#        t_i_dev = t_i_new - t_i_mean
#        sigma_i_inv_t_i_dev = np.linalg.solve(sigma_i, t_i_dev)
#        log_density = -0.5 * np.matmul(t_i_dev.T, sigma_i_inv_t_i_dev)

        volume_initial = ev.volume()
        max_thickness_initial = ev.max_thickness
        t_ib_new = t_ib
        t_ib_new[:num_int] = t_i_new
        
        ev.thickness_local["t"] = np.append(t_ib_new, t_pick)
        ev.thickness_local["x"] = x_ibo
        ev.thickness_local["y"] = y_ibo
        
        volume_final = ev.volume()
        
        volume_change = (volume_final - volume_initial) / volume_initial
        
        # Accept/reject based on ratio of realized density to modal density
        
        if volume_change < -0.5 or volume_change > 0.5:
            #print("rejected")
            self.remove_last_event()
            return
        
        new_max_thickness = ev.thickness_local["t"].max()
        old_max_thickness = max_thickness_initial
        max_thickness_change = (new_max_thickness - old_max_thickness)/old_max_thickness
        
        if max_thickness_change < -0.5 or max_thickness_change > 0.5:
            #print("rejected")
            self.remove_last_event()
            return
        
        ev.max_thickness = new_max_thickness
        self.top_surface = self.previous_top_surface + self.grid.evaluate(ev.thickness)
        self.num_assimilated += 1
        
        print("accepted, {} picks assimilated".format(self.num_assimilated))
        print("rel. thickness change: {}".format(thickness_change))
        print("max thickness change: {}".format(max_thickness_change))
        print("rel. volume change: {}".format(volume_change))
        
        return
    
        
class Well:
    def __init__(self,
                 xc, yc, zc,
                 **kwargs):
        self.n_t = kwargs.get("n_t", 1001)
        self.t = np.linspace(0.0, 1.0, self.n_t)    
        self.control_points = [[x, y, z] for (x, y, z) in zip(xc, yc, zc)]

    def bezier_curve(self, points, t):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.
    
           points should be a list of lists, or list of tuples
           such as [ [1,0,1], 
                     [2,0,3], 
                     [4,0,5], ..[Xn, Yn, Zn] ]
            nTimes is the number of time steps, defaults to 1000
    
            See http://processingjs.nihongoresources.com/bezierinfo/
        """
    
        def bernstein_poly(i, n, t):
            """
             The Bernstein polynomial of n, i as a function of t
            """
            
            return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
    
        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])
        zPoints = np.array([p[2] for p in points])
    
        polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(nPoints-1, -1, -1)])
    
        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)
        zvals = np.dot(zPoints, polynomial_array)
    
        return xvals, yvals, zvals
    
    def trajectory(self, t):
        return self.bezier_curve(self.control_points, t)
    
    def find_intersections(self, proc):
        x_grid = proc.grid.x_coords
        y_grid = proc.grid.y_coords
        g = self.trajectory
        
        def intersection_solve(f, t_l, t_u):
            def delta(t):
                x, y, z_c = g(t)
                z_s = f(x, y)
                return z_s - z_c
            return optimize.bisect(delta, t_l, t_u)
        
        t_intersect = []
        epsilon = 1.0 / self.n_t
        current_surface = proc.base_surface.copy()
        f_current = RectBivariateSpline(x_grid, y_grid, current_surface.T)        
        t_intersect.append(intersection_solve(f_current, epsilon, 1.0 - epsilon))        
        
        for ev in proc.events:
            current_surface += proc.grid.evaluate(ev.thickness)
            f_current = RectBivariateSpline(x_grid, y_grid, current_surface.T)
            t_previous = t_intersect[-1]
            t_current = intersection_solve(f_current, epsilon, t_previous + epsilon)
            if abs(t_current - t_previous) > epsilon:
                t_intersect.append(t_current)
            
        return np.array(t_intersect)
