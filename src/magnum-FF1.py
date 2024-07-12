# %%
import underworld3 as uw
import numpy as np
import matplotlib.pyplot as plt

# %%
height = 500. 
length = 1000
alpha = 1.7184 # in degrees
K = 1

resolution = 64

alpha_rad = np.deg2rad(alpha)

qdeg = 3
Vdeg = 2
Hdeg = 1

# %%
# function for calculating the hydraulic head 
def calculate_h(x, y, length, height, alpha_in_rad, n_terms = 100):

    sub_expr1 = height + 0.5 * length * np.tan(alpha_in_rad)

    val = 0
    for m in range(n_terms):
        eta_m = (2 * m + 1) * np.pi / length

        # just one of many ways to do this
        val1 = np.cos (eta_m * x) / (eta_m**2)
        val2 = np.cosh (eta_m * y) / np.cosh(eta_m * height)

        # if np.abs(val1) < 1e-10: val1 = 0
        # if np.abs(val2) < 1e-10: val2 = 0
        
        val += val1 * val2 
        #print(val1, val2)

    out = sub_expr1 - (4 * np.tanh(alpha) / length) * val
    
    return out

# %%
# function for calculating the hydraulic head 
def calculate_vel(x, y, length, height, alpha_in_rad, K, n_terms = 100):

    amp = 4 * K * np.tan(alpha_in_rad) / length

    u_dummy = 0
    v_dummy = 0
    for m in range(n_terms):
        eta_m = (2 * m + 1) * np.pi / length

        # just one of many ways to do this
        dummy1 = np.sin(eta_m * x) * np.cosh(eta_m * y) / (eta_m * np.cosh(eta_m * height))
        dummy2 = np.cos(eta_m * x) * np.sinh(eta_m * y) / (eta_m * np.cosh(eta_m * height))

        # # to prevent overflow
        # if np.abs(dummy1) < 1e-10: dummy1 = 0
        # if np.abs(dummy2) < 1e-10: dummy2 = 0
        
        u_dummy += dummy1
        v_dummy += dummy2

    u = -amp * u_dummy
    v = amp * v_dummy

    return u, v

# %%
xmin, xmax = 0, length
ymin, ymax = 0, height

# %%
meshbox = uw.meshing.UnstructuredSimplexBox(
                                                minCoords= (xmin, ymin),
                                                maxCoords= (xmax, ymax),
                                                cellSize= length / resolution,
                                                regular=False,
                                                qdegree = qdeg
                                        )

# %%
v_soln  = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
h_soln  = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Hdeg)

# %%
with meshbox.access(h_soln, v_soln):
    for (i, (x, y)) in enumerate(h_soln.coords):
        #print(x, y)
        h_soln.data[i] = calculate_h(x = x, y = y, length = length, height = height, alpha_in_rad = alpha_rad, n_terms = 100)
    
    for (i, (x, y)) in enumerate(v_soln.coords):
        v_soln.data[i, :] = calculate_vel(x = x, y = y, length = length, height = height, alpha_in_rad = alpha_rad, K = K, n_terms = 100)

    print(h_soln.data.min(), h_soln.data.max())
    print(v_soln.data[:, 0].min(), v_soln.data[:, 0].max())
    print(v_soln.data[:, 1].min(), v_soln.data[:, 1].max())



# %%
with meshbox.access(h_soln):
    fig, ax = plt.subplots(dpi = 150)
    ax.scatter(h_soln.coords[:, 0], h_soln.coords[:, 1], c = h_soln.data[:], s = 5)
    ax.set_aspect("equal")

# %%
# hydraulic head 
xy_samp1 = np.zeros([100, 2])
xy_samp1[:, 0] = np.linspace(0, length, 100)
xy_samp1[:, 1] = 250

xy_samp2 = np.zeros([100, 2])
xy_samp2[:, 0] = 50
xy_samp2[:, 1] = np.linspace(0, height, 100)

xy_samp3 = np.zeros([100, 2])
xy_samp3[:, 0] = 500
xy_samp3[:, 1] = np.linspace(0, height, 100)

xy_samp4 = np.zeros([100, 2])
xy_samp4[:, 0] = 950
xy_samp4[:, 1] = np.linspace(0, height, 100)

xy_samp5 = np.zeros([100, 2])
xy_samp5[:, 0] = np.linspace(0, length, 100)
xy_samp5[:, 1] = height

# velocities sampling points
# horizontal velocity 
xy_samp6 = np.zeros([100, 2])
xy_samp6[:, 0] = np.linspace(0, length, 100)
xy_samp6[:, 1] = 285

# vertical velocity
xy_samp7 = np.zeros([100, 2])
xy_samp7[:, 0] = 25
xy_samp7[:, 1] = np.linspace(0, height, 100)

xy_samp8 = np.zeros([100, 2])
xy_samp8[:, 0] = 545
xy_samp8[:, 1] = np.linspace(0, height, 100)

xy_samp9 = np.zeros([100, 2])
xy_samp9[:, 0] = 975
xy_samp9[:, 1] = np.linspace(0, height, 100)

# %%
h_samp1 = uw.function.evaluate(h_soln.sym, xy_samp1)
h_samp2 = uw.function.evaluate(h_soln.sym, xy_samp2)
h_samp3 = uw.function.evaluate(h_soln.sym, xy_samp3)
h_samp4 = uw.function.evaluate(h_soln.sym, xy_samp4)
h_samp5 = uw.function.evaluate(h_soln.sym, xy_samp5)

# horizontal velocities
v_samp1 = uw.function.evaluate(v_soln.sym[1], xy_samp6)

# vertical velocities
u_samp1 = uw.function.evaluate(v_soln.sym[0], xy_samp7)
u_samp2 = uw.function.evaluate(v_soln.sym[0], xy_samp8)
u_samp3 = uw.function.evaluate(v_soln.sym[0], xy_samp9)

# %%
fig, ax = plt.subplots(dpi = 100)
ax.plot(xy_samp1[:, 0], h_samp1)

# %%
fig, ax = plt.subplots(dpi = 150)
ax.plot(h_samp2, xy_samp2[:, 1], label = "x = 50")
ax.plot(h_samp3, xy_samp3[:, 1], label = "x = 500")
ax.plot(h_samp4, xy_samp4[:, 1], label = "x = 950")
ax.set_aspect("equal")
ax.legend()

# %%
fig, ax = plt.subplots(dpi = 150)
ax.plot(xy_samp5[:, 0], h_samp5)

# %%
# horizontal velocity
fig, ax = plt.subplots(dpi = 100)
ax.plot(xy_samp1[:, 0], v_samp1)

# %%
fig, ax = plt.subplots(dpi = 150)
ax.plot(u_samp1, xy_samp7[:, 1], label = "x = 25")
ax.plot(u_samp2, xy_samp8[:, 1], label = "x = 545")
ax.plot(u_samp3, xy_samp9[:, 1], label = "x = 975")
ax.set_xlim([-0.05, 0])
ax.set_ylim([0, 500])
ax.legend()

# %%



