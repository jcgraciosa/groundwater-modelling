# %% [markdown]
# from IPython.display import display, Markdown
# 
# equation = r"""
# $$
# h = \left[ \left(H + \frac{L \tan(\alpha)}{2}\right) - \frac{4 \tan(\alpha)}{L} \right] \lim_{{m \to \infty}} \left\{ \frac{\cos \left( \frac{(2m + 1) \pi x}{L} \right) \cosh \left( \frac{(2m + 1) \pi y}{L} \right)}{\left( \frac{(2m + 1) \pi}{L} \right)^2 \cosh \left( \frac{(2m + 1) \pi H}{L} \right)} \right\}
# $$
# where $h$ is the hydraulic head distribution, $K$ is the hydraulic conductivity, $H$ is the elevation of the water table above the datum, and $\alpha$ is the angle of slope of the water table.
# """
# 
# display(Markdown(equation))
# 
# from IPython.display import display, Markdown
# 
# equation = r"""
# $$
# U = -\frac{4K \tan(\alpha)}{L} \lim_{{m \to \infty}} \left\{ \frac{\sin \left( \frac{(2m + 1) \pi x}{L} \right) \cosh \left( \frac{(2m + 1) \pi y}{L} \right)}{\left( \frac{(2m + 1) \pi}{L} \right) \cosh \left( \frac{(2m + 1) \pi H}{L} \right)} \right\}
# $$
# where $U$ is the horizontal Darcy velocity, $K$ is the hydraulic conductivity, $H$ is the elevation of the water table above the datum, and $\alpha$ is the angle of slope of the water table.
# """
# 
# display(Markdown(equation))
# from IPython.display import display, Markdown
# 
# equation = r"""
# $$
# V = \frac{4K \tan(\alpha)}{L} \lim_{{m \to \infty}} \left\{ \frac{\cos \left( \frac{(2m + 1) \pi x}{L} \right) \sinh \left( \frac{(2m + 1) \pi y}{L} \right)}{\left( \frac{(2m + 1) \pi}{L} \right) \cosh \left( \frac{(2m + 1) \pi H}{L} \right)} \right\}
# $$
# where $V$ is the vertical Darcy velocity, $K$ is the hydraulic conductivity, $H$ is the elevation of the water table above the datum, and $\alpha$ is the angle of slope of the water table.
# """
# 
# display(Markdown(equation))

# %%
import underworld3 as uw
import numpy as np
import matplotlib.pyplot as plt
import sympy

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
xv, yv = meshbox.X

v_soln  = uw.discretisation.MeshVariable("U", meshbox, meshbox.dim, degree = Vdeg)
h_soln  = uw.discretisation.MeshVariable("P", meshbox, 1, degree = Hdeg)

h_num  = uw.discretisation.MeshVariable("Pn", meshbox, 1, degree = Hdeg)

u_num  = uw.discretisation.MeshVariable("Un", meshbox, 1, degree = Vdeg)
v_num  = uw.discretisation.MeshVariable("Vn", meshbox, 1, degree = Vdeg)

u_calc = uw.systems.Projection(meshbox, u_num)
u_calc.uw_function = K * sympy.diff(h_num.sym, xv)
u_calc.uw_function

v_calc = uw.systems.Projection(meshbox, v_num)
v_calc.uw_function = K * sympy.diff(h_num.sym, yv)
v_calc.uw_function

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
    ax.scatter(h_soln.coords[:, 0], h_soln.coords[:, 1], c = h_soln.data[:], s = 20)
    ax.set_aspect("equal")
    print(h_soln.data[:].min(), h_soln.data[:].max())


# %%
with meshbox.access(h_soln):
    fig, ax = plt.subplots(dpi = 150)
    ax.scatter(h_num.coords[:, 0], h_num.coords[:, 1], c = h_num.data[:], s = 5)
    ax.set_aspect("equal")
    print(h_num.data[:].min(), h_num.data[:].max())

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

# vertical velocities
v_samp1 = uw.function.evaluate(v_soln.sym[1], xy_samp6)

# horizontal velocities
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
poisson_solver = uw.systems.Poisson(
                                mesh = meshbox,
                                u_Field = h_num,
                                solver_name = "poisson",
                                verbose = True,
                            )
poisson_solver.constitutive_model = uw.constitutive_models.DiffusionModel
poisson_solver.constitutive_model.Parameters.diffusivity = 1

# first argu - sympy.Matrix
# gradT . n - supposed
# switch to dev branch
# poisson_solver.add_natural_bc((0., None), "Left", components = (0,))
# poisson_solver.add_natural_bc((0., None), "Right", components = (0,))
# poisson_solver.add_natural_bc((None, 0.), "Bottom", components = (1,))

xv, yv = meshbox.X
poisson_solver.add_dirichlet_bc(height + xv * sympy.tan(alpha_rad), "Top")

# %%
height + xv * sympy.tan(alpha_rad)

# %%
help(meshbox.Gamma)
# gamma - 

# %%
help(poisson_solver.add_natural_bc)

# %%
poisson_solver.solve()

# %%
with meshbox.access(h_soln, v_soln):
    print(h_soln.data.min(), h_soln.data.max())
    print(v_soln.data[:, 0].min(), v_soln.data[:, 0].max())
    print(v_soln.data[:, 1].min(), v_soln.data[:, 1].max())
    print(h_num.data.min(), h_num.data.max())
    # print(v_num.data[:, 0].min(), v_num.data[:, 0].max())
    # print(v_num.data[:, 1].min(), v_num.data[:, 1].max())

# %%
# poisson_solver.view()

# %% [markdown]
# This class provides functionality for a discrete representation of the Poisson equation
# 
# $$ \nabla \cdot \color{Blue}{\underbrace{\Bigl[ \boldsymbol\kappa \nabla u \Bigr]}_{\mathbf{F}}} = \color{Maroon}{\underbrace{\Bigl[ f \Bigl] }_{\mathbf{f}}} $$
# 
# The term $\mathbf{F}$ relates the flux to gradients in the unknown $u$
# 
# Properties
# The unknown is $u$
# 
# The diffusivity tensor, $\kappa$ is provided by setting the constitutive_model property to one of the scalar uw.constitutive_models classes and populating the parameters. It is usually a constant or a function of position / time and may also be non-linear or anisotropic.
# 
# $f$ is a volumetric source term

# %%
with meshbox.access(h_num):
    fig, ax = plt.subplots(dpi = 150)
    ax.scatter(h_num.coords[:, 0], h_num.coords[:, 1], c = h_num.data[:], s = 20)
    ax.set_aspect("equal")

# %%
u_calc.solve()
v_calc.solve()

# %%
with meshbox.access(u_num):
    fig, ax = plt.subplots(dpi = 100)
    out = ax.scatter(u_num.coords[:, 0], u_num.coords[:, 1], c = u_num.data[:], s = 20)
    ax.set_aspect("equal")
    cbar = fig.colorbar(out)

    fig, ax = plt.subplots(dpi = 100)
    out2 = ax.scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = v_soln.data[:, 0], s = 20)
    ax.set_aspect("equal")
    cbar2 = fig.colorbar(out2)

# %%
with meshbox.access(v_num):
    fig, ax = plt.subplots(dpi = 100)
    out = ax.scatter(v_num.coords[:, 0], v_num.coords[:, 1], c = v_num.data[:], s = 20)
    ax.set_aspect("equal")
    cbar = fig.colorbar(out)
    ax.set_title("Whatever title")

    fig, ax = plt.subplots(dpi = 100)
    out2 = ax.scatter(v_soln.coords[:, 0], v_soln.coords[:, 1], c = v_soln.data[:, 1], s = 20)
    ax.set_aspect("equal")
    cbar2 = fig.colorbar(out2)

# %%



