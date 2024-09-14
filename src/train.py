import deepxde as dde
from models.pinns_model import ETModel


et_model = ETModel()
geom = dde.geometry.Interval(0.0, et_model.D_depth)
timedomain = dde.geometry.TimeDomain(0.0, 2 * et_model.t_lambda)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)

# Define Boundary conditions
initial_condition = dde.icbc.IC(geomtime, lambda x: 1.0, et_model.initial)
neumann_bc = dde.icbc.NeumannBC(geomtime, lambda x: et_model.neumann_boundary_condition_cq(x), et_model.boundary_right)
robin_bc = dde.icbc.RobinBC(geomtime, et_model.robin_boundary_condition, et_model.boundary_left)

# Define the PDE problem and configurations of the network:
data = dde.data.TimePDE(
    geomtime,
    et_model.pde,
    [initial_condition, neumann_bc, robin_bc],
    num_domain=20000,
    num_boundary=1000,
    num_initial=1000,
    num_test=2000
)


layer_size = [2] + [7] * 3 + [5] * 3 + [1]
activation = 'tanh'
initializer = 'Glorot uniform'
net = dde.nn.FNN(layer_size, activation, initializer)

loss_weights = [1, 1, 1, 1e-5]

# Build the model:
model = dde.Model(data, net)
model.compile("adam", lr=1e-3, loss_weights=loss_weights)

checker = dde.callbacks.ModelCheckpoint(
    "Results/saved_models/model_cq.ckpt", save_better_only=True, period=10000, verbose=1
)

# Train Save the model
losshistory, train_state = model.train(iterations=100000, callbacks=[checker])

# save training the results
dde.saveplot(losshistory,
             train_state,
             output_dir = "Results/saved_models/model_cq_training_files",
             issave=True,
             isplot=True)
