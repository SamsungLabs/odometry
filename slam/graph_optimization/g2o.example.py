import numpy as np
import g2o


solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
solver = g2o.OptimizationAlgorithmLevenberg(solver)

optimizer = g2o.SparseOptimizer()
optimizer.set_verbose(True)
optimizer.set_algorithm(solver)

for i in range(3):
    pose = g2o.Isometry3d()
    pose.set_translation([0, 0, i])
    q = g2o.Quaternion(np.array([1, 0, 0, 0]))
    pose.set_rotation(q)
    vertex = g2o.VertexSE3()
    vertex.set_estimate(pose)
    vertex.set_id(i)
    optimizer.add_vertex(vertex)

for i in range(3):
    measurement = g2o.Isometry3d()
    measurement.set_translation([0, 0, i])
    q = g2o.Quaternion(np.array([1, 0, 0, 0]))
    measurement.set_rotation(q)
    edge = g2o.EdgeSE3()
    edge.set_measurement(measurement)
    edge.set_information(np.eye(6, 6))
    edge.set_vertex(0, optimizer.vertex(i))
    edge.set_vertex(1, optimizer.vertex((i + 1) % 3))
    optimizer.add_edge(edge)

optimizer.initialize_optimization()
optimizer.optimize(10)
optimizer.save('result.txt')

