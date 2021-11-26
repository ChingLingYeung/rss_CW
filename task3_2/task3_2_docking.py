import subprocess, math, time, sys, os, numpy as np
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data
import time

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path + '/..'
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation import Simulation

# specific settings for this task

taskId = 3.2

try:
    if sys.argv[1] == 'nogui':
        gui = False
    else:
        gui = True
except:
    gui = True

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": gui,
    "panels": False,
    "realTime": False,
    "controlFrequency": 1000,
    "updateFrequency": 250,
    "gravity": -9.81,
    "gravityCompensation": 1.,
    "floor": True,
    "cameraSettings": (1.2, 90, -22.8, (-0.12, -0.01, 0.99))
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains.yaml",
    "robotStartPos": [0, 0, 0.85],
    "robotStartOrientation": [0, 0, 0, 1],
    "fixedBase": True,
    "colored": False
}

sim = Simulation(pybulletConfigs, robotConfigs)

##### Please leave this function unchanged, feel free to modify others #####
def getReadyForTask():
    global finalTargetPos
    global taleId, cubeId, targetId, obstacle
    finalTargetPos = np.array([0.35,0.38,1.0])
    # compile target urdf
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_2_target.urdf"])

    sim.p.resetJointState(bodyUniqueId=1, jointIndex=12, targetValue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)

    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition          = [0.8, 0, 0],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/2]),                                  
        useFixedBase          = True,             
        globalScaling         = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_dumb_bell.urdf", 
        basePosition          = [0.5, 0, 1.1],            
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,0]),                                  
        useFixedBase          = False,             
        globalScaling         = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1,1,0,1])
    
    targetId = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/task3_2_target_compiled.urdf",
        basePosition          = finalTargetPos,             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )
    obstacle = sim.p.loadURDF(
        fileName              = abs_path+"/lib/task_urdfs/cubes/task3_2_obstacle.urdf",
        basePosition          = [0.43,0.275,0.9],             
        baseOrientation       = sim.p.getQuaternionFromEuler([0,0,math.pi/4]), 
        useFixedBase          = True,             
        globalScaling         = 1
    )

    for _ in range(300):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():
    # TODO: Add your code here
    leftStartPos = sim.getJointPosition('LARM_JOINT5')
    rightStartPos = sim.getJointPosition('RARM_JOINT5')
    cubePos = np.array([0.5, 0, 1.1])

    ltarget0 = np.array([cubePos[0], cubePos[1] + 0.2, cubePos[2]-0.02])
    rtarget0 = np.array([cubePos[0]-0.2, cubePos[1] - 0.2, cubePos[2]-0.02])

    ltarget1 = np.array([cubePos[0], cubePos[1] + 0.025, cubePos[2]-0.02])
    rtarget1 = np.array([cubePos[0], cubePos[1] - 0.025, cubePos[2]-0.02])
    ltarget2 = np.array([leftStartPos[0], leftStartPos[1] - 0.2, leftStartPos[2] + 0.3])
    rtarget2 = np.array([rightStartPos[0], rightStartPos[1] + 0.2, rightStartPos[2] + 0.3])
    ltarget3 = finalTargetPos - [-0.1, -0.1, 0]
    rtarget3 = finalTargetPos - [0.1, 0.1, 0]

    leftEndEffector = 'LHAND'
    rightEndEffector = 'RHAND'

    sim.move_with_PD(leftEndEffector, ltarget0, speed=0.01, orientation=[0,-1,0], 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)
    sim.move_with_PD(rightEndEffector, rtarget1, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)
    sim.move_with_PD(leftEndEffector, ltarget1, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)
    sim.move_with_PD(rightEndEffector, rtarget1, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)

    for i in range(10):
        ltarget2 = np.array([cubePos[0], cubePos[1] + 0.025, cubePos[2] + (0.02 * i)])
        rtarget2 = np.array([cubePos[0], cubePos[1] - 0.025, cubePos[2] + (0.02 * i)])
        sim.move_with_PD(leftEndEffector, ltarget2, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=50, debug=False, verbose=False)
        sim.move_with_PD(rightEndEffector, rtarget2, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=50, debug=False, verbose=False)

    for i in range(10):
        ltarget3 = np.array([finalTargetPos[0], finalTargetPos[1], cubePos[2] + 0.2]) - np.multiply([0.000, -0.002, 0], 2*i)
        rtarget3 = np.array([finalTargetPos[0], finalTargetPos[1], cubePos[2] + 0.2]) - np.multiply([-0.002, 0.001, 0], 2*i)
        sim.move_with_PD(leftEndEffector, ltarget3, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=30, debug=False, verbose=False)
        sim.move_with_PD(rightEndEffector, rtarget3, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=30, debug=False, verbose=False)

    l3 = np.array(sim.getJointPosition("LARM_JOINT5"))
    r3 = np.array(sim.getJointPosition("RARM_JOINT5"))

    for i in range(10):
        ltarget4 = np.array([l3[0], l3[1], finalTargetPos[2] + (0.005 * (10 - i))])
        rtarget4 = np.array([r3[0]+0.05, r3[1]+0.05, finalTargetPos[2] + (0.005 * (10 - i))])
        
        sim.move_with_PD(rightEndEffector, rtarget4, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=50, debug=False, verbose=False)
        sim.move_with_PD(leftEndEffector, ltarget4, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=50, debug=False, verbose=False)

    ltarget4 = np.array([0, ltarget4[1], ltarget4[2]])
    sim.move_with_PD(rightEndEffector, rtarget4, speed=0.01, orientation=None, 
            threshold=1e-3, maxIter=5000, debug=False, verbose=False)
        
            
tableId, cubeId, targetId = getReadyForTask()
solution()
for _ in range(10000):
        sim.tick()
        time.sleep(1./1000)
finalCubePos, finalCubeOr = sim.p.getBasePositionAndOrientation(cubeId)
distance = np.linalg.norm(finalTargetPos - finalCubePos)*1000
print(distance)