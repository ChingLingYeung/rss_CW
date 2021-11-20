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

taskId = 3.1

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
    "cameraSettings": (1.07, 90.0, -52.8, (0.07, 0.01, 0.76))
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
    # compile urdfs
    finalTargetPos = np.array([0.7, 0.00, 0.91])
    urdf_compiler_path = core_path + "/urdf_compiler.py"
    subprocess.call([urdf_compiler_path,
                     "-o", abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
                     abs_path+"/lib/task_urdfs/task3_1_target.urdf"])
#TODO: probably need to be some distance from this for xue=-0.4)
    sim.p.resetJointState(bodyUniqueId=1, jointIndex=6, targetValue=-0.4)
    # load the table in front of the robot
    tableId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/table/table_taller.urdf",
        basePosition        = [0.8, 0, 0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi/2]),
        useFixedBase        = True,
        globalScaling       = 1.4
    )
    cubeId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/cubes/cube_small.urdf",
        basePosition        = [0.33, 0, 1.0],
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase        = False,
        globalScaling       = 1.4
    )
    sim.p.resetVisualShapeData(cubeId, -1, rgbaColor=[1, 1, 0, 1])

    targetId = sim.p.loadURDF(
        fileName            = abs_path+"/lib/task_urdfs/task3_1_target_compiled.urdf",
        basePosition        = finalTargetPos,
        baseOrientation     = sim.p.getQuaternionFromEuler([0, 0, math.pi]),
        useFixedBase        = True,
        globalScaling       = 1
    )
    for _ in range(200):
        sim.tick()
        time.sleep(1./1000)

    return tableId, cubeId, targetId


def solution():
    # TODO: Add your code here
    startPos = sim.getJointPosition('LARM_JOINT5')
    cubePos = np.array([0.33, 0, 1.0])
    
    target1 = np.array([0.2, startPos[1], startPos[2]])
    target2 = np.array([0.2, 0, 1])
    target3 = finalTargetPos - [0.13, 0, 0]

    # target1 = np.array([0.22, 0.22, .95])
    # target2 = np.array([0.22, 0.0, .94])
    # target3 = np.array([0.63, 0.0, .94])

    endEffector = 'LHAND'
    sim.move_with_PD(endEffector, target1, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)
    print("target 1")
    time.sleep(1)
    sim.move_with_PD(endEffector, target2, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=500, debug=False, verbose=False)
    print("target 2")
    time.sleep(1)
    sim.move_with_PD(endEffector, target3, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=1000, debug=False, verbose=False)
    print("target 3")
    time.sleep(1)

    # target0 = np.array([startPos[0], startPos[1], 1.3])
    target1 = np.array([0.2, startPos[1], 1])
    target2 = np.array([0.15, 0, 1])
    target3 = finalTargetPos - [0.13, 0, 0.95]

    # sim.move_with_PD(endEffector, target0, speed=0.01, orientation=None, 
        # threshold=1e-3, maxIter=1000, debug=False, verbose=False)
    # sim.move_with_PD(endEffector, target1, speed=0.01, orientation=[0,0,1], 
    #     threshold=1e-3, maxIter=5000, debug=False, verbose=False)
    # sim.move_with_PD(endEffector, target1, speed=0.01, orientation=None, 
    #     threshold=1e-3, maxIter=500, debug=False, verbose=False)
    # sim.move_with_PD(endEffector, target2, speed=0.01, orientation=None, 
    #     threshold=1e-3, maxIter=500, debug=False, verbose=False)
    # input()
    # sim.move_with_PD(endEffector, target2, speed=0.01, orientation=[0,0,1], 
    #     threshold=1e-3, maxIter=2500, debug=False, verbose=False)
    # sim.move_with_PD(endEffector, target3, speed=0.01, orientation=[0,0,1], 
    #     threshold=1e-3, maxIter=5000, debug=False, verbose=False)

tableId, cubeId, targetId = getReadyForTask()
solution()
time.sleep(5)