import subprocess
import math
import time
import sys
import os
import numpy as np
import pybullet as bullet_simulation
import pybullet_data

# setup paths and load the core
abs_path = os.path.dirname(os.path.realpath(__file__))
root_path = abs_path
core_path = root_path + '/core'
sys.path.append(core_path)
from Pybullet_Simulation_template import Simulation_template

pybulletConfigs = {
    "simulation": bullet_simulation,
    "pybullet_extra_data": pybullet_data,
    "gui": True,   # Ture | False
    "panels": False,  # Ture | False
    "realTime": False,  # Ture | False
    "controlFrequency": 1000,   # Recommand 1000 Hz
    "updateFrequency": 250,    # Recommand 250 Hz
    "gravity": -9.81,  # Gravity constant
    "gravityCompensation": 1.,     # Float, 0.0 to 1.0 inclusive
    "floor": True,   # Ture | False
    "cameraSettings": 'cameraPreset1'  # cameraPreset{1..3},
}
robotConfigs = {
    "robotPath": core_path + "/nextagea_description/urdf/NextageaOpen.urdf",
    "robotPIDConfigs": core_path + "/PD_gains_template.yaml",
    "robotStartPos": [0, 0, 0.85],  # (x, y, z)
    "robotStartOrientation": [0, 0, 0, 1],  # (x, y, z, w)
    "fixedBase": False,        # Ture | False
    "colored": True          # Ture | False
}

sim = Simulation_template(pybulletConfigs, robotConfigs)
print(sim.joints)

def getMotorJointStates(p, robot):
    joint_states = p.getJointStates(robot, range(p.getNumJoints(robot)))
    joint_infos = [p.getJointInfo(robot, i) for i in range(p.getNumJoints(robot))]
    joint_states = [j for j, i in zip(joint_states, joint_infos) if i[3] > -1]
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    joint_torques = [state[3] for state in joint_states]
    return joint_positions, joint_velocities, joint_torques


mpos, mvel, mtorq = getMotorJointStates(bullet_simulation, sim.robot)
res = bullet_simulation.getLinkState(sim.robot,
                     sim.jointIds['LARM_JOINT5'],
                     computeLinkVelocity=1,
                     computeForwardKinematics=1)
link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = res
j_geo, j_rot = bullet_simulation.calculateJacobian(
    sim.robot, 
    sim.jointIds['LARM_JOINT5'],
    com_trn, 
    mpos,
    [0.0] * len(mpos),
    [0.0] * len(mpos),
)

def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # TODO modify from here
        # You can implement the cross product yourself or use calculateJacobian().
        bodyUniqueId=self.robot
        if endEffector == 'RHAND':
            linkIndex = self.jointIds["RARM_JOINT5"]
            localPosition = self.getJointPosition("RARM_JOINT5")
        elif endEffector == 'LHAND':
            linkIndex = self.jointIds["LARM_JOINT5"]
            localPosition = self.getJointPosition("LARM_JOINT5")
        else:
            raise Exception("[jacobianMatrix \
                endEffector not valid]")
        
        #objPositions = joint positions
        objPositions = [
        super().getJointPos('CHEST_JOINT0'),
        super().getJointPos('HEAD_JOINT0'),
        super().getJointPos('HEAD_JOINT1'),
        super().getJointPos('LARM_JOINT0'),
        super().getJointPos('LARM_JOINT1'), 
        super().getJointPos('LARM_JOINT2'),
        super().getJointPos('LARM_JOINT3'),
        super().getJointPos('LARM_JOINT4'),
        super().getJointPos('LARM_JOINT5'),
        super().getJointPos('RARM_JOINT0'),
        super().getJointPos('RARM_JOINT1'),
        super().getJointPos('RARM_JOINT2'),
        super().getJointPos('RARM_JOINT3'),
        super().getJointPos('RARM_JOINT4'),
        super().getJointPos('RARM_JOINT5')]
        zeroVec = [0.0] * (len(self.joints) - 2)
        objVelocities = zeroVec
        objAccelerations = zeroVec
        j_geo, j_rot =  bullet_simulation.calculateJacobian(bodyUniqueId, linkIndex, localPosition, objPositions, objVelocities, objAccelerations)
        return j_geo, j_rot

print()
for col in j_geo:
    print(col)
print()
for col in j_rot:
    print(col)
print()

myJ_geo, myJ_rot = sim.jacobianMatrix('LHAND')

for col in myJ_geo:
    print(col)
for col in myJ_rot:
    print(col)
print()

try:
    time.sleep(float(sys.argv[1]))
except:
    time.sleep(10)