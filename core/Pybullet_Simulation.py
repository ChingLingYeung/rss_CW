from scipy.spatial.transform import Rotation as npRotation
from scipy.special import comb
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import pybullet as bullet_simulation
import pybullet_data
import numpy as np
import math
import re
import time
import yaml

from Pybullet_Simulation_base import Simulation_base

# TODO: Rename class name after copying this file
class Simulation(Simulation_base):
    """A Bullet simulation involving Nextage robot"""

    def __init__(self, pybulletConfigs, robotConfigs, refVect=None):
        """Constructor
        Creates a simulation instance with Nextage robot.
        For the keyword arguments, please see in the Pybullet_Simulation_base.py
        """
        super().__init__(pybulletConfigs, robotConfigs)
        if refVect:
            self.refVector = np.array(refVect)
        else:
            self.refVector = np.array([1,0,0])

    ########## Task 1: Kinematics ##########
    # Task 1.1 Forward Kinematics
    jointRotationAxis = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT0': np.array([0, 0, 1]),
        'HEAD_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT0': np.array([0, 0, 1]),
        'LARM_JOINT1': np.array([0, 1, 0]),
        'LARM_JOINT2': np.array([0, 1, 0]),
        'LARM_JOINT3': np.array([1, 0, 0]),
        'LARM_JOINT4': np.array([0, 1, 0]),
        'LARM_JOINT5': np.array([0, 0, 1]),
        'RARM_JOINT0': np.array([0, 0, 1]),
        'RARM_JOINT1': np.array([0, 1, 0]),
        'RARM_JOINT2': np.array([0, 1, 0]),
        'RARM_JOINT3': np.array([1, 0, 0]),
        'RARM_JOINT4': np.array([0, 1, 0]),
        'RARM_JOINT5': np.array([0, 0, 1]),
        'RHAND'      : np.array([0, 0, 1]),
        'LHAND'      : np.array([0, 0, 1])
    }

    frameTranslationFromParent = {
        'base_to_dummy': np.zeros(3),  # Virtual joint
        'base_to_waist': np.zeros(3),  # Fixed joint
        # TODO: modify from here
        'CHEST_JOINT0': np.array([0, 0, (0.267+0.85)]), 
        'HEAD_JOINT0': np.array([0, 0, 0.302]), 
        'HEAD_JOINT1': np.array([0, 0, 0.066]), 
        'LARM_JOINT0': np.array([0.04, 0.135, 0.1015]), 
        'LARM_JOINT1': np.array([0, 0, 0.066]), 
        'LARM_JOINT2': np.array([0, 0.095, -0.25]),
        'LARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'LARM_JOINT4': np.array([0.1495, 0, 0]),
        'LARM_JOINT5': np.array([0, 0, -0.1335]),
        'RARM_JOINT0': np.array([0.04, -0.135, 0.1015]),
        'RARM_JOINT1': np.array([0, 0, 0.066]),
        'RARM_JOINT2': np.array([0, -0.095, -0.25]),
        'RARM_JOINT3': np.array([0.1805, 0, -0.03]),
        'RARM_JOINT4': np.array([0.1495, 0, 0]),
        'RARM_JOINT5': np.array([0, 0, -0.1335])
    }

    robotJoints = ["CHEST_JOINT0", "HEAD_JOINT0", "HEAD_JOINT1", "LARM_JOINT0", "LARM_JOINT1", 'LARM_JOINT2', 'LARM_JOINT3',
        'LARM_JOINT4', 'LARM_JOINT5','RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2','RARM_JOINT3','RARM_JOINT4','RARM_JOINT5']

    def getJointRotationalMatrix(self, jointName=None, theta=None):
        """
            Returns the 3x3 rotation matrix for a joint from the axis-angle representation,
            where the axis is given by the revolution axis of the joint and the angle is theta.
        """
        if jointName == None: 
            raise Exception("[getJointRotationalMatrix] \
                Must provide a joint in order to compute the rotational matrix!") 
        # TODO modify from here
        # rotXMat = np.matrix([
        #         [1, 0, 0],
        #         [0, np.cos(theta), -np.sin(theta)],
        #         [0, np.sin(theta), np.cos(theta)]
        #     ])
        # rotYMat = np.matrix([
        #         [np.cos(theta), 0, np.sin(theta)],
        #         [0, 1, 0],
        #         [-np.sin(theta), 0, np.cos(theta)]
        #     ])
        # rotZMat = np.matrix([
        #         [np.cos(theta), -np.sin(theta), 0],
        #         [np.sin(theta), np.cos(theta), 0],
        #         [0, 0, 1]
        #     ])
        
        if jointName == "CHEST_JOINT0":
            theta = super().getJointPos("CHEST_JOINT0")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif jointName == "HEAD_JOINT0":
            theta = super().getJointPos("HEAD_JOINT0")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif jointName == "HEAD_JOINT1":
            theta = super().getJointPos("HEAD_JOINT1")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "LARM_JOINT0":
            theta = super().getJointPos("LARM_JOINT0")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif jointName == "LARM_JOINT1":
            theta = super().getJointPos("LARM_JOINT1")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "LARM_JOINT2":
            theta = super().getJointPos("LARM_JOINT2")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "LARM_JOINT3":
            theta = super().getJointPos("LARM_JOINT3")
            return np.matrix([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif jointName == "LARM_JOINT4":
            theta = super().getJointPos("LARM_JOINT4")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "LARM_JOINT5":
            theta = super().getJointPos("LARM_JOINT5")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif jointName == "RARM_JOINT0":
            theta = super().getJointPos("RARM_JOINT0")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        elif jointName == "RARM_JOINT1":
            theta = super().getJointPos("RARM_JOINT1")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "RARM_JOINT2":
            theta = super().getJointPos("RARM_JOINT2")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "RARM_JOINT3":
            theta = super().getJointPos("RARM_JOINT3")
            return np.matrix([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
        elif jointName == "RARM_JOINT4":
            theta = super().getJointPos("RARM_JOINT4")
            return np.matrix([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ])
        elif jointName == "RARM_JOINT5":
            theta = super().getJointPos("RARM_JOINT5")
            return np.matrix([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ])
        else:
            raise Exception("[getJointRotationalMatrix] \
                Invalid jointName provided")
    
    def getTransformationMatrices(self):
        """
            Returns the homogeneous transformation matrices for each joint as a dictionary of matrices.
        """
        transformationMatrices = {}
        # TODO modify from here
        translateCHEST_JOINT0 = self.frameTranslationFromParent["CHEST_JOINT0"]
        transformationMatrices["CHEST_JOINT0"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("CHEST_JOINT0"), translateCHEST_JOINT0], [np.array([0, 0, 0, 1])]])

        translateHEAD_JOINT0 = self.frameTranslationFromParent["HEAD_JOINT0"]
        transformationMatrices["HEAD_JOINT0"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("HEAD_JOINT0"), translateHEAD_JOINT0], [np.array([0, 0, 0, 1])]])

        translateHEAD_JOINT1 = self.frameTranslationFromParent["HEAD_JOINT1"]
        transformationMatrices["HEAD_JOINT1"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("HEAD_JOINT1"), translateHEAD_JOINT1], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT0 = self.frameTranslationFromParent["LARM_JOINT0"]
        transformationMatrices["LARM_JOINT0"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT0"), translateLARM_JOINT0], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT1 = self.frameTranslationFromParent["LARM_JOINT1"]
        transformationMatrices["LARM_JOINT1"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT1"), translateLARM_JOINT1], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT2 = self.frameTranslationFromParent["LARM_JOINT2"]
        transformationMatrices["LARM_JOINT2"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT2"), translateLARM_JOINT2], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT3 = self.frameTranslationFromParent["LARM_JOINT3"]
        transformationMatrices["LARM_JOINT3"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT3"), translateLARM_JOINT3], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT4 = self.frameTranslationFromParent["LARM_JOINT4"]
        transformationMatrices["LARM_JOINT4"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT4"), translateLARM_JOINT4], [np.array([0, 0, 0, 1])]])

        translateLARM_JOINT5 = self.frameTranslationFromParent["LARM_JOINT5"]
        transformationMatrices["LARM_JOINT5"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("LARM_JOINT5"), translateLARM_JOINT5], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT0 = self.frameTranslationFromParent["RARM_JOINT0"]
        transformationMatrices["RARM_JOINT0"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT0"), translateRARM_JOINT0], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT1 = self.frameTranslationFromParent["RARM_JOINT1"]
        transformationMatrices["RARM_JOINT1"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT1"), translateRARM_JOINT1], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT2 = self.frameTranslationFromParent["RARM_JOINT2"]
        transformationMatrices["RARM_JOINT2"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT2"), translateRARM_JOINT2], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT3 = self.frameTranslationFromParent["RARM_JOINT3"]
        transformationMatrices["RARM_JOINT3"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT3"), translateRARM_JOINT3], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT4 = self.frameTranslationFromParent["RARM_JOINT4"]
        transformationMatrices["RARM_JOINT4"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT4"), translateRARM_JOINT4], [np.array([0, 0, 0, 1])]])

        translateRARM_JOINT5 = self.frameTranslationFromParent["RARM_JOINT5"]
        transformationMatrices["RARM_JOINT5"] = np.asmatrix(np.r_[np.c_[self.getJointRotationalMatrix("RARM_JOINT5"), translateRARM_JOINT5], [np.array([0, 0, 0, 1])]])

        return transformationMatrices

    def getJointLocationAndOrientation(self, jointName):
        """
            Returns the position and rotation matrix of each joint using Forward Kinematics 
            according to the topology of the Nextage robot.
        """
        # Remember to multiply the transformation matrices following the kinematic chain for each arm.
        #TODO modify from here
        tMatrices = self.getTransformationMatrices()
        if jointName == "CHEST_JOINT0":
            j = tMatrices["CHEST_JOINT0"]
        elif jointName == "HEAD_JOINT0":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["HEAD_JOINT0"]
        elif jointName == "HEAD_JOINT1":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["HEAD_JOINT0"] @ tMatrices["HEAD_JOINT1"]
        elif jointName == "LARM_JOINT0":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"]
        elif jointName == "LARM_JOINT1":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"] @ tMatrices["LARM_JOINT1"]
        elif jointName == "LARM_JOINT2":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"] @ tMatrices["LARM_JOINT1"] @ tMatrices["LARM_JOINT2"]
        elif jointName == "LARM_JOINT3":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"] @ tMatrices["LARM_JOINT1"] @ tMatrices["LARM_JOINT2"] @ tMatrices["LARM_JOINT3"]
        elif jointName == "LARM_JOINT4":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"] @ tMatrices["LARM_JOINT1"] @ tMatrices["LARM_JOINT2"] @ tMatrices["LARM_JOINT3"] @ tMatrices["LARM_JOINT4"]
        elif jointName == "LARM_JOINT5":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["LARM_JOINT0"] @ tMatrices["LARM_JOINT1"] @ tMatrices["LARM_JOINT2"] @ tMatrices["LARM_JOINT3"] @ tMatrices["LARM_JOINT4"] @ tMatrices["LARM_JOINT5"]
        elif jointName == "RARM_JOINT0":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"]
        elif jointName == "RARM_JOINT1":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"] @ tMatrices["RARM_JOINT1"]
        elif jointName == "RARM_JOINT2":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"] @ tMatrices["RARM_JOINT1"] @ tMatrices["RARM_JOINT2"]
        elif jointName == "RARM_JOINT3":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"] @ tMatrices["RARM_JOINT1"] @ tMatrices["RARM_JOINT2"] @ tMatrices["RARM_JOINT3"]
        elif jointName == "RARM_JOINT4":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"] @ tMatrices["RARM_JOINT1"] @ tMatrices["RARM_JOINT2"] @ tMatrices["RARM_JOINT3"] @ tMatrices["RARM_JOINT4"]
        elif jointName == "RARM_JOINT5":
            j = tMatrices["CHEST_JOINT0"] @ tMatrices["RARM_JOINT0"] @ tMatrices["RARM_JOINT1"] @ tMatrices["RARM_JOINT2"] @ tMatrices["RARM_JOINT3"] @ tMatrices["RARM_JOINT4"] @ tMatrices["RARM_JOINT5"]
        else:
            raise Exception("[getJointLocationAndOrientation] \
                 Must provide valid jointName")
        
        pos = [j[0,3], j[1,3], j[2,3]]
        rotmat = [[j[0,0], j[0,1], j[0,2]], [j[1,0], j[1,1], j[1,2]], [j[2,0], j[2,1], j[2,2]]]
        return pos, rotmat
    
    def getJointPosition(self, jointName):
        """Get the position of a joint in the world frame, leave this unchanged please."""
        return self.getJointLocationAndOrientation(jointName)[0]
    
    def getJointOrientation(self, jointName, ref=None):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        if ref is None:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.refVector).squeeze()
        else:
            return np.array(self.getJointLocationAndOrientation(jointName)[1] @ ref).squeeze()

    def getJointAxis(self, jointName):
        """Get the orientation of a joint in the world frame, leave this unchanged please."""
        return np.array(self.getJointLocationAndOrientation(jointName)[1] @ self.jointRotationAxis[jointName]).squeeze()

    def jacobianMatrix(self, endEffector):
        """Calculate the Jacobian Matrix for the Nextage Robot."""
        # TODO modify from here
        # You can implement the cross product yourself or use calculateJacobian().
        bodyUniqueId=self.robot
        if endEffector == 'RHAND':
            linkIndex = self.jointIds["RARM_JOINT5"]
            jointName = "RARM_JOINT5"
            localPosition = self.getJointPosition("RARM_JOINT5")
        elif endEffector == 'LHAND':
            linkIndex = self.jointIds["LARM_JOINT5"]
            jointName = "LARM_JOINT5"
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

        
        # needed to use getLinkState() to get center of mass for calculateJacobian()
        com = self.getLinkState(jointName)[2]

        # comtest = self.getCenterOfMassPosition()
        
        #used for debugging purposes
        # result = bullet_simulation.getLinkState(self.robot,
        #                 linkIndex,
        #                 computeLinkVelocity=1,
        #                 computeForwardKinematics=1)
        # link_trn, link_rot, com_trn, com_rot, frame_pos, frame_rot, link_vt, link_vr = result

        # print(com==comtest)
        # print(com_trn)
        # print(comtest)

        j_geo, j_rot =  bullet_simulation.calculateJacobian(bodyUniqueId, linkIndex, com, objPositions, objVelocities, objAccelerations)
        # j_geo, j_rot =  bullet_simulation.calculateJacobian(bodyUniqueId, linkIndex, com_trn, objPositions, objVelocities, objAccelerations)
        return j_geo, j_rot

    def newJacobian(self, endEffector):
        if endEffector == 'RHAND':
            jointName = "RARM_JOINT5"
            localPosition = self.getJointPosition("RARM_JOINT5")
        elif endEffector == 'LHAND':
            jointName = "LARM_JOINT5"
            localPosition = self.getJointPosition("LARM_JOINT5")
        else:
            raise Exception("[jacobianMatrix \
                endEffector not valid]")

        aeff = self.getJointAxis(jointName)
        peff = np.array(localPosition)

        jPos = [[],[],[]]
        jVec = [[],[],[]]

        for j in self.robotJoints:
            # print(j)
            ai = self.getJointAxis(j)
            pi = np.array(self.getJointPosition(j))
            # refVec?
            # orientation instead?
            
            # print(peff)
            # print(pi)
            pdiff = peff - pi
            crossPos = np.cross(ai, pdiff)
            # crossPosT = np.array([np.cross(ai, pdiff)]).T
            # print(cross1)

            # jPos = np.column_stack((jPos, crossPos))
            jPos = np.c_[jPos, crossPos]
            # jPos.append(crossPos)
            # jPos = np.hstack((jPos,cross1))

            crossVec = np.cross(ai, aeff)
            # crossVecT = np.array([np.cross(ai, aeff)]).T

            # jVec = np.column_stack((jVec,crossVec))
            jVec = np.c_[jVec, crossVec]
            # jVec.append(crossVec)
            # jVec = np.hstack((jVec, crossVec))

        
        # print(jPos)
        # print(jPos.shape)
        # print(jVec.shape)
        return jPos, jVec


    # Task 1.2 Inverse Kinematics

    def inverseKinematics(self, endEffector, targetPosition, orientation, frame=None):
        """Your IK solver \\
        Arguments: \\
            endEffector: the jointName the end-effector \\
            targetPosition: final destination the the end-effector \\
        Keywork Arguments: \\
            orientation: the desired orientation of the end-effector
                         together with its parent link \\
            speed: how fast the end-effector should move (m/s) \\
            orientation: the desired orientation \\
            compensationRatio: naive gravity compensation ratio \\
            debugLine: optional \\
            verbose: optional \\
        Return: \\
            List of x_refs
        """
        # TODO add your code here

        #get initial end effector position
        if endEffector == 'LHAND':
            endEffectorPos = self.getJointPosition("LARM_JOINT5")
            endEffectorRotMat = self.getJointRotationalMatrix("LARM_JOINT5")
            efEuler = npRotation.from_matrix(endEffectorRotMat).as_euler('zyx', degrees=False)
            endEffectorOrientation = self.getJointOrientation("LARM_JOINT5")
        elif endEffector == 'RHAND':
            endEffectorPos = self.getJointPosition("RARM_JOINT5")
            endEffectorRotMat = self.getJointRotationalMatrix("RARM_JOINT5")
            efEuler = npRotation.from_matrix(endEffectorRotMat).as_euler('zyx', degrees=False)
            endEffectorOrientation = self.getJointOrientation("RARM_JOINT5")
        else:
            raise Exception("[inverseKinematics \
                end effector not valid]")

        # print(endEffectorOrientation)
        # input()

        # current joint angles
        newTheta = [
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


        # J_geo, J_rot = self.jacobianMatrix(endEffector)
        J_geo, J_rot = self.newJacobian(endEffector)

        J = np.vstack((J_geo, J_rot))

        deltaPosition = targetPosition - endEffectorPos           
        
        if orientation is None:
            radTheta = (np.linalg.pinv(J_geo) @ deltaPosition) 
        else:
            orientation = np.nan_to_num(super().normaliseVector(orientation))
            endEffectorOrientation = np.nan_to_num(super().normaliseVector(endEffectorOrientation))

            #https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
            angleBtwnOr = np.arccos(np.clip(np.dot(orientation, endEffectorOrientation), -1.0, 1.0))

            deltaOrientation = (orientation - endEffectorOrientation) * np.absolute(angleBtwnOr)
            # deltaOrientation = orientation - endEffectorOrientation

            # deltaOrientation = np.nan_to_num(super().normaliseVector(orientation)) - np.nan_to_num(super().normaliseVector(endEffectorOrientation))
            # deltaOrientation = np.nan_to_num(super().normaliseVector(deltaOrientation))
            # print(deltaOrientation)
            # deltaOrientation = orientation - efEuler
            deltas = np.hstack((deltaPosition, deltaOrientation))           
            # print(deltaOrientation)

            # weightedTarget = np.hstack((targetPosition, np.multiply(orientation, 0.001)))
            # current = np.hstack((endEffectorPos, np.multiply(endEffectorOrientation, 0.001)))
            weightedTarget = np.hstack((targetPosition, orientation))
            current = np.hstack((endEffectorPos, endEffectorOrientation))
            
            error = np.linalg.norm(weightedTarget - current)

            errorPos = np.linalg.norm(targetPosition - endEffectorPos)

            normalOr = np.nan_to_num(super().normaliseVector(orientation))
            efOr = np.nan_to_num(super().normaliseVector(endEffectorOrientation))
            normDiff = np.nan_to_num(super().normaliseVector(orientation-endEffectorOrientation))
            errorOr = np.linalg.norm(normalOr - efOr)
            
            errorDiff = np.linalg.norm(np.nan_to_num(normDiff))
            # errorOr = np.linalg.norm(orientation - endEffectorOrientation) #* 0.01

            errorSum = errorPos + errorOr
            # print(errorPos)
            # print(errorOr)
            # print(errorSum)
            # input()

            radTheta = (np.linalg.pinv(J) @ deltas) * errorSum

            # radTheta1 = (np.linalg.pinv(J_geo) @ deltaPosition) * errorPos
            # radTheta2 = (np.linalg.pinv(J_rot) @ deltaOrientation) * errorOr #* errorPos# * errorOr
            # radTheta = (radTheta1 + radTheta2) * error# * errorOr

        # print(radTheta)
        newTheta = newTheta + radTheta # new joint positions
        return newTheta


    def move_without_PD(self, endEffector, targetPosition, speed=0.01, orientation=None, 
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using Inverse Kinematics solver (without using PD control).
        This method should update joint states directly.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # iterate through joints and update joint states based on IK solver
        if endEffector == 'LHAND':
            initPosition = self.getJointPosition("LARM_JOINT5")
            initOrientation = self.getJointOrientation("LARM_JOINT5")
        elif endEffector == 'RHAND':
            initPosition = self.getJointPosition("RARM_JOINT5")
            initOrientation = self.getJointOrientation("RARM_JOINT5")
        else:
            raise Exception("[move_without_PD \
                end effector not valid]")
        
        error = np.linalg.norm(targetPosition - initPosition)
        # initOrientation = np.multiply(initOrientation, 0.01)
        # print(initOrientation)
        # print(np.array(initOrientation))

        if orientation is not None:
            orientation = np.nan_to_num(super().normaliseVector(orientation))
            endEffectorOrientation = np.nan_to_num(super().normaliseVector(initOrientation))

            errorOr = np.linalg.norm(orientation - endEffectorOrientation)
            
            error = error + errorOr
            # error = np.linalg.norm(target - initPose)

            
        curIter = 0
        pltTime = []
        pltDistance = []

        step_positions = np.linspace(initPosition, targetPosition, maxIter)

        if orientation is not None:
            step_orientations = np.linspace(initOrientation, orientation, maxIter)

        min_error = 9

        for i in range(maxIter):
            if error <= threshold:
                break
            
            # TODO: might need to verify if jointTargetPos is valid
            if orientation is None:
                x_refs = self.inverseKinematics(endEffector, step_positions[i, :], orientation)
            else:
                x_refs = self.inverseKinematics(endEffector, step_positions[i, :], step_orientations[i, :])
            self.jointTargetPos["CHEST_JOINT0"] = x_refs[0]
            self.jointTargetPos["HEAD_JOINT0"] = x_refs[1]
            self.jointTargetPos["HEAD_JOINT1"] = x_refs[2]
            self.jointTargetPos["LARM_JOINT0"] = x_refs[3]
            self.jointTargetPos["LARM_JOINT1"] = x_refs[4]
            self.jointTargetPos["LARM_JOINT2"] = x_refs[5]
            self.jointTargetPos["LARM_JOINT3"] = x_refs[6]
            self.jointTargetPos["LARM_JOINT4"] = x_refs[7]
            self.jointTargetPos["LARM_JOINT5"] = x_refs[8]
            self.jointTargetPos["RARM_JOINT0"] = x_refs[9]
            self.jointTargetPos["RARM_JOINT1"] = x_refs[10]
            self.jointTargetPos["RARM_JOINT2"] = x_refs[11]
            self.jointTargetPos["RARM_JOINT3"] = x_refs[12]
            self.jointTargetPos["RARM_JOINT4"] = x_refs[13]
            self.jointTargetPos["RARM_JOINT5"] = x_refs[14]

            self.tick_without_PD()
            # input()

            if endEffector == 'LHAND':
                endEffectorPos = self.getJointPosition("LARM_JOINT5")
                endEffectorOrientation = self.getJointOrientation("LARM_JOINT5")
            elif endEffector == 'RHAND':
                endEffectorPos = self.getJointPosition("RARM_JOINT5")
                endEffectorOrientation = self.getJointOrientation("RARM_JOINT5")


            error = np.linalg.norm(targetPosition - endEffectorPos)

            if orientation is not None:
                # currentPose = np.hstack((endEffectorPos, np.multiply(endEffectorOrientation, 0.001)))
                # target = np.hstack((targetPosition, np.multiply(orientation, 0.001)))
                currentPose = np.hstack((endEffectorPos, endEffectorOrientation))
                target = np.hstack((targetPosition, orientation))
                # if not np.array_equal(endEffectorOrientation, orientation):
                #     dot = np.dot(endEffectorOrientation, orientation)
                #     angle = np.arccos(dot)
                #     print("radian")
                #     print(angle)
                # error = np.linalg.norm(target - currentPose)
                normalOr = super().normaliseVector(orientation)
                efOr = super().normaliseVector(endEffectorOrientation)
                errorOr = np.linalg.norm(normalOr - efOr)
                error = error + errorOr
            
            # print(curIter)
            if error < min_error:
                print("min error")
                print(error)
                # print(endEffectorOrientation)

                min_error = error
            curIter += 1
            pltTime.append(curIter)
            pltDistance.append(error)

        # print(endEffectorOrientation)

        return pltTime, pltDistance

    def tick_without_PD(self):
        """Ticks one step of simulation without PD control. """
        # TODO modify from here
        # Iterate through all joints and update joint states. 
            # For each joint, you can use the shared variable self.jointTargetPos.
        bullet_simulation.resetJointState(self.robot, self.jointIds["CHEST_JOINT0"], self.jointTargetPos["CHEST_JOINT0"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["HEAD_JOINT0"], self.jointTargetPos["HEAD_JOINT0"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["HEAD_JOINT1"], self.jointTargetPos["HEAD_JOINT1"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT0"], self.jointTargetPos["LARM_JOINT0"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT1"], self.jointTargetPos["LARM_JOINT1"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT2"], self.jointTargetPos["LARM_JOINT2"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT3"], self.jointTargetPos["LARM_JOINT3"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT4"], self.jointTargetPos["LARM_JOINT4"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["LARM_JOINT5"], self.jointTargetPos["LARM_JOINT5"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT0"], self.jointTargetPos["RARM_JOINT0"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT1"], self.jointTargetPos["RARM_JOINT1"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT2"], self.jointTargetPos["RARM_JOINT2"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT3"], self.jointTargetPos["RARM_JOINT3"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT4"], self.jointTargetPos["RARM_JOINT4"])
        bullet_simulation.resetJointState(self.robot, self.jointIds["RARM_JOINT5"], self.jointTargetPos["RARM_JOINT5"])
        
        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)


    ########## Task 2: Dynamics ##########
    # Task 2.1 PD Controller
    def calculateTorque(self, x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd):
        """ This method implements the closed-loop control \\
        Arguments: \\
            x_ref - the target position \\
            x_real - current position \\
            dx_ref - target velocity \\
            dx_real - current velocity \\
            integral - integral term (set to 0 for PD control) \\
            kp - proportional gain \\
            kd - derivetive gain \\
            ki - integral gain \\
        Returns: \\
            u(t) - the manipulation signal
        """
        # TODO: Add your code here
        u_t = kp * (x_ref - x_real) + kd * (dx_ref - dx_real) + ki * integral
        # print(u_t)
        return u_t

    # Task 2.2 Joint Manipulation
    def moveJoint(self, joint, targetPosition, targetVelocity, verbose=False):
        """ This method moves a joint with your PD controller. \\
        Arguments: \\
            joint - the name of the joint \\
            targetPos - target joint position \\
            targetVel - target joint velocity 
        """
        def toy_tick(x_ref, x_real, dx_ref, dx_real, integral):
            # loads your PID gains
            jointController = self.jointControllers[joint]
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Start your code here: ###
            # Calculate the torque with the above method you've made
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, integral, kp, ki, kd)
            ### To here ###

            pltTorque.append(torque)

            # send the manipulation signal to the joint
            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )
            # calculate the physics and update the world
            self.p.stepSimulation()
            time.sleep(self.dt)

        targetPosition, targetVelocity = float(targetPosition), float(targetVelocity)

        # disable joint velocity controller before apply a torque
        self.disableVelocityController(joint)
        # logging for the graph
        pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity = [], [], [], [], [], []

        ### TODO: implement your code from here

        controlCycles = int(self.controlFrequency/self.updateFrequency)
        xOld = 0
        eOld = 0
        curTime = 0

        self.disableVelocityController(joint)

        #TODO: delete while
        endTime = 10
        while curTime < endTime:
            xReal = self.getJointPos(joint)
            e = targetPosition - xReal
            de = (e - eOld)/(self.dt * controlCycles)

            for i in range(controlCycles):
                # calculate
                toy_tick(targetPosition, xReal, targetVelocity, self.getJointVel(joint), 0)
                pltTime.append(curTime)
                pltTarget.append(targetPosition)
                pltPosition.append(self.getJointPos(joint))
                pltVelocity.append(self.getJointVel(joint))
                pltTorqueTime.append(curTime)
            
            xOld = xReal
            eOld = e
            curTime += controlCycles * self.dt        

        # print(targetPosition - self.getJointPos(joint))

        return pltTime, pltTarget, pltTorque, pltTorqueTime, pltPosition, pltVelocity

    def move_with_PD(self, endEffector, targetPosition, speed=0.01, orientation=None,
        threshold=1e-3, maxIter=3000, debug=False, verbose=False):
        """
        Move joints using inverse kinematics solver and using PD control.
        This method should update joint states using the torque output from the PD controller.
        Return:
            pltTime, pltDistance arrays used for plotting
        """
        #TODO add your code here
        # Iterate through joints and use states from IK solver as reference states in PD controller.
        # Perform iterations to track reference states using PD controller until reaching
        # max iterations or position threshold.

        if endEffector == 'LHAND':
            initPosition = self.getJointPosition("LARM_JOINT5")
            initOrientation = self.getJointOrientation("LARM_JOINT5")
        elif endEffector == 'RHAND':
            initPosition = self.getJointPosition("RARM_JOINT5")
            initOrientation = self.getJointOrientation("RARM_JOINT5")
        else:
            raise Exception("[move_with_PD \
                end effector not valid]")

        error = np.linalg.norm(targetPosition - initPosition)
        curIter = 0
        pltTime = []
        pltDistance = []

        step_positions = np.linspace(initPosition, targetPosition, maxIter)

        if orientation is not None:
            step_orientations = np.linspace(initOrientation, orientation, maxIter)
            orientation = np.nan_to_num(super().normaliseVector(orientation))
            endEffectorOrientation = np.nan_to_num(super().normaliseVector(initOrientation))

            errorOr = np.linalg.norm(orientation - endEffectorOrientation)
            error = error + errorOr

        for i in range(maxIter):
            if error <= threshold:
                break

            if orientation is None:
                x_refs = self.inverseKinematics(endEffector, step_positions[i, :], orientation)
            else:
                x_refs = self.inverseKinematics(endEffector, step_positions[i, :], step_orientations[i, :])
            #TODO move joints
            self.jointTargetPos["CHEST_JOINT0"] = x_refs[0]
            self.jointTargetPos["HEAD_JOINT0"] = x_refs[1]
            self.jointTargetPos["HEAD_JOINT1"] = x_refs[2]
            self.jointTargetPos["LARM_JOINT0"] = x_refs[3]
            self.jointTargetPos["LARM_JOINT1"] = x_refs[4]
            self.jointTargetPos["LARM_JOINT2"] = x_refs[5]
            self.jointTargetPos["LARM_JOINT3"] = x_refs[6]
            self.jointTargetPos["LARM_JOINT4"] = x_refs[7]
            self.jointTargetPos["LARM_JOINT5"] = x_refs[8]
            self.jointTargetPos["RARM_JOINT0"] = x_refs[9]
            self.jointTargetPos["RARM_JOINT1"] = x_refs[10]
            self.jointTargetPos["RARM_JOINT2"] = x_refs[11]
            self.jointTargetPos["RARM_JOINT3"] = x_refs[12]
            self.jointTargetPos["RARM_JOINT4"] = x_refs[13]
            self.jointTargetPos["RARM_JOINT5"] = x_refs[14]
            #use self.moveJoint(joint, xref[i], 0.0)?
            self.tick()

            if endEffector == 'LHAND':
                endEffectorPos = self.getJointPosition("LARM_JOINT5")
                endEffectorOrientation = self.getJointOrientation("LARM_JOINT5")
            elif endEffector == 'RHAND':
                endEffectorPos = self.getJointPosition("RARM_JOINT5")
                endEffectorOrientation = self.getJointOrientation("RARM_JOINT5")

            error = np.linalg.norm(targetPosition - endEffectorPos)

            if orientation is not None:
                normalOr = super().normaliseVector(orientation)
                efOr = super().normaliseVector(endEffectorOrientation)
                errorOr = np.linalg.norm(normalOr - efOr)
                error = error + errorOr

            print(curIter)
            print("error")
            print(error)
            curIter += 1
            pltTime.append(curIter)
            pltDistance.append(error)

        return pltTime, pltDistance
        
    def tick(self):
        """Ticks one step of simulation using PD control."""
        # Iterate through all joints and update joint states using PD control. 
        for joint in self.joints:
            # skip dummy joints (world to base joint)
            jointController = self.jointControllers[joint]
            if jointController == 'SKIP_THIS_JOINT':
                continue

            # disable joint velocity controller before apply a torque
            self.disableVelocityController(joint)

            # loads your PID gains
            kp = self.ctrlConfig[jointController]['pid']['p']
            ki = self.ctrlConfig[jointController]['pid']['i']
            kd = self.ctrlConfig[jointController]['pid']['d']

            ### Implement your code from here ... ###
            # TODO: obtain torque from PD controller
            x_ref = self.jointTargetPos[joint]
            x_real = self.getJointPos(joint)
            dx_ref = 0.0
            dx_real = self.getJointVel(joint)
            torque = self.calculateTorque(x_ref, x_real, dx_ref, dx_real, 0, kp, ki, kd) # TODO: fix me
            
            ### ... to here ###

            self.p.setJointMotorControl2(
                bodyIndex=self.robot,
                jointIndex=self.jointIds[joint],
                controlMode=self.p.TORQUE_CONTROL,
                force=torque
            )

            # Gravity compensation
            # A naive gravitiy compensation is provided for you
            # If you have embeded a better compensation, feel free to modify
            compensation = self.jointGravCompensation[joint]
            self.p.applyExternalForce(
                objectUniqueId=self.robot,
                linkIndex=self.jointIds[joint],
                forceObj=[0, 0, -compensation],
                posObj=self.getLinkCoM(joint),  
                flags=self.p.WORLD_FRAME
            )
            # Gravity compensation ends here

        self.p.stepSimulation()
        self.drawDebugLines()
        time.sleep(self.dt)

    ########## Task 3: Robot Manipulation ##########
    def cubic_interpolation(self, points, nTimes=100):
        """
        Given a set of control points, return the
        cubic spline defined by the control points,
        sampled nTimes along the curve.
        """
        #TODO add your code here
        # Return 'nTimes' points per dimension in 'points' (typically a 2xN array), 
        # sampled from a cubic spline defined by 'points' and a boundary condition. 
        # You may use methods found in scipy.interpolate

        #use scipy.interpolate.CubicSpline(x, y, axis=0, bc_type='not-a-knot', extrapolate=None)?

        #return xpoints, ypoints
        pass 

    # Task 3.1 Pushing
    def dockingToPosition(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, 
            threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass
    
    # Task 3.2 Grasping & Docking
    def clamp(self, leftTargetAngle, rightTargetAngle, angularSpeed=0.005, threshold=1e-1, maxIter=300, verbose=False):
        """A template function for you, you are free to use anything else"""
        # TODO: Append your code here
        pass

 ### END


