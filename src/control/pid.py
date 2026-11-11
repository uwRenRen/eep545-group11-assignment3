from __future__ import division
import numpy as np

from control.controller import BaseController
from control.controller import compute_position_in_frame


class PIDController(BaseController):
    def __init__(self, **kwargs):
        self.kp = kwargs.pop("kp")
        self.kd = kwargs.pop("kd")

        # Get the keyword args that we didn't consume with the above initialization
        super(PIDController, self).__init__(**kwargs)


    def get_error(self, pose, reference_xytv):
        """Compute the PD error.

        Args:
            pose: current state of the vehicle [x, y, heading]
            reference_xytv: reference state and speed

        Returns:
            error: across-track and cross-track error
        """
        return compute_position_in_frame(pose, reference_xytv[:3])

    def get_control(self, pose, reference_xytv, error):
        """Compute the PD control law.

        Args:
            pose: current state of the vehicle [x, y, heading]
            reference_xytv: reference state and speed
            error: error vector from get_error

        Returns:
            control: np.array of velocity and steering angle
                (velocity should be copied from reference velocity)
        """
        # BEGIN QUESTION 2.1
        # Cross error signal
        crossError = error[1]

        # Compute proportional part
        ProportionalC = crossError * self.kp

        # Compute derivative part. Change of error in time
        v = reference_xytv[3]
        DerivativeC = self.kd * v * np.sin(pose[2] - reference_xytv[2])

        # Sum the parts and get steering angle (current steering angle + control)
        steeringAngle =  - ( ProportionalC + DerivativeC )
        # Velocity stays the same
        return np.array([v, steeringAngle])
        
        # END QUESTION 2.1
