/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// This orientation handler will use the specified thumbstick to adjust the landing orientation of the teleport.
/// </summary>
public class TeleportOrientationHandlerThumbstick : TeleportOrientationHandler
{
	/// <summary>
	/// HeadRelative=Character will orient to match the arrow. ForwardFacing=When user orients to match the arrow, they will be facing the sensors.
	/// </summary>
	[Tooltip("HeadRelative=Character will orient to match the arrow. ForwardFacing=When user orients to match the arrow, they will be facing the sensors.")]
	public OrientationModes OrientationMode;

	/// <summary>
	/// Which thumbstick is to be used for adjusting the teleport orientation.
	/// </summary>
	[Tooltip("Which thumbstick is to be used for adjusting the teleport orientation. Supports LTouch, RTouch, or Touch for either.")]
	public OVRInput.Controller Thumbstick;

	/// <summary>
	/// The orientation will only change if the thumbstick magnitude is above this value. This will usually be larger than the TeleportInputHandlerTouch.ThumbstickTeleportThreshold.
	/// </summary>
	[Tooltip("The orientation will only change if the thumbstick magnitude is above this value. This will usually be larger than the TeleportInputHandlerTouch.ThumbstickTeleportThreshold.")]
	public float RotateStickThreshold = 0.8f;

	private Quaternion _initialRotation;
	private Quaternion _currentRotation;
	private Vector2 _lastValidDirection;

	protected override void InitializeTeleportDestination()
	{
		_initialRotation = LocomotionTeleport.GetHeadRotationY();
		_currentRotation = _initialRotation;
		_lastValidDirection = new Vector2();
	}

	protected override void UpdateTeleportDestination()
	{
        float magnitude;
        Vector2 direction;
        if (Thumbstick == OVRInput.Controller.Touch)
        {
            Vector2 leftDir = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
            Vector2 rightDir = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
            float leftMag = leftDir.magnitude;
            float rightMag = rightDir.magnitude;
            if (leftMag > rightMag)
            {
                magnitude = leftMag;
                direction = leftDir;
            }
            else
            {
                magnitude = rightMag;
                direction = rightDir;
            }
        }
        else
        {
            if(Thumbstick == OVRInput.Controller.LTouch) direction = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
            else direction = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
            magnitude = direction.magnitude;
        }

		if (!AimData.TargetValid)
		{
			_lastValidDirection = new Vector2();
		}

		if (magnitude < RotateStickThreshold)
		{
			direction = _lastValidDirection;
			magnitude = direction.magnitude;

			if (magnitude < RotateStickThreshold)
			{
				_initialRotation = LocomotionTeleport.GetHeadRotationY();
				direction.x = 0;
				direction.y = 1;
			}
		}
		else
		{
			_lastValidDirection = direction;
		}

		var tracking = LocomotionTeleport.LocomotionController.CameraRig.trackingSpace.rotation;

		if (magnitude > RotateStickThreshold)
		{
			direction /= magnitude; // normalize the vector
			var rot = _initialRotation * Quaternion.LookRotation(new Vector3(direction.x, 0, direction.y), Vector3.up);
			_currentRotation = tracking * rot;
		}
		else
		{
			_currentRotation = tracking * LocomotionTeleport.GetHeadRotationY();
		}

		LocomotionTeleport.OnUpdateTeleportDestination(AimData.TargetValid, AimData.Destination, _currentRotation, GetLandingOrientation(OrientationMode, _currentRotation));
	}
}
