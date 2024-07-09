/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using System.Collections;
using UnityEngine.Assertions;

/// <summary>
/// When this component is enabled, the player will be able to aim and trigger teleport behavior using Oculus Touch controllers.
/// </summary>
public class TeleportInputHandlerTouch : TeleportInputHandlerHMD
{
	public Transform LeftHand;
	public Transform RightHand;

	/// <summary>
	/// The touch input handler supports three different modes for controlling teleports.
	/// </summary>
	public enum InputModes
	{
		/// <summary>
		/// Touching a capacitive button will start the aiming, and pressing that button will trigger the teleport.
		/// </summary>
		CapacitiveButtonForAimAndTeleport,

		/// <summary>
		/// One button will start the aiming, another button will trigger the teleport.
		/// </summary>
		SeparateButtonsForAimAndTeleport,

		/// <summary>
		/// A thumbstick in any direction is used for starting the aiming, and releasing the thumbstick will trigger the teleport.
		/// </summary>
		ThumbstickTeleport,

		/// <summary>
		/// A thumbstick forward or back is used for starting the aiming, and releasing the thumbstick will trigger the teleport.
		/// </summary>
		ThumbstickTeleportForwardBackOnly
	}

	[Tooltip("CapacitiveButtonForAimAndTeleport=Activate aiming via cap touch detection, press the same button to teleport.\nSeparateButtonsForAimAndTeleport=Use one button to begin aiming, and another to trigger the teleport.\nThumbstickTeleport=Push a thumbstick to begin aiming, release to teleport.")]
	public InputModes InputMode;

	/// <summary>
	/// These buttons are used for selecting which capacitive button is used when InputMode==CapacitiveButtonForAimAndTeleport
	/// </summary>
	public enum AimCapTouchButtons
	{
		A,
		B,
		LeftTrigger,
		LeftThumbstick,
		RightTrigger,
		RightThumbstick,
		X,
		Y
	}

	private readonly OVRInput.RawButton[] _rawButtons = {
		OVRInput.RawButton.A,
		OVRInput.RawButton.B,
		OVRInput.RawButton.LIndexTrigger,
		OVRInput.RawButton.LThumbstick,
		OVRInput.RawButton.RIndexTrigger,
		OVRInput.RawButton.RThumbstick,
		OVRInput.RawButton.X,
		OVRInput.RawButton.Y
	};

	private readonly OVRInput.RawTouch[] _rawTouch = {
		OVRInput.RawTouch.A,
		OVRInput.RawTouch.B,
		OVRInput.RawTouch.LIndexTrigger,
		OVRInput.RawTouch.LThumbstick,
		OVRInput.RawTouch.RIndexTrigger,
		OVRInput.RawTouch.RThumbstick,
		OVRInput.RawTouch.X,
		OVRInput.RawTouch.Y
	};

	/// <summary>
	/// Which controller is being used for aiming.
	/// </summary>
	[Tooltip("Select the controller to be used for aiming. Supports LTouch, RTouch, or Touch for either.")]
	public OVRInput.Controller AimingController;

    private OVRInput.Controller InitiatingController;

	/// <summary>
	/// The button to use for triggering aim and teleport when InputMode==CapacitiveButtonForAimAndTeleport
	/// </summary>
	[Tooltip("Select the button to use for triggering aim and teleport when InputMode==CapacitiveButtonForAimAndTeleport")]
	public AimCapTouchButtons CapacitiveAimAndTeleportButton;

	/// <summary>
	/// The thumbstick magnitude required to trigger aiming and teleports when InputMode==InputModes.ThumbstickTeleport
	/// </summary>
	[Tooltip("The thumbstick magnitude required to trigger aiming and teleports when InputMode==InputModes.ThumbstickTeleport")]
	public float ThumbstickTeleportThreshold = 0.5f;

	void Start ()
    {
	}

	/// <summary>
	/// Based on the input mode, controller state, and current intention of the teleport controller, return the apparent intention of the user.
	/// </summary>
	/// <returns></returns>
	public override LocomotionTeleport.TeleportIntentions GetIntention()
	{
		if (!isActiveAndEnabled)
		{
			return global::LocomotionTeleport.TeleportIntentions.None;
		}

		// If capacitive touch isn't being used, the base implementation will do the work.
		if (InputMode == InputModes.SeparateButtonsForAimAndTeleport)
		{
			return base.GetIntention();
		}

        // ThumbstickTeleport will begin aiming when the thumbstick is pushed.
        if (InputMode == InputModes.ThumbstickTeleport || InputMode == InputModes.ThumbstickTeleportForwardBackOnly)
        {
            // Note there's a bit of wasted work here if you're only using 1 thumbstick to trigger teleport.
            // Feel free to delete the extra code for the unnecessary stick.
            Vector2 leftStick = OVRInput.Get(OVRInput.RawAxis2D.LThumbstick);
            Vector2 rightStick = OVRInput.Get(OVRInput.RawAxis2D.RThumbstick);
            float leftMag = 0.0f;
            float rightMag = 0.0f;
            float bestMag = 0.0f;
            OVRInput.Controller bestController = OVRInput.Controller.Touch;
            bool leftTouched = OVRInput.Get(OVRInput.RawTouch.LThumbstick);
            bool rightTouched = OVRInput.Get(OVRInput.RawTouch.RThumbstick);

            if (InputMode == InputModes.ThumbstickTeleportForwardBackOnly && LocomotionTeleport.CurrentIntention != LocomotionTeleport.TeleportIntentions.Aim)
            {
                // If user is aiming, ThumbstickTeleport and ThumbstickTeleportForwardBackOnly are identical. But if not, we only want magnitude along the forward or back vector.
                leftMag = Mathf.Abs(Vector2.Dot(leftStick, Vector2.up));
                rightMag = Mathf.Abs(Vector2.Dot(rightStick, Vector2.up));
            }
            else
            {
                leftMag = leftStick.magnitude;
                rightMag = rightStick.magnitude;
            }
            if (AimingController == OVRInput.Controller.LTouch)
            {
                bestMag = leftMag;
                bestController = OVRInput.Controller.LTouch;
            }
            else if (AimingController == OVRInput.Controller.RTouch)
            {
                bestMag = rightMag;
                bestController = OVRInput.Controller.RTouch;
            }
            else
            {
                if(leftMag > rightMag)
                {
                    bestMag = leftMag;
                    bestController = OVRInput.Controller.LTouch;
                }
                else
                {
                    bestMag = rightMag;
                    bestController = OVRInput.Controller.RTouch;
                }
            }

            bool touching = bestMag > ThumbstickTeleportThreshold
                || (AimingController == OVRInput.Controller.Touch && (leftTouched || rightTouched))
                || (AimingController == OVRInput.Controller.LTouch && leftTouched)
                || (AimingController == OVRInput.Controller.RTouch && rightTouched);
			if (!touching)
			{
				if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.Aim)
				{
					// If the user has released the thumbstick, enter the PreTeleport state unless FastTeleport is enabled, 
					// in which case enter the Teleport state.
					return FastTeleport ? LocomotionTeleport.TeleportIntentions.Teleport : LocomotionTeleport.TeleportIntentions.PreTeleport;
				}

				// If the user is already in the preteleport state, the intention will be to either remain in this state or switch to Teleport
				if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.PreTeleport)
				{
					return LocomotionTeleport.TeleportIntentions.Teleport;
				}
			}
			else
			{
				if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.Aim)
				{
					return LocomotionTeleport.TeleportIntentions.Aim;
				}
			}

			if (bestMag > ThumbstickTeleportThreshold)
			{
                InitiatingController = bestController;
				return LocomotionTeleport.TeleportIntentions.Aim;
			}

			return LocomotionTeleport.TeleportIntentions.None;
		}

		// Capacitive touch logic is essentially the same as the base logic, except the button types are different
		// so different methods need to be used.
		var teleportButton = _rawButtons[(int)CapacitiveAimAndTeleportButton];

		if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.Aim)
		{
			// If the user has actually pressed the teleport button, enter the preteleport state.
			if (OVRInput.GetDown(teleportButton))
			{
				// If the user has released the thumbstick, enter the PreTeleport state unless FastTeleport is enabled, 
				// in which case enter the Teleport state.
				return FastTeleport ? LocomotionTeleport.TeleportIntentions.Teleport : LocomotionTeleport.TeleportIntentions.PreTeleport;
			}
		}

		// If the user is already in the PreTeleport state, the intention will be to either remain in this state or switch to Teleport
		if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.PreTeleport)
		{
			// If they released the button, switch to Teleport.
			if (FastTeleport || OVRInput.GetUp(teleportButton))
			{
				// Button released, enter the Teleport state.
				return LocomotionTeleport.TeleportIntentions.Teleport;
			}
			// Button still down, remain in PreTeleport so they can orient the destination if an orientation handler supports it.
			return LocomotionTeleport.TeleportIntentions.PreTeleport;
		}

		// If it made it this far, then we need to determine if the user intends to be aiming with the capacitive touch.
		// The first check is if cap touch has been triggered. 
		if (OVRInput.GetDown(_rawTouch[(int)CapacitiveAimAndTeleportButton]))
		{
			return LocomotionTeleport.TeleportIntentions.Aim;
		}

		if (LocomotionTeleport.CurrentIntention == LocomotionTeleport.TeleportIntentions.Aim)
		{
			if (!OVRInput.GetUp(_rawTouch[(int)CapacitiveAimAndTeleportButton]))
			{
				return LocomotionTeleport.TeleportIntentions.Aim;
			}
		}

		return LocomotionTeleport.TeleportIntentions.None;
	}

	public override void GetAimData(out Ray aimRay)
	{
		OVRInput.Controller sourceController = AimingController;
		if(sourceController == OVRInput.Controller.Touch)
		{
			sourceController = InitiatingController;
		}
		Transform t = (sourceController == OVRInput.Controller.LTouch) ? LeftHand : RightHand;
		aimRay = new Ray(t.position, t.forward);
	}
}
