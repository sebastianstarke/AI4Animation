/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// When this component is enabled, the player will be able to aim and trigger teleport behavior using HMD aiming.
/// </summary>
public class TeleportInputHandlerHMD : TeleportInputHandler
{
	public Transform Pointer { get; private set; }

	/// <summary>
	/// The button used to begin aiming for a teleport.
	/// </summary>
	[Tooltip("The button used to begin aiming for a teleport.")]
	public OVRInput.RawButton AimButton;

	/// <summary>
	/// The button used to trigger the teleport after aiming. It can be the same button as the AimButton, however you cannot 
	/// abort a teleport if it is.
	/// </summary>
	[Tooltip("The button used to trigger the teleport after aiming. It can be the same button as the AimButton, however you cannot abort a teleport if it is.")]
	public OVRInput.RawButton TeleportButton;

	/// <summary>
	/// When true, the system will not use the PreTeleport intention which will allow a teleport to occur on a button downpress. 
	/// When false, the button downpress will trigger the PreTeleport intention and the Teleport intention when the button is released.
	/// </summary>
	[Tooltip("When true, the system will not use the PreTeleport intention which will allow a teleport to occur on a button downpress. When false, the button downpress will trigger the PreTeleport intention and the Teleport intention when the button is released.")]
	public bool FastTeleport;

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

		if (LocomotionTeleport.CurrentIntention == global::LocomotionTeleport.TeleportIntentions.Aim)
		{
			// If the user has actually pressed the teleport button, enter the preteleport state.
			if (OVRInput.GetDown(TeleportButton))
			{
				return FastTeleport ? global::LocomotionTeleport.TeleportIntentions.Teleport : global::LocomotionTeleport.TeleportIntentions.PreTeleport;
			}
		}

		// If the user is already in the preteleport state, the intention will be to either remain in this state or switch to Teleport
		if (LocomotionTeleport.CurrentIntention == global::LocomotionTeleport.TeleportIntentions.PreTeleport)
		{
			// If they released the button, switch to Teleport.
			if (OVRInput.GetUp(TeleportButton))
			{
				// Button released, enter the teleport state.
				return global::LocomotionTeleport.TeleportIntentions.Teleport;
			}
			// Button still down, remain in PreTeleport so they can orient the destination if an orientation handler supports it.
			return global::LocomotionTeleport.TeleportIntentions.PreTeleport;
		}


		// Update the aim intention based on the button press state of the AimButton.
		if (OVRInput.Get(AimButton))
		{
			return global::LocomotionTeleport.TeleportIntentions.Aim;
		}

		// If the aim button is the same as the teleport button, then there is no way to abort the teleport.
		// Since the process of triggering the teleport is complete, returning the Teleport intention is necessary otherwise
		// the input handler would remain active.
		if (AimButton == TeleportButton)
		{
			return global::LocomotionTeleport.TeleportIntentions.Teleport;
		}

		return LocomotionTeleport.TeleportIntentions.None;
	}

	public override void GetAimData(out Ray aimRay)
	{
		var t = LocomotionTeleport.LocomotionController.CameraRig.centerEyeAnchor;
		aimRay = new Ray(t.position, t.forward);
	}
}
