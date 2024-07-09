/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using System.Collections;

/// <summary>
/// The orientation handler is what determines the final rotation of the player after the teleport.
/// </summary>
public abstract class TeleportOrientationHandler : TeleportSupport
{
	private readonly Action _updateOrientationAction;
	private readonly Action<LocomotionTeleport.AimData> _updateAimDataAction;
	protected LocomotionTeleport.AimData AimData;

	protected TeleportOrientationHandler()
	{
		_updateOrientationAction = () => { StartCoroutine(UpdateOrientationCoroutine()); };
		_updateAimDataAction = UpdateAimData;
	}

	private void UpdateAimData(LocomotionTeleport.AimData aimData)
	{
		AimData = aimData;
	}

	protected override void AddEventHandlers()
	{
		base.AddEventHandlers();
		LocomotionTeleport.EnterStateAim += _updateOrientationAction;
		LocomotionTeleport.UpdateAimData += _updateAimDataAction;
	}

	protected override void RemoveEventHandlers()
	{
		base.RemoveEventHandlers();
		LocomotionTeleport.EnterStateAim -= _updateOrientationAction;
		LocomotionTeleport.UpdateAimData -= _updateAimDataAction;
	}

	private IEnumerator UpdateOrientationCoroutine()
	{
		InitializeTeleportDestination();

		while (LocomotionTeleport.CurrentState == LocomotionTeleport.States.Aim || LocomotionTeleport.CurrentState == LocomotionTeleport.States.PreTeleport)
		{
			if (AimData != null)
			{
				UpdateTeleportDestination();
			}
			yield return null;
		}
	}

	protected abstract void InitializeTeleportDestination();

	protected abstract void UpdateTeleportDestination();

	/// <summary>
	/// The OrientationModes are used to specify how the character should be oriented when they land 
	/// after a teleport.
	/// </summary>
	public enum OrientationModes
	{
		/// <summary>
		/// When the player teleports, they will match the orientation of the destination indicator without adjusting their HMD's 
		/// orientation.  
		/// </summary>
		HeadRelative,

		/// <summary>
		/// When the player teleports, the player will be oriented so that when they turn the HMD to match the destination indicator,
		/// they will be facing forward with respect to the Oculus sensor setup. They will not immediately face the direction of the 
		/// indicator, and will need to rotated the HMD view to match the indicated direction. Once rotated, the player will be facing 
		/// forward to the Oculus sensors.
		/// </summary>
		ForwardFacing
	}

	protected Quaternion GetLandingOrientation(OrientationModes mode, Quaternion rotation)
	{
		return mode == OrientationModes.HeadRelative
			? rotation
			: rotation * Quaternion.Euler(0,
				  -LocomotionTeleport.LocomotionController.CameraRig.trackingSpace.localEulerAngles.y, 0);
	}
}
