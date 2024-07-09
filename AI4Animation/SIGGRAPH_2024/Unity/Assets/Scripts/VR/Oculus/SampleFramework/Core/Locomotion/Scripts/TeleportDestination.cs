/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using UnityEngine.Assertions;

/// <summary>
/// The teleport system uses a prefab with an TeleportDestination component, which will track the target location 
/// and update a MechAnim to reflect if the destination is valid. Since the user can switch quickly between valid 
/// targets, it might be possible for a destination to remain active in the scene until a deactivation animation is 
/// completed. To support this behavior efficiently, these prefabs are managed by a simple object pool. 
/// Target handlers that don’t rely on discrete locations like Nodes are likely to have only one TeleportDestination 
/// prefab active at a time, however it is possible that multiple teleports can occur with any target handler type 
/// and if the TeleportDestination’s MechAnim has a Post Teleport animation it might be necessary to instantiate 
/// additional OVRTeleportDestinations to allow targeting to occur before the previous destination finishes it’s animation.
/// </summary>
public class TeleportDestination : MonoBehaviour
{
	/// <summary>
	/// This can be used by MechAnim to change visuals accordingly.
	/// </summary>
	public bool IsValidDestination { get; private set; }

	[Tooltip("If the target handler provides a target position, this transform will be moved to that position and it's game object enabled. A target position being provided does not mean the position is valid, only that the aim handler found something to test as a destination.")]
	public Transform PositionIndicator;

	[Tooltip("This transform will be rotated to match the rotation of the aiming target. Simple teleport destinations should assign this to the object containing this component. More complex teleport destinations might assign this to a sub-object that is used to indicate the landing orientation independently from the rest of the destination indicator, such as when world space effects are required. This will typically be a child of the PositionIndicator.")]
	public Transform OrientationIndicator;

	[Tooltip("After the player teleports, the character controller will have it's rotation set to this value. It is different from the OrientationIndicator transform.rotation in order to support both head-relative and forward-facing teleport modes (See TeleportOrientationHandlerThumbstick.cs).")]
	public Quaternion LandingRotation;

	/// <summary>
	/// This will be set by the LocomotionTeleport when the object is instantiated.
	/// </summary>
	[NonSerialized] public LocomotionTeleport LocomotionTeleport;

	/// <summary>
	/// The LocomotionTeleport will update this only while this destination is active.
	/// </summary>
	[NonSerialized] public LocomotionTeleport.States TeleportState;

	private readonly Action<bool, Vector3?, Quaternion?, Quaternion?> _updateTeleportDestinationAction;
	private bool _eventsActive;

	TeleportDestination()
	{
		_updateTeleportDestinationAction = UpdateTeleportDestination;
	}

	public void OnEnable()
	{
		// Make sure the position and orientation indicators aren't enabled until the destination is updated, otherwise they will flicker at their current location for a frame.
		PositionIndicator.gameObject.SetActive(false);
		if (OrientationIndicator != null)
		{
			OrientationIndicator.gameObject.SetActive(false);
		}

		LocomotionTeleport.UpdateTeleportDestination += _updateTeleportDestinationAction;
		Assert.IsNotNull(OrientationIndicator);
		_eventsActive = true;
	}

	void TryDisableEventHandlers()
	{
		if (!_eventsActive)
		{
			return;
		}
		LocomotionTeleport.UpdateTeleportDestination -= _updateTeleportDestinationAction;
		_eventsActive = false;
	}

	public void OnDisable()
	{
		TryDisableEventHandlers();
	}

	/// <summary>
	/// If anything is handling the Deactivated event, then the handlers will be responsible for recycling the object when it is finished.
	/// </summary>
	public event Action<TeleportDestination> Deactivated;

	public void OnDeactivated()
	{
		if (Deactivated != null)
		{
			Deactivated(this);
		}
		else
		{
			Recycle();
		}
	}

	public void Recycle()
	{
		LocomotionTeleport.RecycleTeleportDestination(this);
	}

	public virtual void UpdateTeleportDestination(bool isValidDestination, Vector3? position, Quaternion? rotation, Quaternion? landingRotation)
	{
		//Debug.Log("Teleport Destination: "+ position + " " + rotation);

		IsValidDestination = isValidDestination;

		LandingRotation = landingRotation.GetValueOrDefault();

		// Show/hide the position indicator based on there being a valid position for it.
		var go = PositionIndicator.gameObject;
		var goActive = go.activeInHierarchy;
		if (!position.HasValue)
		{
			if (goActive)
			{
				go.SetActive(false);
			}
			return;
		}
		if (!goActive)
		{
			go.SetActive(true);
		}
		transform.position = position.GetValueOrDefault();

		// If there is no orientation indicator specified, rotate the entire indicator to match any provided orientation.
		if (OrientationIndicator == null)
		{
			if (rotation.HasValue)
			{
				transform.rotation = rotation.GetValueOrDefault();
			}
			return;
		}

		// Rotate the orientation indicator to match any valid rotation and make it visible,
		// otherwise hide it.
		var orientationGO = OrientationIndicator.gameObject;
		var ogoActive = orientationGO.activeInHierarchy;
		if (!rotation.HasValue)
		{
			if (ogoActive)
			{
				orientationGO.SetActive(false);
			}
			return;
		}

		OrientationIndicator.rotation = rotation.GetValueOrDefault();
		if (!ogoActive)
		{
			orientationGO.SetActive(true);
		}
	}
}
