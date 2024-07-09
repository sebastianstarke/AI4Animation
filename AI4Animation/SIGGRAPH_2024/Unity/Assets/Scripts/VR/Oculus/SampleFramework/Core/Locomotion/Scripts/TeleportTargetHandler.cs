/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

#define DEBUG_TELEPORT_EVENT_HANDLERS

using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using Debug = UnityEngine.Debug;

/// <summary>
/// The TeleportTargetHandler's main purpose is to determine when the current aim target is valid
/// and to update the teleport destination as required by the design. This allows specialized versions
/// that can simply update the destination to any arbitrary location, or update only when a teleport node
/// is being pointed at, or anything else that fits the design for limiting when & where a teleport is valid.
/// </summary>
public abstract class TeleportTargetHandler : TeleportSupport
{
	/// <summary>
	/// This bitmask controls which game object layers will be included in the targeting collision tests.
	/// </summary>
	[Tooltip("This bitmask controls which game object layers will be included in the targeting collision tests.")]
	public LayerMask AimCollisionLayerMask;

	protected readonly LocomotionTeleport.AimData AimData = new LocomotionTeleport.AimData();
	private readonly Action _startAimAction;

	protected TeleportTargetHandler()
	{
		_startAimAction = () => { StartCoroutine(TargetAimCoroutine()); };
	}
	
	protected override void AddEventHandlers()
	{
		base.AddEventHandlers();
		LocomotionTeleport.EnterStateAim += _startAimAction;
	}

	protected override void RemoveEventHandlers()
	{
		base.RemoveEventHandlers();
		LocomotionTeleport.EnterStateAim -= _startAimAction;
	}

	private readonly List<Vector3> _aimPoints = new List<Vector3>();

	/// <summary>
	/// This coroutine is active while the teleport system is in the aiming state.
	/// </summary>
	/// <returns></returns>
	private IEnumerator TargetAimCoroutine()
	{
		// While the teleport system is in the aim state, perform the aim logic and consider teleporting.
		while (LocomotionTeleport.CurrentState == LocomotionTeleport.States.Aim)
		{
			// With each targeting test, we need to reset the AimData to clear the point list and reset flags.
			ResetAimData();

			// Start the testing with the character's current position to the aiming origin to ensure they 
			// haven't just stuck their hand through something that should have prevented movement.
			//
			// The first test won't be added to the aim data results because the visual effects should be from
			// the aiming origin.

			var current = LocomotionTeleport.transform.position;

			// Enumerate through all the line segments provided by the aim handler, checking for a valid target on each segment,
			// stopping at the first valid target or when the enumerable runs out of line segments.
			_aimPoints.Clear();
			LocomotionTeleport.AimHandler.GetPoints(_aimPoints);

			for(int i = 0; i < _aimPoints.Count; i++)
			{
				var adjustedPoint = _aimPoints[i];
				AimData.TargetValid = ConsiderTeleport(current, ref adjustedPoint);
				AimData.Points.Add(adjustedPoint);
				if (AimData.TargetValid)
				{
					AimData.Destination = ConsiderDestination(adjustedPoint);
					AimData.TargetValid = AimData.Destination.HasValue;
					break;
				}
				current = _aimPoints[i];
			}
			LocomotionTeleport.OnUpdateAimData(AimData);
			yield return null;
		}
	}

	/// When a parabolic or other aiming method that consists of many line segments is being used, ConsiderTeleport 
	/// will be called once for each segment so if there is expensive work that can be cached in advance,
	/// override the ResetAimData method to prepare that data.
	protected virtual void ResetAimData()
	{
		AimData.Reset();
	}

	/// <summary>
	/// This method will be called while the LocmotionTeleport component is in the aiming state, once for each
	/// line segment that the targeting beam requires. 
	/// The function should return true whenever an actual target location has been selected.
	/// </summary>
	/// <param name="start"></param>
	/// <param name="end"></param>
	protected abstract bool ConsiderTeleport(Vector3 start, ref Vector3 end);

	const float ERROR_MARGIN = 0.1f;

	/// <summary>
	/// Adjust the provided located to account for character height and perform any checks that might
	/// invalidate the target position of the character controller, such as collision with scene geometry
	/// or other actors.
	/// </summary>
	/// <param name="location"></param>
	/// <returns></returns>
	public virtual Vector3? ConsiderDestination(Vector3 location)
	{
		var character = LocomotionTeleport.LocomotionController.CharacterController;
		var radius = character.radius - ERROR_MARGIN;

		var start = location;
		start.y += radius + ERROR_MARGIN;
		var end = start;
		end.y += character.height - ERROR_MARGIN;

		var result = Physics.CheckCapsule(start, end, radius,
			AimCollisionLayerMask, QueryTriggerInteraction.Ignore);
		if (result)
		{
			return null;
		}
		return location; 
	}
}
