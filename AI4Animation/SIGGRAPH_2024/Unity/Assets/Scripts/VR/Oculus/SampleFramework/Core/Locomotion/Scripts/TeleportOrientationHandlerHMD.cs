/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;

/// <summary>
/// This orientation handler will aim the player at the point they aim the HMD at after they choose the teleport location.
/// </summary>
public class TeleportOrientationHandlerHMD : TeleportOrientationHandler
{
	/// <summary>
	/// HeadRelative=Character will orient to match the arrow. ForwardFacing=When user orients to match the arrow, they will be facing the sensors.
	/// </summary>
	[Tooltip("HeadRelative=Character will orient to match the arrow. ForwardFacing=When user orients to match the arrow, they will be facing the sensors.")]
	public OrientationModes OrientationMode;

	/// <summary>
	/// Should the destination orientation be updated during the aim state in addition to the PreTeleport state?
	/// </summary>
	[Tooltip("Should the destination orientation be updated during the aim state in addition to the PreTeleport state?")]
	public bool UpdateOrientationDuringAim;

	/// <summary>
	/// How far from the destination must the HMD be pointing before using it for orientation
	/// </summary>
	[Tooltip("How far from the destination must the HMD be pointing before using it for orientation")]
	public float AimDistanceThreshold;

	/// <summary>
	/// How far from the destination must the HMD be pointing before rejecting the teleport
	/// </summary>
	[Tooltip("How far from the destination must the HMD be pointing before rejecting the teleport")]
	public float AimDistanceMaxRange;

	private Quaternion _initialRotation;

	protected override void InitializeTeleportDestination()
	{
	  _initialRotation = Quaternion.identity;
	}

	protected override void UpdateTeleportDestination()
	{
		// Only update the orientation during preteleport, or if configured to do updates during aim.
		if (AimData.Destination.HasValue && (UpdateOrientationDuringAim || LocomotionTeleport.CurrentState == LocomotionTeleport.States.PreTeleport))
		{
			var t = LocomotionTeleport.LocomotionController.CameraRig.centerEyeAnchor;
			var destination = AimData.Destination.GetValueOrDefault(); 

			// create a plane that contains the destination, with the normal pointing to the HMD.
			var plane = new Plane(Vector3.up, destination);

			// find the point on the plane that the HMD is looking at.
			float d;
			bool hit = plane.Raycast(new Ray(t.position, t.forward), out d);
			if (hit)
			{
				var target = t.position + t.forward * d;
				var local = target - destination;
				local.y = 0;
				var distance = local.magnitude;
				if (distance > AimDistanceThreshold)
				{
					local.Normalize();

					// Some debug draw code to visualize what the math is doing.

					//OVRDebugDraw.AddCross(target, 0.2f, 0.01f, Color.yellow, 0.1f);
					//OVRDebugDraw.AddCross(destination + new Vector3(local.x, 0, local.z), 0.2f, 0.01f, Color.blue, 0.1f);

					//OVRDebugDraw.AddLine(t.position + new Vector3(0, 0.1f, 0), target, 0.01f, Color.yellow, 1.0f);
					//OVRDebugDraw.AddLine(target + new Vector3(0, 1f, 0), target - new Vector3(0, 1f, 0), 0.01f, Color.blue, 1.0f);

					var rot = Quaternion.LookRotation(new Vector3(local.x, 0, local.z), Vector3.up);
					_initialRotation = rot;

					if (AimDistanceMaxRange > 0 && distance > AimDistanceMaxRange)
					{
						AimData.TargetValid = false;
					}

					LocomotionTeleport.OnUpdateTeleportDestination(AimData.TargetValid, AimData.Destination, rot, GetLandingOrientation(OrientationMode, rot));
					return;
				}
			}
		}
		LocomotionTeleport.OnUpdateTeleportDestination(AimData.TargetValid, AimData.Destination, _initialRotation, GetLandingOrientation(OrientationMode, _initialRotation));
	}
}
