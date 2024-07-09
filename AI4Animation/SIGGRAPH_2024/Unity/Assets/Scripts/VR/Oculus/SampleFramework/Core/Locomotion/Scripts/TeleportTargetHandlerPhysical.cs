/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using System;
using UnityEngine;
using System.Collections;
using UnityEngine.EventSystems;

/// <summary>
/// This target handler simply returns any location that is detected by the aim collision tests.
/// Essentially, any space the player will fit will be a valid teleport destination.
/// </summary>
public class TeleportTargetHandlerPhysical : TeleportTargetHandler
{
	/// <summary>
	/// This method will be called while the LocmotionTeleport component is in the aiming state, once for each
	/// line segment that the targeting beam requires. 
	/// The function should return true whenever an actual target location has been selected.
	/// </summary>
	/// <param name="start"></param>
	/// <param name="end"></param>
	protected override bool ConsiderTeleport(Vector3 start, ref Vector3 end)
	{
		// If the ray hits the world, consider it valid and update the aimRay to the end point.
		if (LocomotionTeleport.AimCollisionTest(start, end, AimCollisionLayerMask, out AimData.TargetHitInfo))
		{
			var d = (end - start).normalized;

			end = start + d * AimData.TargetHitInfo.distance;
			return true;
		}
		return false;
	}
}
