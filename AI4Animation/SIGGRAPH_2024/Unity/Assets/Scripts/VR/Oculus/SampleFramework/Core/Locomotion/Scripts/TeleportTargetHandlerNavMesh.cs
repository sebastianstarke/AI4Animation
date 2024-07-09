/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

// Enable this define to visualize the navigation solution that was used to validate access to the target location.
//#define SHOW_PATH_RESULT

using UnityEngine;
using System.Collections;
using System.Diagnostics;

public class TeleportTargetHandlerNavMesh : TeleportTargetHandler
{
	/// <summary>
	/// Controls which areas are to be used when doing nav mesh queries.
	/// </summary>
	public int NavMeshAreaMask = UnityEngine.AI.NavMesh.AllAreas;

	/// <summary>
	/// A NavMeshPath that is necessary for doing pathing queries and is reused with each request.
	/// </summary>
	private UnityEngine.AI.NavMeshPath _path;

	void Awake()
	{
		_path = new UnityEngine.AI.NavMeshPath();
	}

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

	/// <summary>
	/// This version of ConsiderDestination will only return a valid location if the pathing system is able to find a route 
	/// from the current position to the candidate location.
	/// </summary>
	/// <param name="location"></param>
	/// <returns></returns>
	public override Vector3? ConsiderDestination(Vector3 location)
	{
		var result = base.ConsiderDestination(location);
		if (result.HasValue)
		{
            Vector3 start = LocomotionTeleport.GetCharacterPosition();
            Vector3 dest = result.GetValueOrDefault();
            UnityEngine.AI.NavMesh.CalculatePath(start, dest, NavMeshAreaMask, _path);
                
			if (_path.status == UnityEngine.AI.NavMeshPathStatus.PathComplete)
			{
				return result;
			}
		}
		return null;
	}

	[Conditional("SHOW_PATH_RESULT")]
	private void OnDrawGizmos()
	{
#if SHOW_PATH_RESULT
		if (_path == null)
			return;

		var corners = _path.corners;
		if (corners == null || corners.Length == 0)
			return;
		var p = corners[0];
		for(int i = 1; i < corners.Length; i++)
		{
			var p2 = corners[i];
			Gizmos.DrawLine(p, p2);
			p = p2;
		}
#endif
	}
}
