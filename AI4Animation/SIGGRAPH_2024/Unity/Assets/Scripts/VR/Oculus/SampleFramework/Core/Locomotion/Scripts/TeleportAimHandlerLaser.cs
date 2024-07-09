/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;
using System.Collections.Generic;

public class TeleportAimHandlerLaser : TeleportAimHandler
{
	/// <summary>
	/// Maximum range for aiming.
	/// </summary>
	[Tooltip("Maximum range for aiming.")]
	public float Range = 100;

	/// <summary>
	/// Return the set of points that represent the aiming line.
	/// </summary>
	/// <param name="points"></param>
	public override void GetPoints(List<Vector3> points)
	{
		Ray aimRay;
		LocomotionTeleport.InputHandler.GetAimData(out aimRay);
		points.Add(aimRay.origin);
		points.Add(aimRay.origin + aimRay.direction * Range);
	}
}
