/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// This orientation handler doesn't actually do anything with the orientation at all; this is for users
/// who have a 360 setup and don't need to be concerned with choosing an orientation because they just
/// turn whatever direction they want.
/// </summary>
public class TeleportOrientationHandler360 : TeleportOrientationHandler
{
	protected override void InitializeTeleportDestination()
	{
	}

	protected override void UpdateTeleportDestination()
	{
		LocomotionTeleport.OnUpdateTeleportDestination(AimData.TargetValid, AimData.Destination, null, null);
	}
}
