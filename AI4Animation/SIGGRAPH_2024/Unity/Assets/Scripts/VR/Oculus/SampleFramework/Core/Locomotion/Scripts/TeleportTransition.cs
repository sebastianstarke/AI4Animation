/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// Teleport transitions manage the actual relocation of the player from the current position and orientation
/// to the teleport destination. 
/// All teleport transition behaviors derive from this class, primarily for type safety 
/// within the LocomotionTeleport to track the current transition type.
/// </summary>
public abstract class TeleportTransition : TeleportSupport
{
	protected override void AddEventHandlers()
	{
		LocomotionTeleport.EnterStateTeleporting += LocomotionTeleportOnEnterStateTeleporting;
		base.AddEventHandlers();
	}

	protected override void RemoveEventHandlers()
	{
		LocomotionTeleport.EnterStateTeleporting -= LocomotionTeleportOnEnterStateTeleporting;
		base.RemoveEventHandlers();
	}

	/// <summary>
	/// When the teleport state is entered, simply move the player to the new location
	/// without any delay or other side effects.
	/// If the transition is not immediate, the transition handler will need to set the LocomotionTeleport.IsTeleporting 
	/// to true for the duration of the transition, setting it to false when the transition is finished which will
	/// then allow the teleport state machine to switch to the PostTeleport state.
	/// </summary>
	protected abstract void LocomotionTeleportOnEnterStateTeleporting();
}
