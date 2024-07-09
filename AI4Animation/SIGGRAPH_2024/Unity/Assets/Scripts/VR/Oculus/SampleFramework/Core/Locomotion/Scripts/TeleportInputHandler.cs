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
/// The TeleportInputHandler provides interfaces used to control aim related to teleports and related behaviors.
/// There are derived implementations of this for Touch controllers, gamepad and HMD based aiming mechanics. 
/// Supporting any of these, or other future controllers, is possible by implementing and enabling a different 
/// derived type of TeleportInputHandler.
/// </summary>
public abstract class TeleportInputHandler : TeleportSupport
{
	private readonly Action _startReadyAction;
	private readonly Action _startAimAction;

	protected TeleportInputHandler()
	{
		_startReadyAction = () => { StartCoroutine(TeleportReadyCoroutine()); };
		_startAimAction = () => { StartCoroutine(TeleportAimCoroutine()); };
	}

	protected override void AddEventHandlers()
	{
		LocomotionTeleport.InputHandler = this;
		base.AddEventHandlers();
		LocomotionTeleport.EnterStateReady += _startReadyAction;
		LocomotionTeleport.EnterStateAim += _startAimAction;
	}

	protected override void RemoveEventHandlers()
	{
		if(LocomotionTeleport.InputHandler == this)
		{
			LocomotionTeleport.InputHandler = null;
		}
		LocomotionTeleport.EnterStateReady -= _startReadyAction;
		LocomotionTeleport.EnterStateAim -= _startAimAction;
		base.RemoveEventHandlers();
	}

	/// <summary>
	/// This coroutine will be active while the teleport system is in the Ready state.
	/// </summary>
	/// <returns></returns>
	IEnumerator TeleportReadyCoroutine()
	{
		while (GetIntention() != LocomotionTeleport.TeleportIntentions.Aim)
		{
			yield return null;
		}
		LocomotionTeleport.CurrentIntention = LocomotionTeleport.TeleportIntentions.Aim;
	}

	/// <summary>
	/// This coroutine will be active while the teleport system is in the Aim or PreTeleport state.
	/// It remains active in both the Aim and PreTeleport states because these states are the ones that 
	/// need to switch to different states based on the user intention as detected by the input handler.
	/// </summary>
	/// <returns></returns>
	IEnumerator TeleportAimCoroutine()
	{
		LocomotionTeleport.TeleportIntentions intention = GetIntention();

		while (intention == LocomotionTeleport.TeleportIntentions.Aim || intention == LocomotionTeleport.TeleportIntentions.PreTeleport)
		{
			LocomotionTeleport.CurrentIntention = intention;
			yield return null;
			intention = GetIntention();
		}
		LocomotionTeleport.CurrentIntention = intention;
	}

	/// <summary>
	/// One of the core functions of the TeleportInputHandler is to notify the LocomotionTeleport of the current intentions of the 
	/// user with respect to aiming, teleporting, and abandoning a pending teleport. 
	/// Derivations of this class will check buttons or whatever inputs they require to return values indicating what the user is
	/// trying to do.
	/// </summary>
	/// <returns></returns>
	public abstract LocomotionTeleport.TeleportIntentions GetIntention();

	/// <summary>
	/// Returns the aim ray for pointing at targets, which is generally based on a touch controller or HMD pose.
	/// </summary>
	public abstract void GetAimData(out Ray aimRay);
}
