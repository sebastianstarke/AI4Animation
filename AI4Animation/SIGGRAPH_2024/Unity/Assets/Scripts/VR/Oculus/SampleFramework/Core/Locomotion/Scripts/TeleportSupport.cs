/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

using UnityEngine;
using System.Collections;

/// <summary>
/// The TeleportSupport is an EventHandlerBehavior used by a number of derived behaviors 
/// which all rely on the a LocomotionTeleport component being present and available.
/// </summary>
public abstract class TeleportSupport : MonoBehaviour
{
	protected LocomotionTeleport LocomotionTeleport { get; private set; }

	/// <summary>
	/// This boolean is used to verify that OnEnable/OnDisable virtual functions always call
	/// the base class implementations to ensure event handlers are attached/detached correctly.
	/// </summary>
	private bool _eventsActive;

	/// <summary>
	/// OnEnable is virtual so that derived classes can set up references to objects before 
	/// AddEventHandlers is called, as is required by TeleportSupport derivations.
	/// </summary>
	protected virtual void OnEnable()
	{
		LocomotionTeleport = GetComponent<LocomotionTeleport>();
		Debug.Assert(!_eventsActive);
		AddEventHandlers();
		Debug.Assert(_eventsActive);
	}

	protected virtual void OnDisable()
	{
		Debug.Assert(_eventsActive);
		RemoveEventHandlers();
		Debug.Assert(!_eventsActive);
		LocomotionTeleport = null;
	}

	[System.Diagnostics.Conditional("DEBUG_TELEPORT_EVENT_HANDLERS")]
	void LogEventHandler(string msg)
	{
		Debug.Log("EventHandler: " + GetType().Name + ": " + msg);
	}

	/// <summary>
	/// Derived classes that need to use event handlers need to override this method and
	/// call the base class to ensure all event handlers are added as intended.
	/// </summary>
	protected virtual void AddEventHandlers()
	{
		LogEventHandler("Add");
		_eventsActive = true;
	}

	/// <summary>
	/// Derived classes that need to use event handlers need to override this method and
	/// call the base class to ensure all event handlers are removed as intended.
	/// </summary>
	protected virtual void RemoveEventHandlers()
	{
		LogEventHandler("Remove");
		_eventsActive = false;
	}
}
