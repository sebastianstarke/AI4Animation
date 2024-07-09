/************************************************************************************

See SampleFramework license.txt for license terms.  Unless required by applicable law 
or agreed to in writing, the sample code is provided “AS IS” WITHOUT WARRANTIES OR 
CONDITIONS OF ANY KIND, either express or implied.  See the license for specific 
language governing permissions and limitations under the license.

************************************************************************************/

// Enable DEBUG_TELEPORT_STATES to cause messages to be logged when teleport state changes occur.
//#define DEBUG_TELEPORT_STATES

using System;
using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine.EventSystems;
using Debug = UnityEngine.Debug;


/// <summary>
/// The LocomotionTeleport class controls and centralizes functionality for the various types 
/// of teleports. The system is designed to work as a set of components that are each responsible 
/// for different aspects of the teleport process. This makes it possible for different kinds of 
/// teleport behaviors to be occur by simply enabling different combinations of components.
/// </summary>
public class LocomotionTeleport : MonoBehaviour
{
	/// <summary>
	/// The process of teleporting is represented by a simple state machine, and each of the 
	/// possible states are represented by this enum.
	/// </summary>
	public enum States
	{
		Ready,
		Aim,
		CancelAim,
		PreTeleport,
		CancelTeleport,
		Teleporting,
		PostTeleport
	}

	#region Linear movement control booleans.
	/// <summary>
	/// Allow linear movement prior to the teleport system being activated.
	/// </summary>
	[Tooltip("Allow linear movement prior to the teleport system being activated.")]
	public bool EnableMovementDuringReady = true;

	/// <summary>
	/// Allow linear movement while the teleport system is in the process of aiming for a teleport target.
	/// </summary>
	[Tooltip("Allow linear movement while the teleport system is in the process of aiming for a teleport target.")]
	public bool EnableMovementDuringAim = true;

	/// <summary>
	/// Allow linear movement while the teleport system is in the process of configuring the landing orientation.
	/// </summary>
	[Tooltip("Allow linear movement while the teleport system is in the process of configuring the landing orientation.")]
	public bool EnableMovementDuringPreTeleport = true;

	/// <summary>
	/// Allow linear movement after the teleport has occurred but before the system has returned to the ready state.
	/// </summary>
	[Tooltip("Allow linear movement after the teleport has occurred but before the system has returned to the ready state.")]
	public bool EnableMovementDuringPostTeleport = true;

	/// <summary>
	/// Helper function to enable linear movement during the various teleport states.
	/// Movement may not be desired at all states, for instance during aiming it may be preferred to prevent movement to allow a 
	/// thumbstick to be used for choosing the landing orientation.
	/// </summary>
	/// <param name="ready"></param>
	/// <param name="aim"></param>
	/// <param name="pre"></param>
	/// <param name="post"></param>
	public void EnableMovement(bool ready, bool aim, bool pre, bool post)
	{
		EnableMovementDuringReady = ready;
		EnableMovementDuringAim = aim;
		EnableMovementDuringPreTeleport = pre;
		EnableMovementDuringPostTeleport = post;
	}
	#endregion

	#region Rotation control booleans.
	/// <summary>
	/// Allow rotation prior to the teleport system being activated.
	/// </summary>
	[Tooltip("Allow rotation prior to the teleport system being activated.")]
	public bool EnableRotationDuringReady = true;

	/// <summary>
	/// Allow rotation while the teleport system is in the process of aiming for a teleport target.
	/// </summary>
	[Tooltip("Allow rotation while the teleport system is in the process of aiming for a teleport target.")]
	public bool EnableRotationDuringAim = true;

	/// <summary>
	/// Allow rotation while the teleport system is in the process of configuring the landing orientation.
	/// </summary>
	[Tooltip("Allow rotation while the teleport system is in the process of configuring the landing orientation.")]
	public bool EnableRotationDuringPreTeleport = true;

	/// <summary>
	/// Allow rotation after the teleport has occurred but before the system has returned to the ready state.
	/// </summary>
	[Tooltip("Allow rotation after the teleport has occurred but before the system has returned to the ready state.")]
	public bool EnableRotationDuringPostTeleport = true;

	/// <summary>
	/// Helper function to enable rotation movement during the various teleport states.
	/// Rotation may not be desired at all states, for instance during aiming it may be preferred to prevent rotation (snap turn or linear) 
	/// to prevent the camera from being rotated while preparing to teleport.
	/// </summary>
	/// <param name="ready"></param>
	/// <param name="aim"></param>
	/// <param name="pre"></param>
	/// <param name="post"></param>
	public void EnableRotation(bool ready, bool aim, bool pre, bool post)
	{
		EnableRotationDuringReady = ready;
		EnableRotationDuringAim = aim;
		EnableRotationDuringPreTeleport = pre;
		EnableRotationDuringPostTeleport = post;
	}
	#endregion

	/// <summary>
	/// The current state of the teleport state machine.
	/// </summary>
	public States CurrentState { get; private set; }

	/// <summary>
	/// Aiming is handled by one specific aim handler at a time. When the aim handler component is enabled, it 
	/// will set this reference to the AimHandler so that other parts of the system which need access to the 
	/// current aim handler can be sure to use the correct component. 
	/// </summary>
	[NonSerialized]
	public TeleportAimHandler AimHandler;

	/// <summary>
	/// This prefab will be instantiated as needed and updated to match the current aim target.
	/// </summary>
	[Tooltip("This prefab will be instantiated as needed and updated to match the current aim target.")]
	public TeleportDestination TeleportDestinationPrefab;
	[Tooltip("TeleportDestinationPrefab will be instantiated into this layer.")]
	public int TeleportDestinationLayer = 0;
	
	#region Support Events
	/// <summary>
	/// This event is raised when the teleport destination is in the process of being updated. It is used by the active TeleportDestination
	/// to update it's visual state, position and orientation indicator to match the results of the teleport aim and targeting system.
	/// </summary>
	public event Action<bool, Vector3?, Quaternion?, Quaternion?> UpdateTeleportDestination;

	/// <summary>
	/// When the active aim and orientation handler finishes preparing the data for the teleport destination, this method will be called
	/// in order to raise the UpdateTeleportDestination event, which will in turn give any active teleport destination objects an opportunity
	/// to update their visual state accordingly.
	/// </summary>
	/// <param name="isValidDestination"></param>
	/// <param name="position"></param>
	/// <param name="rotation"></param>
	public void OnUpdateTeleportDestination(bool isValidDestination, Vector3? position, Quaternion? rotation, Quaternion? landingRotation)
	{
		if (UpdateTeleportDestination != null)
		{
			UpdateTeleportDestination(isValidDestination, position, rotation, landingRotation);
		}
	}

	/// <summary>
	/// The TeleportInputHandler is responsible for converting input events to TeleportIntentions. 
	/// </summary>
	[NonSerialized]
	public TeleportInputHandler InputHandler;

	/// <summary>
	/// TeleportIntentions track what the TeleportState should attempt to transition to.
	/// </summary>
	public enum TeleportIntentions
	{
		None,           // No teleport is requested.
		Aim,            // The user wants to aim for a teleport. 
		PreTeleport,    // The user has selected a location to teleport, and the input handler will now control how long it stays in PreTeleport.
		Teleport        // The user has chosen to teleport. If the destination is valid, the state will transition to Teleporting, otherwise it will switch to CancelTeleport.
	}

	/// <summary>
	/// The CurrentIntention is used by the state machine to know when it is time to switch to a new state.
	/// </summary>
	[NonSerialized]
	public TeleportIntentions CurrentIntention;

	/// <summary>
	/// The state machine will not exit the PreTeleport state while IsPreTeleportRequested is true.
	/// The sample doesn't currently use this, however this provides new components the ability to delay exiting 
	/// the PreTeleport state until game logic is ready for it.
	/// </summary>
	[NonSerialized]
	public bool IsPreTeleportRequested;

	/// <summary>
	/// The state machine will not exit the Teleporting state while IsTransitioning is true. This is how the BlinkTransition and WarpTransition 
	/// force the system to remain in the Teleporting state until the transition is complete.
	/// </summary>
	[NonSerialized]
	public bool IsTransitioning;

	/// <summary>
	/// The state machine will not exit the PostTeleport state while IsPostTeleportRequested is true. 
	/// The sample doesn't currently use this, however this provides new components the ability to delay exiting 
	/// the PostTeleport state until game logic is ready for it.
	/// </summary>
	[NonSerialized]
	public bool IsPostTeleportRequested;

	/// <summary>
	/// Created at runtime, this gameobject is used to track where the player will teleport. 
	/// The actual position depends on the type of Aim Handler and Target Handler that is active.
	/// Aim Handlers:
	/// * Laser: player capsule swept along aim ray until it hits terrain or valid target.
	/// * Parabolic: player capsule swept along a series of line segments approximating a parabolic curve until it hits terrain or valid target.
	/// Target Handlers:
	/// * NavMesh = Destination only valid if it lies within the nav mesh.
	/// * Node = Destination valid if within radius of a teleport node. Target is invalidated when aim leaves the node radius.
	/// * Physical = Any terrain is valid.
	/// </summary>
	private TeleportDestination _teleportDestination;

	/// <summary>
	/// Returns the orientation of the current teleport destination's orientation indicator.
	/// </summary>
	public Quaternion DestinationRotation
	{
		get { return _teleportDestination.OrientationIndicator.rotation; }
	}

	#endregion

	/// <summary>
	/// The LocomotionController that is used by object to discover shared references.
	/// </summary>
	public LocomotionController LocomotionController { get; private set; }

	/// <summary>
	/// The aiming system uses a common function for testing collision with the world, which can be configured to use different
	/// shapes for testing. 
	/// </summary>
	public enum AimCollisionTypes
	{
		Point,  // ray casting
		Sphere, // swept sphere test
		Capsule // swept capsule test, optionally sized to match the character controller dimensions. 
	}

	/// <summary>
	/// When aiming at possible destinations, the aim collision type determines which shape to use for collision tests.
	/// </summary>
	[Tooltip("When aiming at possible destinations, the aim collision type determines which shape to use for collision tests.")]
	public AimCollisionTypes AimCollisionType;

	/// <summary>
	/// Use the character collision radius/height/skinwidth for sphere/capsule collision tests.
	/// </summary>
	[Tooltip("Use the character collision radius/height/skinwidth for sphere/capsule collision tests.")]
	public bool UseCharacterCollisionData;

	/// <summary>
	/// Radius of the sphere or capsule used for collision testing when aiming to possible teleport destinations. Ignored if UseCharacterCollisionData is true.
	/// </summary>
	[Tooltip("Radius of the sphere or capsule used for collision testing when aiming to possible teleport destinations. Ignored if UseCharacterCollisionData is true.")]
	public float AimCollisionRadius;

	/// <summary>
	/// Height of the capsule used for collision testing when aiming to possible teleport destinations. Ignored if UseCharacterCollisionData is true.
	/// </summary>
	[Tooltip("Height of the capsule used for collision testing when aiming to possible teleport destinations. Ignored if UseCharacterCollisionData is true.")]
	public float AimCollisionHeight;

	/// <summary>
	/// AimCollisionTest is used by many of the aim handlers to standardize the testing of aiming beams. By choosing between the increasingly restrictive 
	/// point, sphere and capsule tests, the aiming system can limit targeting to routes which are not physically blocked. For example, a sphere test 
	/// is good for ensuring the player can't teleport through bars to get out of a jail cell. 
	/// </summary>
	/// <param name="start"></param>
	/// <param name="end"></param>
	/// <param name="aimCollisionLayerMask"></param>
	/// <param name="hitInfo"></param>
	/// <returns></returns>
	public bool AimCollisionTest(Vector3 start, Vector3 end, LayerMask aimCollisionLayerMask, out RaycastHit hitInfo)
	{
		var delta = end - start;
		var distance = delta.magnitude;
		var direction = delta / distance;

		switch (AimCollisionType)
		{
			case AimCollisionTypes.Capsule:
			{
				float r, h;
				if (UseCharacterCollisionData)
				{
					var c = LocomotionController.CharacterController;
					h = c.height;
					r = c.radius;
				}
				else
				{
					h = AimCollisionHeight;
					r = AimCollisionRadius;
				}
				return Physics.CapsuleCast(start + new Vector3(0, r, 0),
					start + new Vector3(0, h + r, 0), r, direction,
					out hitInfo, distance, aimCollisionLayerMask, QueryTriggerInteraction.Ignore);
			}

			case AimCollisionTypes.Point:
				return Physics.Raycast(start, direction, out hitInfo, distance, aimCollisionLayerMask,QueryTriggerInteraction.Ignore);

			case AimCollisionTypes.Sphere:
			{
				float r;
				if (UseCharacterCollisionData)
				{
					var c = LocomotionController.CharacterController;
					//r = c.radius - c.skinWidth;
					r = c.radius;
				}
				else
				{
					r = AimCollisionRadius;
				}
				return Physics.SphereCast(start, r, direction, out hitInfo, distance, aimCollisionLayerMask,
					QueryTriggerInteraction.Ignore);
			}
		}

		// App should never get here.
		throw new Exception();
	}


	/// <summary>
	/// Internal logging function that is conditionally enabled via DEBUG_TELEPORT_STATES
	/// </summary>
	/// <param name="msg"></param>
	[Conditional("DEBUG_TELEPORT_STATES")]
	protected void LogState(string msg)
	{
		Debug.Log(Time.frameCount + ": " +  msg);
	}

	/// <summary>
	/// This is called whenever a new teleport destination is required. This might occur when rapidly switching
	/// between targets, or when teleporting multiple times in quick succession when the teleport destination
	/// requires additional time to complete any animations that are triggered by these actions.
	/// </summary>
	protected void CreateNewTeleportDestination()
	{
		TeleportDestinationPrefab.gameObject.SetActive(false); // ensure the prefab isn't active in order to delay event handler setup until after it has been configured with a reference to this object.
		TeleportDestination td = GameObject.Instantiate(TeleportDestinationPrefab);
		td.LocomotionTeleport = this;
		td.gameObject.layer = TeleportDestinationLayer;
		_teleportDestination = td;
		_teleportDestination.LocomotionTeleport = this;
	}

	/// <summary>
	/// Notify the teleport destination that it needs to deactivate.
	/// If the destination has event handlers hooked up, the destination game object may not be immediately deactivated
	/// in order to allow it to trigger animated effects.
	/// </summary>
	private void DeactivateDestination()
	{
		_teleportDestination.OnDeactivated();
	}

	public void RecycleTeleportDestination(TeleportDestination oldDestination)
	{
		if (oldDestination == _teleportDestination)
		{
			CreateNewTeleportDestination();
		}
		GameObject.Destroy(oldDestination.gameObject);	
	}

	/// <summary>
	/// Each state has booleans that determine if linear motion or rotational motion is enabled for that state.
	/// This method is called when entering each state with the appropriate values.
	/// </summary>
	/// <param name="enableLinear"></param>
	/// <param name="enableRotation"></param>
	private void EnableMotion(bool enableLinear, bool enableRotation)
	{
		LocomotionController.PlayerController.EnableLinearMovement = enableLinear;
		LocomotionController.PlayerController.EnableRotation = enableRotation;
	}

	/// <summary>
	/// When the component first wakes up, cache the LocomotionController and the initial
	/// TeleportDestination object.
	/// </summary>
	private void Awake()
	{
		LocomotionController = GetComponent<LocomotionController>();
		CreateNewTeleportDestination();
	}

	/// <summary>
	/// Start the state machine coroutines.
	/// </summary>
	public virtual void OnEnable ()
	{
		CurrentState = States.Ready;
		StartCoroutine(ReadyStateCoroutine());
	}
	public virtual void OnDisable ()
	{
		StopAllCoroutines();
	}

	/// <summary>
	/// This event is raised when entering the Ready state. The initial use for this is for the input handler to start 
	/// processing input in order to eventually set the TeleportIntention to Aim when the user requests it.
	/// </summary>
	public event Action EnterStateReady;

	/// <summary>
	/// This coroutine will be running while the component is in the Ready state.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator ReadyStateCoroutine()
	{
		LogState("ReadyState: Start");

		// yield once so that all the components will have time to process their OnEnable message before this 
		// does work that relies on the events being hooked up.
		yield return null;

		LogState("ReadyState: Ready");

		CurrentState = States.Ready;
		EnableMotion(EnableMovementDuringReady, EnableRotationDuringReady);

		if (EnterStateReady != null)
		{
			EnterStateReady();
		}

		// Wait until a teleport is requested.
		while (CurrentIntention != TeleportIntentions.Aim)
		{
			yield return null;
		}
		LogState("ReadyState: End");

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		StartCoroutine(AimStateCoroutine());
	}

	/// <summary>
	/// The AimData contains data provided by the Aim Handler which represents the final set of points
	/// that were used for aiming the teleport. This is provided to the AimVisual for rendering an aim effect.
	/// Note that the set of points provided here can be different from the points used by the Aim Handler to 
	/// determine the teleport destination. For instance, the aim handler might use a very long line segment
	/// for an aim laser but would provide a shorter line segment in the AimData representing the line
	/// from the player to the teleport destination.
	/// </summary>
	public class AimData
	{
		public AimData()
		{
			Points = new List<Vector3>();
		}

		public RaycastHit TargetHitInfo;
		public bool TargetValid;
		public Vector3? Destination;
		public float Radius;

		public List<Vector3> Points { get; private set; }

		public void Reset()
		{
			Points.Clear();
			TargetValid = false;
			Destination = null;
		}
	}

	/// <summary>
	/// This event is raised when the user begins aiming for a target location for a teleport.
	/// </summary>
	public event Action EnterStateAim;

	/// <summary>
	/// Aim and Target handlers are responsible for populating the AimData with the relevant aim data,
	/// which is used for a number of purposes within the teleport system.
	/// </summary>
	public event Action<AimData> UpdateAimData;

	/// <summary>
	/// The target handler will call this method when the aim data has been updated and is ready to be
	/// processed by anything that needs to be aware of any changes. This generally includes a visual
	/// indicator for the aiming and the active orientation handler.
	/// </summary>
	/// <param name="aimData"></param>
	public void OnUpdateAimData(AimData aimData)
	{
		if (UpdateAimData != null)
		{
			UpdateAimData(aimData);
		}
	}

	/// <summary>
	/// This event is raised when the aim state is exited. This is typically used by aim visualizers to
	/// deactivate any visual effects related to aiming.
	/// </summary>
	public event Action ExitStateAim;

	/// <summary>
	/// This coroutine will be running while the aim state is active. The teleport destination will become active,
	/// and depending on the target and current intention of the user it might enter the CancelAim state or 
	/// PreTeleport state when it is done.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator AimStateCoroutine()
	{
		LogState("AimState: Start");
		CurrentState = States.Aim;
		EnableMotion(EnableMovementDuringAim, EnableRotationDuringAim);
		if (EnterStateAim != null)
		{
			EnterStateAim();
		}
		_teleportDestination.gameObject.SetActive(true);

		// Wait until the user is done aiming. The input system will turn this off when the button that triggered aiming is released.
		while (CurrentIntention == TeleportIntentions.Aim) 
		{
			yield return null;
		}

		LogState("AimState: End. Intention: " + CurrentIntention);
		if (ExitStateAim != null)
		{
			ExitStateAim();
		}

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		// If target is valid, enter pre-teleport otherwise cancel the teleport.
		LogState("AimState: Switch state. Intention: " + CurrentIntention);
		if ((CurrentIntention == TeleportIntentions.PreTeleport || CurrentIntention == TeleportIntentions.Teleport) && _teleportDestination.IsValidDestination)
		{
			StartCoroutine(PreTeleportStateCoroutine());
		}
		else
		{
			StartCoroutine(CancelAimStateCoroutine());
		}
	}

	/// <summary>
	/// This event is raised when aiming for a teleport destination is aborted. It can be
	/// useful for cleaning up effects that may have been triggered when entering the Aim state.
	/// </summary>
	public event Action EnterStateCancelAim;

	/// <summary>
	/// This coroutine will be executed when the aim state is cancelled.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator CancelAimStateCoroutine()
	{
		LogState("CancelAimState: Start");
		CurrentState = States.CancelAim;
		if (EnterStateCancelAim != null)
		{
			EnterStateCancelAim();
		}
		LogState("CancelAimState: End");

		DeactivateDestination();

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		StartCoroutine(ReadyStateCoroutine());
	}

	/// <summary>
	/// This event is raised when the system enteres the PreTeleport state.
	/// </summary>
	public event Action EnterStatePreTeleport;

	/// <summary>
	/// This coroutine will be active while the system is in the PreTeleport state.
	/// At this point, the user has indicated they want to teleport however there is a possibility that the
	/// target they have chosen might be or become invalid so the next state will be either Teleporting or 
	/// CancelTeleporting.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator PreTeleportStateCoroutine()
	{
		LogState("PreTeleportState: Start");
		CurrentState = States.PreTeleport;
		EnableMotion(EnableMovementDuringPreTeleport, EnableRotationDuringPreTeleport);
		if (EnterStatePreTeleport != null)
		{
			EnterStatePreTeleport();
		}

		while (CurrentIntention == TeleportIntentions.PreTeleport || IsPreTeleportRequested)
		{
			yield return null;
		}

		LogState("PreTeleportState: End");

		// Most of the state coroutines will wait until the next frame to proceed to the next state's coroutine,
		// however the PreTeleportState may need to be processed quickly for situations where the teleport needs
		// to occur on the downpress of a button (AvatarTouch capactive touch for aim and teleport, for instance).

		if (_teleportDestination.IsValidDestination)
		{
			StartCoroutine(TeleportingStateCoroutine());
		}
		else
		{
			StartCoroutine(CancelTeleportStateCoroutine());
		}
	}

	/// <summary>
	/// This event will occur if the user cancels the teleport after the destination has been selected.
	/// Typically not much different than cancelling an aim state, however there may be some effect
	/// triggered by the target selection which needs to be cleaned up, or perhaps a different visual 
	/// effect needs to be triggered when a teleport is aborted.
	/// </summary>
	public event Action EnterStateCancelTeleport;

	/// <summary>
	/// This coroutine will be executed when the pre-teleport state is unabled to transition to the teleporting state.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator CancelTeleportStateCoroutine()
	{
		LogState("CancelTeleportState: Start");
		CurrentState = States.CancelTeleport;
		if (EnterStateCancelTeleport != null)
		{
			EnterStateCancelTeleport();
		}
		LogState("CancelTeleportState: End");

		// Teleport was cancelled, notify the teleport destination.
		DeactivateDestination();

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		StartCoroutine(ReadyStateCoroutine());
	}

	/// <summary>
	/// This event will occur when the teleport actually occurs. There should be one Transition Handler
	/// enabled and attached to this event. There may be other handlers attached to this event to trigger
	/// sound effects or gameplay logic to respond to the teleport event.
	/// 
	/// The transition handler is responsible for actually moving the player to the destination, and can achieve
	/// this goal however it wants. Example teleport transition handlers include:
	/// * Instant - Just move the player with no delay or effect.
	/// * Blink - Fade the camera to black, teleport, then fade back up. 
	/// * Warp - Translate the camera over some fixed amount of time to the new destination.
	/// </summary>
	public event Action EnterStateTeleporting;

	/// <summary>
	/// This coroutine will yield until IsTransitioning back to false, which will be immediately unless there
	/// is a transition handler that needs to take some time to move the player to the new location.
	/// This allows transition handlers to take as much (or little) time as they need to complete their task
	/// of moving the player to the teleport destination.
	/// </summary>
	protected IEnumerator TeleportingStateCoroutine()
	{
		LogState("TeleportingState: Start");
		CurrentState = States.Teleporting;
		EnableMotion(false, false); // movement is always disabled during teleport.
		if (EnterStateTeleporting != null)
		{
			EnterStateTeleporting();
		}

		// If a handler sets this, it needs to clear it when the transition completes.
		while (IsTransitioning)
		{
			yield return null;
		}

		LogState("TeleportingState: End");

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		StartCoroutine(PostTeleportStateCoroutine());
	}

	/// <summary>
	/// This event will occur after the teleport has completed.
	/// </summary>
	public event Action EnterStatePostTeleport;

	/// <summary>
	/// The PostTeleport coroutine is typically just a single frame state that deactivates the destination 
	/// indicator and raises the EnterStatePostTeleport event which could be used for any number of gameplay
	/// purposes such as triggering a character animation, sound effect or possibly delaying the exit of the 
	/// PostTeleport state for some gameplay reason such as a cooldown on teleports.
	/// </summary>
	/// <returns></returns>
	protected IEnumerator PostTeleportStateCoroutine()
	{
		LogState("PostTeleportState: Start");
		CurrentState = States.PostTeleport;
		EnableMotion(EnableMovementDuringPostTeleport, EnableRotationDuringPostTeleport);
		if (EnterStatePostTeleport != null)
		{
			EnterStatePostTeleport();
		}

		while (IsPostTeleportRequested)
		{
			yield return null;
		}

		LogState("PostTeleportState: End");

		DeactivateDestination();

		// Wait until the next frame to proceed to the next state's coroutine.
		yield return null;

		StartCoroutine(ReadyStateCoroutine());
	}

	/// <summary>
	/// This event is raised when the character actually teleports, which is typically triggered by the 
	/// transition handler.
	/// 
	/// The first parameter is the character controller's transform.
	/// The second and third are the position and rotation, respectively, which the character controller 
	/// will be assigned immediately after the event is raised.
	/// </summary>
	public event Action<Transform, Vector3, Quaternion> Teleported;

	/// <summary>
	/// Perform the actual teleport.
	/// Note that warp transitions do not call this function and instead moves the game object 
	/// during the transition time time.
	/// </summary>
	public void DoTeleport()
	{
		var character = LocomotionController.CharacterController;
		var characterTransform = character.transform;
		var destTransform = _teleportDestination.OrientationIndicator;

		Vector3 destPosition = destTransform.position;
		destPosition.y += character.height * 0.5f;
		Quaternion destRotation = _teleportDestination.LandingRotation;// destTransform.rotation;
#if false
		Quaternion destRotation = destTransform.rotation;

		//Debug.Log("Rots: " + destRotation + " " + destTransform.rotation * Quaternion.Euler(0, -LocomotionController.CameraRig.trackingSpace.localEulerAngles.y, 0));

		destRotation = destRotation * Quaternion.Euler(0, -LocomotionController.CameraRig.trackingSpace.localEulerAngles.y, 0);
#endif
		if (Teleported != null)
		{
			Teleported(characterTransform, destPosition, destRotation);
		}

		characterTransform.position = destPosition;
		characterTransform.rotation = destRotation;
	}

	/// <summary>
	/// Convenience method for finding the character's position.
	/// </summary>
	/// <returns></returns>
	public Vector3 GetCharacterPosition()
	{
		return LocomotionController.CharacterController.transform.position;
	}

	/// <summary>
	/// Return a quaternion for the Y axis of the HMD's orientation. 
	/// Used by orientation handlers to track the current heading before processing user input to adjust it.
	/// </summary>
	/// <returns></returns>
	public Quaternion GetHeadRotationY()
	{
		Quaternion headRotation = Quaternion.identity;
#if UNITY_2019_1_OR_NEWER
		UnityEngine.XR.InputDevice device = UnityEngine.XR.InputDevices.GetDeviceAtXRNode(UnityEngine.XR.XRNode.Head);
		if (device.isValid)
		{
			device.TryGetFeatureValue(UnityEngine.XR.CommonUsages.deviceRotation, out headRotation);
		}
#elif UNITY_2017_2_OR_NEWER
		List<UnityEngine.XR.XRNodeState> nodeStates = new List<UnityEngine.XR.XRNodeState>();
		UnityEngine.XR.InputTracking.GetNodeStates(nodeStates);
		foreach (UnityEngine.XR.XRNodeState n in nodeStates)
		{
			if (n.nodeType == UnityEngine.XR.XRNode.Head)
			{
				n.TryGetRotation(out headRotation);
				break;
			}
		}
#else
		headRotation = InputTracking.GetLocalRotation(VRNode.Head);
#endif
		Vector3 euler = headRotation.eulerAngles;
		euler.x = 0;
		euler.z = 0;
		headRotation = Quaternion.Euler(euler);
		return headRotation;
	}

	/// <summary>
	/// Warp just the position towards the destination.
	/// </summary>
	/// <param name="startPos"></param>
	/// <param name="positionPercent"></param>
	public void DoWarp(Vector3 startPos, float positionPercent)
	{
		var destTransform = _teleportDestination.OrientationIndicator;
		Vector3 destPosition = destTransform.position;
		destPosition.y += LocomotionController.CharacterController.height/2.0f;

		var character = LocomotionController.CharacterController;
		var characterTransform = character.transform;

		var lerpPosition = Vector3.Lerp(startPos, destPosition, positionPercent);

		characterTransform.position = lerpPosition;

		//LocomotionController.PlayerController.Teleported = true;
	}
}
