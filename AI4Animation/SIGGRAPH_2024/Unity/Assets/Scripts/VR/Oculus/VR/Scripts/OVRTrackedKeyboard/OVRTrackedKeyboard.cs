/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Assertions;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

public class OVRTrackedKeyboard : MonoBehaviour
{
	private static readonly float underlayScaleMultX_ = 1.475f;
	private static readonly float underlayScaleConstY_ = 0.001f;
	private static readonly float underlayScaleMultZ_ = 2.138f;
	private static readonly Vector3 underlayOffset_ = new Vector3 { x = 0.0f, y = 0.0f, z = -0.028f };
	private static readonly float boundingBoxAboveKeyboardY_ = 0.08f;
	private static readonly float initialHorizontalDistanceKeyboard_ = 0.30f; // 20 cm / 8 in
	private static readonly float initialVerticalDistanceKeyboard_ = 0.45f; // 45 cm / 18 in

	/// <summary>
	/// Used by TrackingState property to give the current state of keyboard tracking.
	/// </summary>
	public enum TrackedKeyboardState
	{
		/// <summary>
		/// The OVRTrackedKeyboard component has not yet been initialized.
		/// </summary>
		Uninitialized,
		/// <summary>
		/// Component is initialized but user has not selected a keyboard
		/// to track in the system settings.
		/// </summary>
		NoTrackableKeyboard,
		/// <summary>
		/// Keyboard tracking has been stopped or has not yet started for current keyboard.
		/// </summary>
		Offline,
		/// <summary>
		/// Keyboard tracking has been started but no tracking data is yet available.
		/// This can occur if the keyboard is not visible to the cameras.
		/// </summary>
		StartedNotTracked,
		/// <summary>
		/// Keyboard tracking has been started but no tracking data has been available for a while.
		/// This can occur if the keyboard is no longer visible to the cameras.
		/// </summary>
		Stale,
		/// <summary>
		/// Keyboard is currently being tracked and recent tracking data is available.
		/// </summary>
		Valid,
		/// <summary>
		/// An error occurred while initializing keyboard tracking.
		/// </summary>
		Error,
		/// <summary>
		/// Was unable to retrieve system keyboard info. Can occur if required
		/// keyboard extension is not properly enabled in the application manifest.
		/// </summary>
		ErrorExtensionFailed
	}

	/// <summary>
	/// Determines which visualization is used for the tracked keyboard.
	/// </summary>
	public enum KeyboardPresentation
	{
		/// <summary>
		/// The keyboard is rendered as an opaque model in VR and if the user's hands are
		/// placed over it, they are rendered using passthrough.
		/// </summary>
		PreferOpaque,
		/// <summary>
		/// The keyboard and hands are rendered using a rectangular passthrough window
		/// around the keyboard, and only the key labels are rendered in VR on top of the keyboard.
		/// </summary>
		PreferKeyLabels,
	}

	/// <summary>
	/// Determines amount that keyboard is tilted from its ordinary horizontal position. For internal use.
	/// </summary>
	public float CurrentKeyboardAngleFromUp { get; private set; } = 0f;

	/// <summary>
	/// Current state of keyboard tracking.
	/// </summary>
	public TrackedKeyboardState TrackingState { get; private set; } = TrackedKeyboardState.Uninitialized;
	/// <summary>
	/// Provides information about the keyboard currently being tracked by this
	/// OVRTrackedKeyboard component.
	/// </summary>
	public OVRKeyboard.TrackedKeyboardInfo ActiveKeyboardInfo { get; private set; }
	/// <summary>
	/// Provides information about the keyboard currently selected for tracking in
	/// the system settings. May not yet be tracked by this OVRTrackedKeyboard component.
	/// </summary>
	public OVRKeyboard.TrackedKeyboardInfo SystemKeyboardInfo { get; private set; }

	/// <summary>
	/// Determines which visualization will be used to present the tracked keyboard
	/// to the user.
	/// </summary>
	public KeyboardPresentation Presentation
	{
		get
		{
			return presentation;
		}
		set
		{
			presentation = value;
			UpdatePresentation(GetKeyboardVisibility());
		}
	}

	/// <summary>
	/// Specifies whether or not the OVRTrackedKeyboard component will attempt to search
	/// for and track a keyboard. If true, the component will continually search
	/// for a tracked keyboard. If one is detected it will be shown. If false,
	/// no keyboard is shown and the prefab is inactive. The keyboard can still
	/// be used to enter text into input fields even though it cannot be seen in VR.
	/// </summary>
	public bool TrackingEnabled
	{
		get
		{
			return trackingEnabled;
		}
		set
		{
			trackingEnabled = value;
		}
	}

	/// <summary>
	/// Specifies whether or not the keyboard must be connected via Bluetooth in
	/// order to be tracked. If set to true, the keyboard must be connected to the
	/// headset via Bluetooth in order to be tracked. The keyboard will stop being
	/// tracked if it is powered off or disconnected from the headset. If set to false,
	/// the keyboard will be tracked as long as it is visible to the headset's cameras.
	/// </summary>
	public bool ConnectionRequired
	{
		get
		{
			return connectionRequired;
		}
		set
		{
			connectionRequired = value;
		}
	}

	/// <summary>
	/// If true, will show the keyboard even if it is not currently connected or
	/// visible to the cameras. This is mainly useful for testing the feature when
	/// you don't have access to a physical keyboard. The keyboard that appears will
	/// be based on which keyboard is selected in Settings on the headset. The
	/// keyboard will appear in front of the user at waist level.
	/// </summary>
	public bool ShowUntracked
	{
		get
		{
			return showUntracked;
		}
		set
		{
			showUntracked = value;
		}
	}

	public bool RemoteKeyboard
	{
		get
		{
			if (KeyboardQueryFlags == OVRPlugin.TrackedKeyboardQueryFlags.Local)
			{
				return false;
			}
			else
			{
				return true;
			}
		}
		set
		{
			if(value == true)
			{
				KeyboardQueryFlags = OVRPlugin.TrackedKeyboardQueryFlags.Remote;
			} else
			{
				KeyboardQueryFlags = OVRPlugin.TrackedKeyboardQueryFlags.Local;
			}
		}
	}

	/// <summary>
	/// Specifies whether to search for local keyboards attached to the headset
	/// or for remote keyboards not attached to the headset.
	/// </summary>
	public OVRPlugin.TrackedKeyboardQueryFlags KeyboardQueryFlags
	{
		get
		{
			return keyboardQueryFlags;
		}
		set
		{
			keyboardQueryFlags = value;
		}
	}

#region User settings
	// These properties can be modified by the user of the prefab
	[Header("Settings")]
	[SerializeField]
	[Tooltip("If true, will continually try to track and show keyboard. If false, no keyboard will be shown.")]
	private bool trackingEnabled = true;

	[SerializeField]
	[Tooltip("If true, system keyboard must be paired and connected to track.")]
	private bool connectionRequired = true;

	[SerializeField]
	[Tooltip("If true, keyboard will be displayed even if it is not currently connected or visible.")]
	private bool showUntracked = false;

	[SerializeField]
	[Tooltip("Which type of keyboard you wish to use.")]
	private OVRPlugin.TrackedKeyboardQueryFlags keyboardQueryFlags = OVRPlugin.TrackedKeyboardQueryFlags.Local;

	[SerializeField]
	[Tooltip("Opaque will render a solid model of the keyboard with passthrough hands. " +
	         "Key Labels will render the entire keyboard in passthrough other than the key labels. " +
	         "These are both suggestions and may not always be available.")]
	private KeyboardPresentation presentation = KeyboardPresentation.PreferOpaque;

	[SerializeField]
	[Tooltip("Changes the Texture Quality setting of the currently used texture. Affects visualization quality only " +
			 "A value of -1 means no filtering. Bilinear is 0 (Unity Default) up to Aniso 16x which is 9.")]
	public OVRTextureQualityFiltering textureFiltering = OVRTextureQualityFiltering.Aniso2x;

	[SerializeField]
	[Tooltip("Changes the MipMap Bias of the currently used texture. Affects visualization quality only.")]
	[Range(-1.0f, 1.0f)]
	public float mipmapBias = -0.3f;

	/// <summary>
	/// How large of a passthrough area to show surrounding the keyboard when using Key Label presentation.
	/// </summary>
	[Tooltip("How large of a passthrough area to show surrounding the keyboard when using Key Label presentation")]
	public float PassthroughBorderMultiplier = 0.2f;

	/// <summary>
	/// The shader used for rendering the keyboard model in opaque mode.
	/// </summary>
	[Tooltip("The shader used for rendering the keyboard model")]
	public Shader keyboardModelShader;

	/// <summary>
	/// The shader used for rendering transparent parts of the keyboard model in opaque mode.
	/// </summary>
	[Tooltip("The shader used for rendering transparent parts of the keyboard model")]
	public Shader keyboardModelAlphaBlendShader;
#endregion

	private OVRPlugin.TrackedKeyboardPresentationStyles currentKeyboardPresentationStyles = 0;
	private OVROverlay projectedPassthroughOpaque_;
	private MeshRenderer[] activeKeyboardRenderers_;
	private GameObject activeKeyboardMesh_;
	private GameObject[] keyboardMeshNodes_;
	private MeshRenderer activeKeyboardMeshRenderer_;
	private GameObject passthroughQuad_;
	private Shader opaqueShader_;
	// This is a copy of the texture loaded from the glb. The original texture might be read-only on the GPU (impossible to modify).
	private Texture2D dynamicQualityTexture_;
	private Vector3 untrackedPosition_;

	// These properties generally don't need to be modified by the user of the prefab

	/// <summary>
	/// Internal only. The shader used to render the keyboard in key label mode.
	/// </summary>
	[Header("Internal")]
	public Shader KeyLabelModeShader;
	/// <summary>
	/// Internal only. The shader used to render the passthrough rectangle in opaque mode.
	/// </summary>
	public Shader PassthroughShader;

#region MR Service Setup
	[SerializeField] private Transform projectedPassthroughRoot;
	[SerializeField] private MeshFilter projectedPassthroughMesh;
#endregion

	/// <summary>
	/// Internal only. The passthrough layer used to render the passthrough rectangle in key label mode.
	/// </summary>
	public OVRPassthroughLayer ProjectedPassthroughKeyLabel;
	/// <summary>
	/// Internal only. The passthrough layer used to render the passthrough rectangle in opaque mode.
	/// </summary>
	public OVROverlay PassthroughOverlay
	{
		get { return projectedPassthroughOpaque_; }
		private set {}
	}

	/// <summary>
	/// Event that is dispatched when the component starts or stops actively tracking the keyboard.
	/// </summary>
	public Action<TrackedKeyboardSetActiveEvent> TrackedKeyboardActiveChanged = delegate { };
	/// <summary>
	/// Event that is dispatched when the state of keyboard tracking changes (e.g. tracking
	/// becomes stale or valid as keyboard passes in/out of camera view).
	/// </summary>
	public Action<TrackedKeyboardVisibilityChangedEvent> TrackedKeyboardVisibilityChanged = delegate { };

	/// <summary>
	/// Transform that determines current position and rotation of the keyboard.
	/// </summary>
	public Transform ActiveKeyboardTransform;

	/// <summary>
	/// Internal only. Determines whether the hands are currently positioned over the keyboard.
	/// In opaque presentation mode, passthrough hands are only shown when this is true.
	/// </summary>
	[HideInInspector]
	public bool HandsOverKeyboard = false;

	private OVRCameraRig cameraRig_;

	private Coroutine updateKeyboardRoutine_;

	private BoxCollider keyboardBoundingBox_;

	private float staleTimeoutCounter_ = 0f;
	private const float STALE_TIMEOUT = 10f;
	private float reacquisitionTimer_ = 0f;
	private float sendFilteredPoseEventTimer_ = 0f;
	private int skippedPoseCount_ = 0;
	private const float FILTERED_POSE_TIMEOUT = 15f;

	// Exponentially-weighted average filter (EWA), smooths out changes in keyboard tracking over time
	private Vector3? EWAPosition = null;
	private Quaternion? EWARotation = null;
	private float HAND_HEIGHT_TUNING = 0.0f;

	/// <summary>
	/// Determines whether rolling average filter and keyboard angle filters are applied.
	/// If true, keyboard will be shown in latest tracked position at all times.
	/// </summary>
	[HideInInspector]
	public bool UseHeuristicRollback = false;

	private IEnumerator Start()
	{
		cameraRig_ = FindObjectOfType<OVRCameraRig>();

		SystemKeyboardInfo = new OVRKeyboard.TrackedKeyboardInfo
		{
			Name = "None",
			Dimensions = new Vector3(0f, 0f, 0f),
			Identifier = uint.MaxValue
		};

		yield return InitializeHandPresenceData();

		yield return UpdateTrackingStateCoroutine();
	}

	private IEnumerator InitializeHandPresenceData()
	{
		GameObject ovrCameraRig = GameObject.Find("OVRCameraRig");
		if (ovrCameraRig == null)
		{
			Debug.LogError("Scene does not contain an OVRCameraRig");
			yield break;
		}

		projectedPassthroughOpaque_ = ovrCameraRig.AddComponent<OVROverlay>();

		projectedPassthroughOpaque_.currentOverlayShape = OVROverlay.OverlayShape.KeyboardHandsPassthrough;

		projectedPassthroughOpaque_.hidden = true;
		projectedPassthroughOpaque_.gameObject.SetActive(true);

		ProjectedPassthroughKeyLabel.hidden = true;
		ProjectedPassthroughKeyLabel.gameObject.SetActive(true);
	}

	void RegisterPassthroughMeshToSDK()
	{
		if (ProjectedPassthroughKeyLabel.IsSurfaceGeometry(projectedPassthroughMesh.gameObject))
		{
			ProjectedPassthroughKeyLabel.RemoveSurfaceGeometry(projectedPassthroughMesh.gameObject);
		}

		ProjectedPassthroughKeyLabel.AddSurfaceGeometry(projectedPassthroughMesh.gameObject, true);
	}

#region Public API

	/// <summary>
	/// Returns the distance from the given point to the keyboard
	/// </summary>
	/// <param name="point">A 3D vector coordinate to use as the reference point</param>
	/// <returns>A floating point value that is the distance to intersect within the keyboard bounds</returns>
	public float GetDistanceToKeyboard(Vector3 point)
	{
		if (keyboardBoundingBox_ == null)
		{
			return Mathf.Infinity;
		}
		if (keyboardBoundingBox_.bounds.Contains(point))
		{
			return 0.0f;
		}

		var closestPointToKb = keyboardBoundingBox_.ClosestPointOnBounds(point);
		var pointToKeyboard = closestPointToKb - point;
		RaycastHit hitInfo;
		bool didHit = keyboardBoundingBox_.Raycast(
			new Ray(point, pointToKeyboard),
			out hitInfo,
			Mathf.Infinity);
		return didHit ? hitInfo.distance : Mathf.Infinity;
	}

	/// <summary>
	/// Invokes an Android broadcast to launch a keyboard selection dialog for local keyboard type.
	/// </summary>
	public void LaunchLocalKeyboardSelectionDialog()
	{
		LaunchOverlayIntent("systemux://dialog/set-local-physical-tracked-keyboard");
	}

	/// <summary>
	/// Invokes an Android broadcast to launch a keyboard selection dialog for remote keyboard type.
	/// </summary>
	public void LaunchRemoteKeyboardSelectionDialog()
	{
		LaunchOverlayIntent("systemux://dialog/set-remote-physical-tracked-keyboard");
	}

#endregion

#region Private Helpers
	private bool KeyboardTrackerIsRunning()
	{
		return (TrackingState != TrackedKeyboardState.NoTrackableKeyboard
				&& TrackingState != TrackedKeyboardState.Offline);
	}

	private IEnumerator UpdateTrackingStateCoroutine()
	{
		for (;;)
		{
			// On Link this is called before initialization.
			//We don't want this on our normal flow because it breaks our tests.
#if !UNITY_ANDROID && !UNITY_EDITOR
			if(OVRPlugin.initialized) {
#endif
			OVRKeyboard.TrackedKeyboardInfo keyboardInfo;
			if (OVRKeyboard.GetSystemKeyboardInfo(KeyboardQueryFlags, out keyboardInfo))
			{
				bool systemKeyboardSwitched = false;
				if (SystemKeyboardInfo.Identifier != keyboardInfo.Identifier || SystemKeyboardInfo.KeyboardFlags != keyboardInfo.KeyboardFlags)
				{
					Debug.Log(String.Format("New System keyboard info: [{0}] {1} (Flags {2}) ({3} {4})",
						keyboardInfo.Identifier, keyboardInfo.Name,
						keyboardInfo.KeyboardFlags,
						(keyboardInfo.SupportedPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.Opaque) != 0 ? "Supports Opaque" : "",
						(keyboardInfo.SupportedPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.KeyLabel) != 0 ? "Supports Key Label" : ""));
					if (TrackingState == TrackedKeyboardState.NoTrackableKeyboard){
						SetKeyboardState(TrackedKeyboardState.Offline);
					}
					SystemKeyboardInfo = keyboardInfo;
					systemKeyboardSwitched = true;
				}

				bool keyboardExists = (keyboardInfo.KeyboardFlags & OVRPlugin.TrackedKeyboardFlags.Exists) != 0;
				if ((keyboardExists && trackingEnabled) || showUntracked)
				{
					bool localKeyboard = (keyboardInfo.KeyboardFlags & OVRPlugin.TrackedKeyboardFlags.Local) != 0;
					bool remoteKeyboard = (keyboardInfo.KeyboardFlags & OVRPlugin.TrackedKeyboardFlags.Remote) != 0;
					bool connectedKeyboard = (keyboardInfo.KeyboardFlags & OVRPlugin.TrackedKeyboardFlags.Connected) != 0;
					bool shouldBeRunning = remoteKeyboard || (localKeyboard && (!connectionRequired || connectedKeyboard)) || showUntracked;

					if(KeyboardTrackerIsRunning() && (systemKeyboardSwitched || !shouldBeRunning))
					{
						StopKeyboardTrackingInternal();
					}

					if(!KeyboardTrackerIsRunning() && shouldBeRunning)
					{
						yield return StartKeyboardTrackingCoroutine();
					}
				}
				else
				{
					if (KeyboardTrackerIsRunning()){
						StopKeyboardTrackingInternal();
					}

					if (!keyboardExists)
					{
						SetKeyboardState(TrackedKeyboardState.NoTrackableKeyboard);
					}
				}
			}
			else
			{
				if (KeyboardTrackerIsRunning()){
					StopKeyboardTrackingInternal();
				}
				SetKeyboardState(TrackedKeyboardState.ErrorExtensionFailed);
			}
			SystemKeyboardInfo = keyboardInfo;
#if !UNITY_ANDROID && !UNITY_EDITOR
			}
#endif
			yield return new WaitForSeconds(.1f);
		}
	}

	private IEnumerator StartKeyboardTrackingCoroutine()
	{
		if (KeyboardTrackerIsRunning())
		{
			Debug.Log("StartKeyboardTracking(): Keyboard already being tracked");
			yield break;
		}

		Assert.IsTrue(
			!KeyboardTrackerIsRunning()
			&& activeKeyboardMesh_ == null
			&& activeKeyboardRenderers_ == null
			&& updateKeyboardRoutine_ == null,
			$"State: {TrackingState}, Mesh: {activeKeyboardMesh_}, Coroutine: {updateKeyboardRoutine_}");

		InitializeKeyboardInfo();
		RegisterPassthroughMeshToSDK();

		Debug.Log("Calling StartKeyboardTracking with id " + SystemKeyboardInfo.Identifier);

		if (!OVRPlugin.StartKeyboardTracking(SystemKeyboardInfo.Identifier))
		{
			if (!showUntracked)
			{
				Debug.LogWarning("OVRKeyboard.StartKeyboardTracking Failed");
				SetKeyboardState(TrackedKeyboardState.Error);
				yield break;
			}
		}

		projectedPassthroughRoot.localScale = new Vector3 { x = SystemKeyboardInfo.Dimensions.x * underlayScaleMultX_, y = underlayScaleConstY_, z = SystemKeyboardInfo.Dimensions.z * underlayScaleMultZ_ };

		currentKeyboardPresentationStyles = SystemKeyboardInfo.SupportedPresentationStyles;
		ActiveKeyboardInfo = SystemKeyboardInfo;
		LoadKeyboardMesh();
		updateKeyboardRoutine_ = StartCoroutine(UpdateKeyboardPose());
		EWAPosition = null;
		EWARotation = null;

		TrackedKeyboardActiveChanged?.Invoke(new TrackedKeyboardSetActiveEvent(isEnabled: true));
		SetKeyboardState(TrackedKeyboardState.StartedNotTracked);
	}

	private void StopKeyboardTrackingInternal()
	{
		if (!KeyboardTrackerIsRunning() || updateKeyboardRoutine_ == null)
		{
			SetKeyboardState(TrackedKeyboardState.Offline);
			return;
		}

		projectedPassthroughOpaque_.hidden = true;
		ProjectedPassthroughKeyLabel.hidden = true;

		TrackedKeyboardActiveChanged?.Invoke(new TrackedKeyboardSetActiveEvent(isEnabled: false));

		Debug.Log($"StopKeyboardTracking {ActiveKeyboardInfo.Name}");

		StopCoroutine(updateKeyboardRoutine_);
		updateKeyboardRoutine_ = null;

		OVRKeyboard.StopKeyboardTracking(ActiveKeyboardInfo);
		InitializeKeyboardInfo();

		if (activeKeyboardMesh_ != null)
		{
			Destroy(activeKeyboardMesh_.gameObject);
			activeKeyboardMesh_ = null;
			activeKeyboardRenderers_ = null;
			keyboardBoundingBox_ = null;
		}

		untrackedPosition_ = Vector3.zero;

		SetKeyboardState(TrackedKeyboardState.Offline);
	}

	private IEnumerator UpdateKeyboardPose()
	{
		while (true)
		{
			transform.position = cameraRig_.trackingSpace.transform.position;
			transform.rotation = cameraRig_.trackingSpace.transform.rotation;

			var poseState = OVRKeyboard.GetKeyboardState();

			// Emulate tracking when showUntracked is set
			if ((!poseState.isPositionValid || !poseState.isPositionTracked) && showUntracked)
			{
				poseState.isPositionValid = true;
				poseState.isPositionTracked = true;

				if (untrackedPosition_ == Vector3.zero && Camera.main != null)
				{
					// Start keyboard in a nice position at waist level in front
					Transform cameraTransform = Camera.main.transform;
					Vector3 cameraDirectionHorizontal =
						Vector3.ProjectOnPlane(cameraTransform.forward, Vector3.up).normalized;
					untrackedPosition_ = cameraTransform.position +
					                     cameraDirectionHorizontal * initialHorizontalDistanceKeyboard_ +
					                     new Vector3(0.0f, -initialVerticalDistanceKeyboard_, 0.0f);
				}
				poseState.position = untrackedPosition_;
			}

			TrackedKeyboardState keyboardState = TrackedKeyboardState.StartedNotTracked;

			if (poseState.isPositionValid)
			{
				if (poseState.isPositionTracked && activeKeyboardMesh_ != null)
				{
					float keyboardAngleFilter = UseHeuristicRollback ? 360f : 20f;
					float ewaAlpha = UseHeuristicRollback ? 0f : 0.65f;

					var worldRotation = transform.rotation * poseState.rotation;
					var upRotated = worldRotation * Vector3.up;
					CurrentKeyboardAngleFromUp = Vector3.Angle(upRotated, Vector3.up);

					if (CurrentKeyboardAngleFromUp < keyboardAngleFilter)
					{
						if (!EWAPosition.HasValue)
						{
							EWAPosition = poseState.position;
						}
						else
						{
							EWAPosition = ewaAlpha * EWAPosition + (1f - ewaAlpha) * poseState.position;
						}

						if (!EWARotation.HasValue)
						{
							EWARotation = poseState.rotation;
						}
						else
						{
							EWARotation = Quaternion.Slerp(EWARotation.Value, poseState.rotation, 1f - ewaAlpha);
						}

						ActiveKeyboardTransform.localPosition = EWAPosition.Value;
						ActiveKeyboardTransform.localRotation = EWARotation.Value;

						projectedPassthroughRoot.localPosition = EWAPosition.Value + underlayOffset_ + new Vector3(0f, HAND_HEIGHT_TUNING, 0f);
						projectedPassthroughRoot.localRotation = EWARotation.Value;
					}
					else
					{
						skippedPoseCount_++;
					}

				}

				keyboardState = poseState.isPositionTracked
					? TrackedKeyboardState.Valid
					: TrackedKeyboardState.Stale;
			}

			SetKeyboardState(keyboardState);
			UpdateSkippedPoseTimer();
			yield return null;
		}
	}

	private void UpdateSkippedPoseTimer()
	{
		sendFilteredPoseEventTimer_ += Time.deltaTime;
		if (sendFilteredPoseEventTimer_ > FILTERED_POSE_TIMEOUT
			&& skippedPoseCount_ > 0)
		{
			// dispatcher_.Dispatch(new TrackedKeyboardSkippedPoseEvent(skippedPoseCount_));
			skippedPoseCount_ = 0;
			sendFilteredPoseEventTimer_ = 0f;
		}
	}

	private void LoadKeyboardMesh()
	{
		Debug.Log("LoadKeyboardMesh");
		activeKeyboardMesh_ = LoadRuntimeKeyboardMesh();
		if(activeKeyboardMesh_ == null) {
			Debug.LogError("Failed to load keyboard mesh.");
			SetKeyboardState(TrackedKeyboardState.Error);
			return;
		}

		keyboardMeshNodes_ = new GameObject[activeKeyboardMesh_.transform.childCount];
		for (int i = 0; i < activeKeyboardMesh_.transform.childCount; i++)
		{
			keyboardMeshNodes_[i] = activeKeyboardMesh_.transform.GetChild(i).gameObject;
		}

		keyboardBoundingBox_ = activeKeyboardMesh_.AddComponent<BoxCollider>();

		keyboardBoundingBox_.center =
			new Vector3(0.0f, ActiveKeyboardInfo.Dimensions.y / 2.0f, 0.0f);
		keyboardBoundingBox_.size =
			new Vector3(ActiveKeyboardInfo.Dimensions.x,
				ActiveKeyboardInfo.Dimensions.y + boundingBoxAboveKeyboardY_,
				ActiveKeyboardInfo.Dimensions.z);

		activeKeyboardMeshRenderer_ = keyboardMeshNodes_[0].GetComponentInChildren<MeshRenderer>();
		if (activeKeyboardMeshRenderer_ == null)
		{
			Debug.LogError("Failed to load activeKeyboardMeshRenderer_.");
			SetKeyboardState(TrackedKeyboardState.Error);
			return;
		}

		opaqueShader_ = activeKeyboardMeshRenderer_.material.shader;

		passthroughQuad_ = GameObject.CreatePrimitive(PrimitiveType.Quad);
		passthroughQuad_.transform.localPosition = new Vector3(0.0f, -0.01f, 0.0f);
		passthroughQuad_.transform.parent = activeKeyboardMesh_.transform;
		passthroughQuad_.transform.localRotation = Quaternion.Euler(90.0f, 0.0f, 0.0f);
		float borderSize = ActiveKeyboardInfo.Dimensions.x * PassthroughBorderMultiplier;
		passthroughQuad_.transform.localScale = new Vector3(
			ActiveKeyboardInfo.Dimensions.x + borderSize,
			ActiveKeyboardInfo.Dimensions.z + borderSize,
			ActiveKeyboardInfo.Dimensions.y);

		MeshRenderer meshRenderer = passthroughQuad_.GetComponent<MeshRenderer>();
		meshRenderer.material.shader = PassthroughShader;

		GameObject parent = new GameObject();
		activeKeyboardMesh_.transform.parent = parent.transform;
		activeKeyboardMesh_ = parent;

		activeKeyboardRenderers_ = activeKeyboardMesh_.GetComponentsInChildren<MeshRenderer>();
		activeKeyboardMesh_.transform.SetParent(ActiveKeyboardTransform, worldPositionStays: false);

		ActiveKeyboardTransform.localRotation = Quaternion.identity;

		// Make a copy of the current main texture (created by the OVRGLTFLoader) to apply our quality setting more freely
		Texture readonlyTexture = activeKeyboardMeshRenderer_.material.mainTexture;
		if (readonlyTexture != null)
		{
			dynamicQualityTexture_ = Texture2D.CreateExternalTexture(
				readonlyTexture.width,
				readonlyTexture.height,
				TextureFormat.BC7,
				mipChain: true, linear: true,
				readonlyTexture.GetNativeTexturePtr());
		}
		UpdateTextureQuality();
		UpdateKeyboardVisibility();
	}

	/// <summary>
	/// Apply the current texture quality settings and reapplies texture to material
	/// </summary>
	void UpdateTextureQuality()
	{
		if (dynamicQualityTexture_ == null)
			return;

		OVRGLTFLoader.ApplyTextureQuality(textureFiltering, ref dynamicQualityTexture_);
		Material currentMat = activeKeyboardMeshRenderer_.material;
		currentMat.mainTexture = dynamicQualityTexture_;
		if (currentMat.HasProperty("_MainTexMMBias"))
			currentMat.SetFloat("_MainTexMMBias", mipmapBias);
		activeKeyboardMeshRenderer_.material = currentMat;
	}

	void UpdatePresentation(bool isVisible)
	{
		KeyboardPresentation presentationToUse = Presentation;
		if(currentKeyboardPresentationStyles != 0) {
			if (Presentation == KeyboardPresentation.PreferOpaque && (currentKeyboardPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.Opaque) == 0) {
				if((currentKeyboardPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.KeyLabel) != 0) {
					presentationToUse = KeyboardPresentation.PreferKeyLabels;
				}
			}
			else if (Presentation == KeyboardPresentation.PreferKeyLabels && (currentKeyboardPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.KeyLabel) == 0) {
				if((currentKeyboardPresentationStyles & OVRPlugin.TrackedKeyboardPresentationStyles.Opaque) != 0) {
					presentationToUse = KeyboardPresentation.PreferOpaque;
				}
			}
		}

		if (!isVisible) {
			projectedPassthroughOpaque_.hidden = true;
			ProjectedPassthroughKeyLabel.hidden = true;
		} else if (presentationToUse == KeyboardPresentation.PreferOpaque) {
			activeKeyboardMeshRenderer_.material.shader = opaqueShader_;
			passthroughQuad_.SetActive(false);
			projectedPassthroughOpaque_.hidden = !GetKeyboardVisibility() || !HandsOverKeyboard;
			ProjectedPassthroughKeyLabel.hidden = true;
			for (int i=1; i < keyboardMeshNodes_.Length; i++)
			{
				keyboardMeshNodes_[i].SetActive(true);
			}
		} else {
			activeKeyboardMeshRenderer_.material.shader = KeyLabelModeShader;
			passthroughQuad_.SetActive(true);
			projectedPassthroughOpaque_.hidden = true;
			ProjectedPassthroughKeyLabel.hidden = false; // Always shown
			for (int i=1; i < keyboardMeshNodes_.Length; i++)
			{
				keyboardMeshNodes_[i].SetActive(false);
			}
		}
	}

	private GameObject LoadRuntimeKeyboardMesh()
	{
		Debug.Log("LoadRuntimekeyboardMesh");
		string[] modelPaths = OVRPlugin.GetRenderModelPaths();
		if (modelPaths != null)
		{
			for (int i = 0; i < modelPaths.Length; i++)
			{
				if ((RemoteKeyboard && modelPaths[i].Equals("/model_fb/keyboard/remote")) ||
					(!RemoteKeyboard && modelPaths[i].Equals("/model_fb/keyboard/local")))
				{
					OVRPlugin.RenderModelProperties modelProps = new OVRPlugin.RenderModelProperties();
					if (OVRPlugin.GetRenderModelProperties(modelPaths[i], ref modelProps))
					{
						if (modelProps.ModelKey != OVRPlugin.RENDER_MODEL_NULL_KEY)
						{
							byte[] data = OVRPlugin.LoadRenderModel(modelProps.ModelKey);
							if (data != null)
							{
								OVRGLTFLoader gltfLoader = new OVRGLTFLoader(data);
								gltfLoader.SetModelShader(keyboardModelShader);
								gltfLoader.SetModelAlphaBlendShader(keyboardModelAlphaBlendShader);
								OVRGLTFScene scene = gltfLoader.LoadGLB(supportAnimation: false, loadMips: true);
								return scene.root;
							}
						}
					}
					Debug.LogError("Failed to load model. Ensure that the correct keyboard is connected.");
					break;
				}
			}
		}
		Debug.LogError("Failed to find keyboard model.");
		return null;
	}

	/// <summary>
	/// Internal only. Updates rendering of keyboard based on its current visibility.
	/// </summary>
	public void UpdateKeyboardVisibility()
	{
		if (activeKeyboardMesh_ == null)
			return;

		var isVisible = GetKeyboardVisibility();
		UpdatePresentation(isVisible);

		if (activeKeyboardRenderers_ == null)
		{
			return;
		}

		foreach (var renderer in activeKeyboardRenderers_)
		{
			renderer.enabled = isVisible;
		}
	}

	private void SetKeyboardState(TrackedKeyboardState state)
	{
		var oldState = TrackingState;
		TrackingState = state;

		bool timedOut = false;

		switch (state)
		{
			case TrackedKeyboardState.Stale:
				if (!HandsOverKeyboard)
				{
					staleTimeoutCounter_ += Time.deltaTime;
					timedOut = staleTimeoutCounter_ - STALE_TIMEOUT > 0f;

					if (timedOut) {
						reacquisitionTimer_ += Time.deltaTime;
						EWAPosition = null;
						EWARotation = null;
					}
				}
				else
				{
					reacquisitionTimer_ = 0f;
					staleTimeoutCounter_ = 0f;
				}
				break;
			case TrackedKeyboardState.Valid:
				staleTimeoutCounter_ = 0f;

				if (oldState == TrackedKeyboardState.Stale
					&& reacquisitionTimer_ > 0f)
				{
					// dispatcher_.Dispatch(new TrackedKeyboardReacquiredEvent(reacquisitionTimer_));
				}
				break;
			case TrackedKeyboardState.StartedNotTracked:
			case TrackedKeyboardState.NoTrackableKeyboard:
			case TrackedKeyboardState.Offline:
				reacquisitionTimer_ = 0f;
				staleTimeoutCounter_ = 0f;
				break;
			default:
				break;
		}

		if (oldState != state || timedOut)
		{
			DispatchVisibilityEvent(timedOut);
		}

		UpdateKeyboardVisibility();
	}

	private bool GetKeyboardVisibility()
	{
		switch (TrackingState)
		{
			case TrackedKeyboardState.Stale:
				if (!HandsOverKeyboard)
				{
					return !(staleTimeoutCounter_ - STALE_TIMEOUT > 0f);
				}
				else
				{
					return true;
				}
			case TrackedKeyboardState.Valid:
				return true;
			default:
				if (showUntracked)
					return true;
				break;
		}

		return false;
	}

	private void InitializeKeyboardInfo()
	{
		ActiveKeyboardInfo = new OVRKeyboard.TrackedKeyboardInfo
		{
			Name = "None",
			Dimensions = new Vector3(0f, 0f, 0f),
			Identifier = uint.MaxValue
		};
	}

	private void LaunchOverlayIntent(String dataUri)
	{
		AndroidJavaObject activityClass = new AndroidJavaClass("com.unity3d.player.UnityPlayer");
		AndroidJavaObject currentActivity = activityClass.GetStatic<AndroidJavaObject>("currentActivity");
		var intent = new AndroidJavaObject("android.content.Intent");

		intent.Call<AndroidJavaObject>("setPackage", "com.oculus.vrshell");
		intent.Call<AndroidJavaObject>("setAction", "com.oculus.vrshell.intent.action.LAUNCH");
		intent.Call<AndroidJavaObject>("putExtra", "intent_data", dataUri);

		// Broadcast instead of starting activity, so that it goes to overlay
		currentActivity.Call("sendBroadcast", intent);
	}
#endregion

	/// <summary>
	/// Stops keyboard tracking and cleans up associated resources.
	/// </summary>
	public void Dispose()
	{
		if (KeyboardTrackerIsRunning())
		{
			StopKeyboardTrackingInternal();
		}

		if (ProjectedPassthroughKeyLabel.IsSurfaceGeometry(projectedPassthroughMesh.gameObject))
		{
			ProjectedPassthroughKeyLabel.RemoveSurfaceGeometry(projectedPassthroughMesh.gameObject);
		}

		if (activeKeyboardMesh_ != null)
		{
			Destroy(activeKeyboardMesh_.gameObject);
		}
	}

	private void DispatchVisibilityEvent(bool timeOut)
	{
		TrackedKeyboardVisibilityChanged?.Invoke(
			new TrackedKeyboardVisibilityChangedEvent(ActiveKeyboardInfo.Name, TrackingState, timeOut));
	}

	/// <summary>
	/// Event sent when tracked keyboard changes visibility (passes in or out of camera view).
	/// </summary>
	public struct TrackedKeyboardVisibilityChangedEvent
	{
		public readonly string ActiveKeyboardName;
		public readonly TrackedKeyboardState State;
		public readonly bool TrackingTimeout;

		public TrackedKeyboardVisibilityChangedEvent(string keyboardModel, TrackedKeyboardState state, bool timeout)
		{
			ActiveKeyboardName = keyboardModel;
			State = state;
			TrackingTimeout = timeout;
		}
	}

	/// <summary>
	/// Event sent when tracked keyboard starts or stops actively tracking.
	/// </summary>
	public struct TrackedKeyboardSetActiveEvent
	{
		public readonly bool IsEnabled;

		public TrackedKeyboardSetActiveEvent(bool isEnabled)
		{
			IsEnabled = isEnabled;
		}
	}
}
