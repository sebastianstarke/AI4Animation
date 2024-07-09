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

using UnityEngine;
using UnityEngine.Assertions;

public class OVRTrackedKeyboardHands : MonoBehaviour
{
	public GameObject LeftHandPresence;
	public GameObject RightHandPresence;
	private bool handPresenceInitialized_ = false;

	private Transform leftHandRoot_;
	private Transform rightHandRoot_;

	public OVRTrackedKeyboard KeyboardTracker;

	private OVRCameraRig cameraRig_;
	private OVRHand leftHand_;
	private OVRSkeleton leftHandSkeleton_;
	private OVRSkeletonRenderer leftHandSkeletonRenderer_;
	private GameObject leftHandSkeletonRendererGO_;
	private SkinnedMeshRenderer leftHandSkinnedMeshRenderer_;
	private OVRMeshRenderer leftHandMeshRenderer_;
	private OVRHand rightHand_;
	private OVRSkeleton rightHandSkeleton_;
	private OVRSkeletonRenderer rightHandSkeletonRenderer_;
	private GameObject rightHandSkeletonRendererGO_;
	private OVRMeshRenderer rightHandMeshRenderer_;
	private SkinnedMeshRenderer rightHandSkinnedMeshRenderer_;

	public bool RightHandOverKeyboard { get; private set; } = false;
	public bool LeftHandOverKeyboard { get; private set; } = false;

	private static readonly float handInnerAlphaThreshold_ = 0.08f;
	private static readonly float handOuterAlphaThreshold_ = 0.20f;
	private static readonly float maximumPassthroughHandsDistance_ = 0.18f;
	private static readonly float minimumModelHandsDistance_ = 0.11f;

	private TrackedKeyboardHandsVisibilityChangedEvent? lastVisibilityEvent_ = null;

	private struct HandBoneMapping
	{
		public Transform LeftHandTransform;
		public Transform LeftPresenceTransform;
		public Transform RightHandTransform;
		public Transform RightPresenceTransform;

		public OVRSkeleton.BoneId BoneName;
		public string HandPresenceLeftBoneName;
		public string HandPresenceRightBoneName;
	};

	private readonly HandBoneMapping[] boneMappings_ = new HandBoneMapping[]
	{
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_WristRoot,
			HandPresenceLeftBoneName = "b_l_wrist",
			HandPresenceRightBoneName = "b_r_wrist"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Thumb0,
			HandPresenceLeftBoneName = "b_l_thumb0",
			HandPresenceRightBoneName = "b_r_thumb0"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Thumb1,
			HandPresenceLeftBoneName = "b_l_thumb1",
			HandPresenceRightBoneName = "b_r_thumb1"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Thumb2,
			HandPresenceLeftBoneName = "b_l_thumb2",
			HandPresenceRightBoneName = "b_r_thumb2"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Thumb3,
			HandPresenceLeftBoneName = "b_l_thumb3",
			HandPresenceRightBoneName = "b_r_thumb3"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Index1,
			HandPresenceLeftBoneName = "b_l_index1",
			HandPresenceRightBoneName = "b_r_index1"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Index2,
			HandPresenceLeftBoneName = "b_l_index2",
			HandPresenceRightBoneName = "b_r_index2"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Index3,
			HandPresenceLeftBoneName = "b_l_index3",
			HandPresenceRightBoneName = "b_r_index3"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Middle1,
			HandPresenceLeftBoneName = "b_l_middle1",
			HandPresenceRightBoneName = "b_r_middle1"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Middle2,
			HandPresenceLeftBoneName = "b_l_middle2",
			HandPresenceRightBoneName = "b_r_middle2"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Middle3,
			HandPresenceLeftBoneName = "b_l_middle3",
			HandPresenceRightBoneName = "b_r_middle3"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Ring1,
			HandPresenceLeftBoneName = "b_l_ring1",
			HandPresenceRightBoneName = "b_r_ring1"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Ring2,
			HandPresenceLeftBoneName = "b_l_ring2",
			HandPresenceRightBoneName = "b_r_ring2"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Ring3,
			HandPresenceLeftBoneName = "b_l_ring3",
			HandPresenceRightBoneName = "b_r_ring3"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Pinky0,
			HandPresenceLeftBoneName = "b_l_pinky0",
			HandPresenceRightBoneName = "b_r_pinky0"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Pinky1,
			HandPresenceLeftBoneName = "b_l_pinky1",
			HandPresenceRightBoneName = "b_r_pinky1"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Pinky2,
			HandPresenceLeftBoneName = "b_l_pinky2",
			HandPresenceRightBoneName = "b_r_pinky2"
		},
		new HandBoneMapping
		{
			BoneName = OVRSkeleton.BoneId.Hand_Pinky3,
			HandPresenceLeftBoneName = "b_l_pinky3",
			HandPresenceRightBoneName = "b_r_pinky3"
		}
	};

	public Material HandsMaterial;

#region MATERIAL PROPERTIES

	private const float XSCALE = 0.73f;
	private const float YSCALE = 0.8f;
	private const float FORWARD_OFFSET = -0.02f;

	private int keyboardPositionID_;
	private int keyboardRotationID_;
	private int keyboardScaleID_;

#endregion

	private void Awake() {
		KeyboardTracker.TrackedKeyboardActiveChanged += TrackedKeyboardActiveUpdated;
		KeyboardTracker.TrackedKeyboardVisibilityChanged += TrackedKeyboardVisibilityChanged;

		keyboardPositionID_ = Shader.PropertyToID("_KeyboardPosition");
		keyboardRotationID_ = Shader.PropertyToID("_KeyboardRotation");
		keyboardScaleID_    = Shader.PropertyToID("_KeyboardScale");
	}

	private void Start()
	{
		cameraRig_ = FindObjectOfType<OVRCameraRig>();
		leftHand_ = cameraRig_.leftHandAnchor.GetComponentInChildren<OVRHand>();
		rightHand_ = cameraRig_.rightHandAnchor.GetComponentInChildren<OVRHand>();
		leftHandSkeleton_ = leftHand_.GetComponent<OVRSkeleton>();
		rightHandSkeleton_ = rightHand_.GetComponent<OVRSkeleton>();

		leftHandMeshRenderer_ = leftHand_.GetComponent<OVRMeshRenderer>();
		rightHandMeshRenderer_ = rightHand_.GetComponent<OVRMeshRenderer>();

		leftHandSkeletonRenderer_ = leftHand_.GetComponent<OVRSkeletonRenderer>();
		rightHandSkeletonRenderer_ = rightHand_.GetComponent<OVRSkeletonRenderer>();
		if (!leftHandSkeletonRenderer_.enabled)
		{
			// App is not using skeleton renderer
			leftHandSkeletonRenderer_ = null;
			rightHandSkeletonRenderer_ = null;
		}

		leftHandSkinnedMeshRenderer_ = leftHand_.GetComponent<SkinnedMeshRenderer>();
		rightHandSkinnedMeshRenderer_ = rightHand_.GetComponent<SkinnedMeshRenderer>();

		var leftHand = GameObject.Instantiate(LeftHandPresence);
		var rightHand = GameObject.Instantiate(RightHandPresence);
		leftHandRoot_ = leftHand.transform;
		rightHandRoot_ = rightHand.transform;

		leftHand.SetActive(false);
		rightHand.SetActive(false);

#if !UNITY_EDITOR  // Initialized in LateUpdate() in editor
		RetargetHandTrackingToHandPresence();
		enabled = false;
#endif
	}

	private bool AreControllersActive =>
		!(leftHand_.IsTracked || rightHand_.IsTracked);

	private void LateUpdate()
	{
#if UNITY_EDITOR
		if (!handPresenceInitialized_)
		{
			if (leftHandSkeleton_.IsInitialized && rightHandSkeleton_.IsInitialized)
			{
				RetargetHandTrackingToHandPresence();
			}
			else
			{
				return;
			}
		}
#endif

		if (AreControllersActive)
		{
			DisableHandObjects();
			return;
		}

		foreach (var boneEntry in boneMappings_)
		{
			boneEntry.LeftPresenceTransform.localRotation = boneEntry.LeftHandTransform.localRotation;

			boneEntry.RightPresenceTransform.localRotation = boneEntry.RightHandTransform.localRotation;

			if (boneEntry.BoneName == OVRSkeleton.BoneId.Hand_WristRoot)
			{
				boneEntry.LeftPresenceTransform.rotation = boneEntry.LeftHandTransform.rotation;

				boneEntry.RightPresenceTransform.rotation = boneEntry.RightHandTransform.rotation;

				var leftScale = leftHand_.HandScale;
				var rightScale = rightHand_.HandScale;

				boneEntry.RightPresenceTransform.localScale = new Vector3(rightScale, rightScale, rightScale);
				boneEntry.LeftPresenceTransform.localScale = new Vector3(leftScale, leftScale, leftScale);
			}
		}
		rightHandRoot_.position = rightHand_.transform.position;
		rightHandRoot_.rotation = rightHand_.transform.rotation;

		leftHandRoot_.position = leftHand_.transform.position;
		leftHandRoot_.rotation = leftHand_.transform.rotation;

		var leftHandDistance = GetHandDistanceToKeyboard(leftHandSkeleton_);
		var rightHandDistance = GetHandDistanceToKeyboard(rightHandSkeleton_);

		LeftHandOverKeyboard = ShouldEnablePassthrough(leftHandDistance);
		RightHandOverKeyboard = ShouldEnablePassthrough(rightHandDistance);

		KeyboardTracker.HandsOverKeyboard = RightHandOverKeyboard || LeftHandOverKeyboard;

		var enableLeftModel = ShouldEnableModel(leftHandDistance);
		var enableRightModel = ShouldEnableModel(rightHandDistance);
		SetHandModelsEnabled(enableLeftModel, enableRightModel);

		if (KeyboardTracker.Presentation == OVRTrackedKeyboard.KeyboardPresentation.PreferOpaque)
		{
			// Used mixed reality service hands
			leftHandRoot_.gameObject.SetActive(false);
			rightHandRoot_.gameObject.SetActive(false);
		}
		else
		{
			leftHandRoot_.gameObject.SetActive(LeftHandOverKeyboard);
			rightHandRoot_.gameObject.SetActive(RightHandOverKeyboard);
		}

		var position = KeyboardTracker.ActiveKeyboardTransform?.position;
		var rotation = KeyboardTracker.ActiveKeyboardTransform?.rotation;
		var offset = KeyboardTracker.ActiveKeyboardTransform == null
			? Vector3.zero
			: KeyboardTracker.ActiveKeyboardTransform.forward * FORWARD_OFFSET;

		HandsMaterial.SetVector(keyboardPositionID_, position.HasValue ? position.Value + offset : Vector3.zero);
		HandsMaterial.SetVector(keyboardRotationID_, rotation.HasValue ? rotation.Value.eulerAngles : Vector3.zero);
		HandsMaterial.SetVector(
			keyboardScaleID_,
			new Vector4(
				KeyboardTracker.ActiveKeyboardInfo.Dimensions.x * XSCALE,
				0.1f,
				KeyboardTracker.ActiveKeyboardInfo.Dimensions.z * YSCALE,
				1f
			)
		);

		if (lastVisibilityEvent_ == null
				|| LeftHandOverKeyboard != lastVisibilityEvent_.Value.leftVisible
				|| RightHandOverKeyboard != lastVisibilityEvent_.Value.rightVisible)
		{
			lastVisibilityEvent_ = new TrackedKeyboardHandsVisibilityChangedEvent
			{
				leftVisible = LeftHandOverKeyboard,
				rightVisible = RightHandOverKeyboard
			};
			KeyboardTracker.UpdateKeyboardVisibility();
		}

		if (LeftHandOverKeyboard || RightHandOverKeyboard)
		{
			var handsIntensity = new OVRPlugin.InsightPassthroughKeyboardHandsIntensity
			{
				LeftHandIntensity =
					ComputeOpacity(leftHandDistance, handInnerAlphaThreshold_, handOuterAlphaThreshold_),
				RightHandIntensity =
					ComputeOpacity(rightHandDistance, handInnerAlphaThreshold_, handOuterAlphaThreshold_)
			};
			OVRPlugin.SetInsightPassthroughKeyboardHandsIntensity(KeyboardTracker.PassthroughOverlay.layerId, handsIntensity);
		}
	}

	private bool ShouldEnablePassthrough(float distance)
	{
		return distance <= maximumPassthroughHandsDistance_;
	}

	private bool ShouldEnableModel(float distance)
	{
		return distance >= minimumModelHandsDistance_;
	}

	private float GetHandDistanceToKeyboard(OVRSkeleton handSkeleton)
	{
		// TODO: Switch back to PointerPose once it's working in OpenXR
		var pinchPosition = handSkeleton.Bones[(int) OVRSkeleton.BoneId.Hand_Index3].Transform.position;
		var handPosition = handSkeleton.Bones[(int) OVRSkeleton.BoneId.Hand_Middle1].Transform.position;
		var pinkyPosition = handSkeleton.Bones[(int) OVRSkeleton.BoneId.Hand_Pinky3].Transform.position;

		return Mathf.Min(KeyboardTracker.GetDistanceToKeyboard(pinchPosition),
			KeyboardTracker.GetDistanceToKeyboard(handPosition),
			KeyboardTracker.GetDistanceToKeyboard(pinkyPosition));
	}

	private float ComputeOpacity(float distance, float innerThreshold, float outerThreshold)
	{
		return Mathf.Clamp((outerThreshold - distance) / (outerThreshold - innerThreshold), 0.0f, 1.0f);
	}

	private void SetHandModelsEnabled(bool enableLeftModel, bool enableRightModel)
	{
		leftHandMeshRenderer_.enabled = enableLeftModel;
		rightHandMeshRenderer_.enabled = enableRightModel;

		leftHandSkinnedMeshRenderer_.enabled = enableLeftModel;
		rightHandSkinnedMeshRenderer_.enabled = enableRightModel;

		if (leftHandSkeletonRenderer_ != null)
		{
			if (leftHandSkeletonRendererGO_ == null)
			{
				leftHandSkeletonRendererGO_ = leftHandSkeletonRenderer_.gameObject.transform.Find("SkeletonRenderer")?.gameObject;
				rightHandSkeletonRendererGO_ = rightHandSkeletonRenderer_.gameObject.transform.Find("SkeletonRenderer")?.gameObject;
			}

			if (leftHandSkeletonRendererGO_ != null)
			{
				leftHandSkeletonRendererGO_.SetActive(enableLeftModel);
			}

			if (rightHandSkeletonRendererGO_ != null)
			{
				rightHandSkeletonRendererGO_.SetActive(enableRightModel);
			}
		}
	}

	private void RetargetHandTrackingToHandPresence()
	{
		Assert.IsTrue(LeftHandPresence != null && RightHandPresence != null);

		for (int index = 0; index < boneMappings_.Length; index++)
		{
			var entry = boneMappings_[index];

			var ovrBoneStringLeft = OVRSkeleton.BoneLabelFromBoneId(OVRSkeleton.SkeletonType.HandLeft, entry.BoneName);
			var ovrBoneStringRight = OVRSkeleton.BoneLabelFromBoneId(OVRSkeleton.SkeletonType.HandRight, entry.BoneName);

			boneMappings_[index].LeftHandTransform =
				leftHand_.transform.FindChildRecursive(ovrBoneStringLeft);
			boneMappings_[index].LeftPresenceTransform = leftHandRoot_.FindChildRecursive(entry.HandPresenceLeftBoneName);

			boneMappings_[index].RightHandTransform =
				rightHand_.transform.FindChildRecursive(ovrBoneStringRight);
			boneMappings_[index].RightPresenceTransform = rightHandRoot_.FindChildRecursive(entry.HandPresenceRightBoneName);

			Assert.IsTrue(
				boneMappings_[index].LeftPresenceTransform != null
				&& boneMappings_[index].RightPresenceTransform != null
				&& boneMappings_[index].RightHandTransform != null
				&& boneMappings_[index].LeftHandTransform != null,
				string.Format(
					"[tracked_keyboard] - entry.lp {0} && entry.rp {1} && entry.rt {2} && entry.lt {3}, {4}, {5}",
					boneMappings_[index].LeftPresenceTransform,
					boneMappings_[index].RightPresenceTransform,
					boneMappings_[index].RightHandTransform,
					boneMappings_[index].LeftHandTransform,
					ovrBoneStringRight,
					ovrBoneStringLeft
				)
			);
		}

		handPresenceInitialized_ = true;
	}

	private void StopHandPresence()
	{
		enabled = false;
		// Re-enable hand models if they are disabled, let OVRHand handle controller/hands switching
		SetHandModelsEnabled(true, true);
		DisableHandObjects();
	}

	private void DisableHandObjects()
	{
		KeyboardTracker.HandsOverKeyboard = false;
		RightHandOverKeyboard = false;
		LeftHandOverKeyboard = false;

		if (leftHandRoot_ != null)
		{
			leftHandRoot_.gameObject.SetActive(false);
		}

		if (rightHandRoot_ != null)
		{
			rightHandRoot_.gameObject.SetActive(false);
		}
	}

	public void TrackedKeyboardActiveUpdated(OVRTrackedKeyboard.TrackedKeyboardSetActiveEvent e)
	{
		if (!e.IsEnabled)
		{
			StopHandPresence();
		}
	}

	public void TrackedKeyboardVisibilityChanged(OVRTrackedKeyboard.TrackedKeyboardVisibilityChangedEvent e)
	{
		switch (e.State)
		{
			case OVRTrackedKeyboard.TrackedKeyboardState.Offline:
			case OVRTrackedKeyboard.TrackedKeyboardState.NoTrackableKeyboard:
			case OVRTrackedKeyboard.TrackedKeyboardState.StartedNotTracked:
				StopHandPresence();
				break;
			case OVRTrackedKeyboard.TrackedKeyboardState.Valid:
				enabled = handPresenceInitialized_;
				break;
			case OVRTrackedKeyboard.TrackedKeyboardState.Stale:
				if (e.TrackingTimeout)
				{
					StopHandPresence();
				}
				break;
			case OVRTrackedKeyboard.TrackedKeyboardState.Uninitialized:
			case OVRTrackedKeyboard.TrackedKeyboardState.Error:
			case OVRTrackedKeyboard.TrackedKeyboardState.ErrorExtensionFailed:
				StopHandPresence();
				Debug.LogWarning("Invalid state passed into TrackedKeyboardVisibilityChanged " + e.State.ToString());
				break;
			default:
				throw new System.Exception(
					$"[tracked_keyboard] - unhandled state: TrackedKeyboardVisibilityChanged {e.State}"
				);
		}
	}

	public struct TrackedKeyboardHandsVisibilityChangedEvent
	{
		public bool leftVisible;
		public bool rightVisible;
	}
}
