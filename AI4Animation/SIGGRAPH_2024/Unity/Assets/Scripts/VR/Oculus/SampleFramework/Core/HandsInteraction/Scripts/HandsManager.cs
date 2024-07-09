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


using System.Collections;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEngine.Assertions;

namespace OculusSampleFramework
{
	public class HandsManager : MonoBehaviour
	{
		private const string SKELETON_VISUALIZER_NAME = "SkeletonRenderer";

		[SerializeField] GameObject _leftHand = null;
		[SerializeField] GameObject _rightHand = null;

		public HandsVisualMode VisualMode = HandsVisualMode.Mesh;
		private OVRHand[] _hand = new OVRHand[(int)OVRHand.Hand.HandRight + 1];
		private OVRSkeleton[] _handSkeleton = new OVRSkeleton[(int)OVRHand.Hand.HandRight + 1];
		private OVRSkeletonRenderer[] _handSkeletonRenderer = new OVRSkeletonRenderer[(int)OVRHand.Hand.HandRight + 1];
		private OVRMesh[] _handMesh = new OVRMesh[(int)OVRHand.Hand.HandRight + 1];
		private OVRMeshRenderer[] _handMeshRenderer = new OVRMeshRenderer[(int)OVRHand.Hand.HandRight + 1];
		private SkinnedMeshRenderer _leftMeshRenderer = null;
		private SkinnedMeshRenderer _rightMeshRenderer = null;
		private GameObject _leftSkeletonVisual = null;
		private GameObject _rightSkeletonVisual = null;
		private float _currentHandAlpha = 1.0f;
		private int HandAlphaId = Shader.PropertyToID("_HandAlpha");

		public enum HandsVisualMode
		{
			Mesh = 0, Skeleton = 1, Both = 2
		}

		public OVRHand RightHand
		{
			get
			{
				return _hand[(int)OVRHand.Hand.HandRight];
			}
			private set
			{
				_hand[(int)OVRHand.Hand.HandRight] = value;
			}
		}

		public OVRSkeleton RightHandSkeleton
		{
			get
			{
				return _handSkeleton[(int)OVRHand.Hand.HandRight];
			}
			private set
			{
				_handSkeleton[(int)OVRHand.Hand.HandRight] = value;
			}
		}

		public OVRSkeletonRenderer RightHandSkeletonRenderer
		{
			get
			{
				return _handSkeletonRenderer[(int)OVRHand.Hand.HandRight];
			}
			private set
			{
				_handSkeletonRenderer[(int)OVRHand.Hand.HandRight] = value;
			}
		}

		public OVRMesh RightHandMesh
		{
			get
			{
				return _handMesh[(int)OVRHand.Hand.HandRight];
			}
			private set
			{
				_handMesh[(int)OVRHand.Hand.HandRight] = value;
			}
		}

		public OVRMeshRenderer RightHandMeshRenderer
		{
			get
			{
				return _handMeshRenderer[(int)OVRHand.Hand.HandRight];
			}
			private set
			{
				_handMeshRenderer[(int)OVRHand.Hand.HandRight] = value;
			}
		}

		public OVRHand LeftHand
		{
			get
			{
				return _hand[(int)OVRHand.Hand.HandLeft];
			}
			private set
			{
				_hand[(int)OVRHand.Hand.HandLeft] = value;
			}
		}

		public OVRSkeleton LeftHandSkeleton
		{
			get
			{
				return _handSkeleton[(int)OVRHand.Hand.HandLeft];
			}
			private set
			{
				_handSkeleton[(int)OVRHand.Hand.HandLeft] = value;
			}
		}

		public OVRSkeletonRenderer LeftHandSkeletonRenderer
		{
			get
			{
				return _handSkeletonRenderer[(int)OVRHand.Hand.HandLeft];
			}
			private set
			{
				_handSkeletonRenderer[(int)OVRHand.Hand.HandLeft] = value;
			}
		}

		public OVRMesh LeftHandMesh
		{
			get
			{
				return _handMesh[(int)OVRHand.Hand.HandLeft];
			}
			private set
			{
				_handMesh[(int)OVRHand.Hand.HandLeft] = value;
			}
		}

		public OVRMeshRenderer LeftHandMeshRenderer
		{
			get
			{
				return _handMeshRenderer[(int)OVRHand.Hand.HandLeft];
			}
			private set
			{
				_handMeshRenderer[(int)OVRHand.Hand.HandLeft] = value;
			}
		}

		public static HandsManager Instance { get; private set; }

		private void Awake()
		{
			if (Instance && Instance != this)
			{
				Destroy(this);
				return;
			}
			Instance = this;

			Assert.IsNotNull(_leftHand);
			Assert.IsNotNull(_rightHand);

			LeftHand = _leftHand.GetComponent<OVRHand>();
			LeftHandSkeleton = _leftHand.GetComponent<OVRSkeleton>();
			LeftHandSkeletonRenderer = _leftHand.GetComponent<OVRSkeletonRenderer>();
			LeftHandMesh = _leftHand.GetComponent<OVRMesh>();
			LeftHandMeshRenderer = _leftHand.GetComponent<OVRMeshRenderer>();

			RightHand = _rightHand.GetComponent<OVRHand>();
			RightHandSkeleton = _rightHand.GetComponent<OVRSkeleton>();
			RightHandSkeletonRenderer = _rightHand.GetComponent<OVRSkeletonRenderer>();
			RightHandMesh = _rightHand.GetComponent<OVRMesh>();
			RightHandMeshRenderer = _rightHand.GetComponent<OVRMeshRenderer>();
			_leftMeshRenderer = LeftHand.GetComponent<SkinnedMeshRenderer>();
			_rightMeshRenderer = RightHand.GetComponent<SkinnedMeshRenderer>();
			StartCoroutine(FindSkeletonVisualGameObjects());
		}

		private void Update()
		{
			switch (VisualMode)
			{
				case HandsVisualMode.Mesh:
				case HandsVisualMode.Skeleton:
					_currentHandAlpha = 1.0f;
					break;
				case HandsVisualMode.Both:
					_currentHandAlpha = 0.6f;
					break;
				default:
					_currentHandAlpha = 1.0f;
					break;
			}
			_rightMeshRenderer.sharedMaterial.SetFloat(HandAlphaId, _currentHandAlpha);
			_leftMeshRenderer.sharedMaterial.SetFloat(HandAlphaId, _currentHandAlpha);
		}

		private IEnumerator FindSkeletonVisualGameObjects()
		{
			while (!_leftSkeletonVisual || !_rightSkeletonVisual)
			{
				if (!_leftSkeletonVisual)
				{
					Transform leftSkeletonVisualTransform = LeftHand.transform.Find(SKELETON_VISUALIZER_NAME);
					if (leftSkeletonVisualTransform)
					{
						_leftSkeletonVisual = leftSkeletonVisualTransform.gameObject;
					}
				}

				if (!_rightSkeletonVisual)
				{
					Transform rightSkeletonVisualTransform = RightHand.transform.Find(SKELETON_VISUALIZER_NAME);
					if (rightSkeletonVisualTransform)
					{
						_rightSkeletonVisual = rightSkeletonVisualTransform.gameObject;
					}
				}
				yield return null;
			}
			SetToCurrentVisualMode();
		}

		public void SwitchVisualization()
		{
			if (!_leftSkeletonVisual || !_rightSkeletonVisual)
			{
				return;
			}
			VisualMode = (HandsVisualMode)(((int)VisualMode + 1) % ((int)HandsVisualMode.Both + 1));
			SetToCurrentVisualMode();
		}

		private void SetToCurrentVisualMode()
		{
			switch (VisualMode)
			{
				case HandsVisualMode.Mesh:
					RightHandMeshRenderer.enabled = true;
					_rightMeshRenderer.enabled = true;
					_rightSkeletonVisual.gameObject.SetActive(false);
					LeftHandMeshRenderer.enabled = true;
					_leftMeshRenderer.enabled = true;
					_leftSkeletonVisual.gameObject.SetActive(false);
					break;
				case HandsVisualMode.Skeleton:
					RightHandMeshRenderer.enabled = false;
					_rightMeshRenderer.enabled = false;
					_rightSkeletonVisual.gameObject.SetActive(true);
					LeftHandMeshRenderer.enabled = false;
					_leftMeshRenderer.enabled = false;
					_leftSkeletonVisual.gameObject.SetActive(true);
					break;
				case HandsVisualMode.Both:
					RightHandMeshRenderer.enabled = true;
					_rightMeshRenderer.enabled = true;
					_rightSkeletonVisual.gameObject.SetActive(true);
					LeftHandMeshRenderer.enabled = true;
					_leftMeshRenderer.enabled = true;
					_leftSkeletonVisual.gameObject.SetActive(true);
					break;
				default:
					break;
			}
		}

		public static List<OVRBoneCapsule> GetCapsulesPerBone(OVRSkeleton skeleton, OVRSkeleton.BoneId boneId)
		{
			List<OVRBoneCapsule> boneCapsules = new List<OVRBoneCapsule>();
			var capsules = skeleton.Capsules;
			for (int i = 0; i < capsules.Count; ++i)
			{
				if (capsules[i].BoneIndex == (short)boneId)
				{
					boneCapsules.Add(capsules[i]);
				}
			}
			return boneCapsules;
		}

		public bool IsInitialized()
		{
			return LeftHandSkeleton && LeftHandSkeleton.IsInitialized &&
				RightHandSkeleton && RightHandSkeleton.IsInitialized &&
				LeftHandMesh && LeftHandMesh.IsInitialized &&
				RightHandMesh && RightHandMesh.IsInitialized;
		}
	}
}
