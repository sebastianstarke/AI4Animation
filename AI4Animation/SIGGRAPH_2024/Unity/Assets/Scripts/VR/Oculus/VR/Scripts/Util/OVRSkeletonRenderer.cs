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
using UnityEngine;

public class OVRSkeletonRenderer : MonoBehaviour
{
	public interface IOVRSkeletonRendererDataProvider
	{
		SkeletonRendererData GetSkeletonRendererData();
	}

	public struct SkeletonRendererData
	{
		public float RootScale { get; set; }
		public bool IsDataValid { get; set; }
		public bool IsDataHighConfidence { get; set; }
		public bool ShouldUseSystemGestureMaterial { get; set; }
	}

	public enum ConfidenceBehavior
	{
		None,
		ToggleRenderer,
	}

	public enum SystemGestureBehavior
	{
		None,
		SwapMaterial,
	}

	[SerializeField]
	private IOVRSkeletonRendererDataProvider _dataProvider;
	[SerializeField]
	private ConfidenceBehavior _confidenceBehavior = ConfidenceBehavior.ToggleRenderer;
	[SerializeField]
	private SystemGestureBehavior _systemGestureBehavior = SystemGestureBehavior.SwapMaterial;
	[SerializeField]
	private bool _renderPhysicsCapsules = false;
	[SerializeField]
	private Material _skeletonMaterial;
	private Material _skeletonDefaultMaterial;
	[SerializeField]
	private Material _capsuleMaterial;
	private Material _capsuleDefaultMaterial;
	[SerializeField]
	private Material _systemGestureMaterial = null;
	private Material _systemGestureDefaultMaterial;

	private const float LINE_RENDERER_WIDTH = 0.005f;
	private List<BoneVisualization> _boneVisualizations;
	private List<CapsuleVisualization> _capsuleVisualizations;
	private OVRSkeleton _ovrSkeleton;
	private GameObject _skeletonGO;
	private float _scale;
	private static readonly Quaternion _capsuleRotationOffset = Quaternion.Euler(0, 0, 90);

	public bool IsInitialized { get; private set; }
	public bool IsDataValid { get; private set; }
	public bool IsDataHighConfidence { get; private set; }
	public bool ShouldUseSystemGestureMaterial { get; private set; }

	private class BoneVisualization
	{
		private GameObject BoneGO;
		private Transform BoneBegin;
		private Transform BoneEnd;
		private LineRenderer Line;
		private Material RenderMaterial;
		private Material SystemGestureMaterial;

		public BoneVisualization(GameObject rootGO,
				Material renderMat,
				Material systemGestureMat,
				float scale,
				Transform begin,
				Transform end)
		{
			RenderMaterial = renderMat;
			SystemGestureMaterial = systemGestureMat;

			BoneBegin = begin;
			BoneEnd = end;

			BoneGO = new GameObject(begin.name);
			BoneGO.transform.SetParent(rootGO.transform, false);

			Line = BoneGO.AddComponent<LineRenderer>();
			Line.sharedMaterial = RenderMaterial;
			Line.useWorldSpace = true;
			Line.positionCount = 2;

			Line.SetPosition(0, BoneBegin.position);
			Line.SetPosition(1, BoneEnd.position);

			Line.startWidth = LINE_RENDERER_WIDTH * scale;
			Line.endWidth = LINE_RENDERER_WIDTH * scale;
		}

		public void Update(float scale,
				bool shouldRender,
				bool shouldUseSystemGestureMaterial,
				ConfidenceBehavior confidenceBehavior,
				SystemGestureBehavior systemGestureBehavior)
		{
			Line.SetPosition(0, BoneBegin.position);
			Line.SetPosition(1, BoneEnd.position);

			Line.startWidth = LINE_RENDERER_WIDTH * scale;
			Line.endWidth = LINE_RENDERER_WIDTH * scale;

			if (confidenceBehavior == ConfidenceBehavior.ToggleRenderer)
			{
				Line.enabled = shouldRender;
			}

			if (systemGestureBehavior == SystemGestureBehavior.SwapMaterial)
			{
				if (shouldUseSystemGestureMaterial && Line.sharedMaterial != SystemGestureMaterial)
				{
					Line.sharedMaterial = SystemGestureMaterial;
				}
				else if (!shouldUseSystemGestureMaterial && Line.sharedMaterial != RenderMaterial)
				{
					Line.sharedMaterial = RenderMaterial;
				}
			}
		}
	}

	private class CapsuleVisualization
	{
		private GameObject CapsuleGO;
		private OVRBoneCapsule BoneCapsule;
		private Vector3 capsuleScale;
		private MeshRenderer Renderer;
		private Material RenderMaterial;
		private Material SystemGestureMaterial;

		public CapsuleVisualization(GameObject rootGO,
				Material renderMat,
				Material systemGestureMat,
				float scale,
				OVRBoneCapsule boneCapsule)
		{
			RenderMaterial = renderMat;
			SystemGestureMaterial = systemGestureMat;

			BoneCapsule = boneCapsule;

			CapsuleGO = GameObject.CreatePrimitive(PrimitiveType.Capsule);
			CapsuleCollider collider = CapsuleGO.GetComponent<CapsuleCollider>();
			Destroy(collider);
			Renderer = CapsuleGO.GetComponent<MeshRenderer>();
			Renderer.sharedMaterial = RenderMaterial;

			capsuleScale = Vector3.one;
			capsuleScale.y = boneCapsule.CapsuleCollider.height / 2;
			capsuleScale.x = boneCapsule.CapsuleCollider.radius * 2;
			capsuleScale.z = boneCapsule.CapsuleCollider.radius * 2;
			CapsuleGO.transform.localScale = capsuleScale * scale;
		}

		public void Update(float scale,
				bool shouldRender,
				bool shouldUseSystemGestureMaterial,
				ConfidenceBehavior confidenceBehavior,
				SystemGestureBehavior systemGestureBehavior)
		{
			if (confidenceBehavior == ConfidenceBehavior.ToggleRenderer)
			{
				if (CapsuleGO.activeSelf != shouldRender)
				{
					CapsuleGO.SetActive(shouldRender);
				}
			}

			CapsuleGO.transform.rotation = BoneCapsule.CapsuleCollider.transform.rotation * _capsuleRotationOffset;
			CapsuleGO.transform.position = BoneCapsule.CapsuleCollider.transform.TransformPoint(BoneCapsule.CapsuleCollider.center);
			CapsuleGO.transform.localScale = capsuleScale * scale;

			if (systemGestureBehavior == SystemGestureBehavior.SwapMaterial)
			{
				if (shouldUseSystemGestureMaterial && Renderer.sharedMaterial != SystemGestureMaterial)
				{
					Renderer.sharedMaterial = SystemGestureMaterial;
				}
				else if (!shouldUseSystemGestureMaterial && Renderer.sharedMaterial != RenderMaterial)
				{
					Renderer.sharedMaterial = RenderMaterial;
				}
			}
		}
	}

	private void Awake()
	{
		if (_dataProvider == null)
		{
			_dataProvider = GetComponent<IOVRSkeletonRendererDataProvider>();
		}

		if (_ovrSkeleton == null)
		{
			_ovrSkeleton = GetComponent<OVRSkeleton>();
		}
	}

	private void Start()
	{
		if (_ovrSkeleton == null)
		{
			this.enabled = false;
			return;
		}

		if (ShouldInitialize())
		{
			Initialize();
		}
	}

	private bool ShouldInitialize()
	{
		if (IsInitialized)
		{
			return false;
		}

		return _ovrSkeleton.IsInitialized;
	}

	private void Initialize()
	{
		_boneVisualizations = new List<BoneVisualization>();
		_capsuleVisualizations = new List<CapsuleVisualization>();
		_ovrSkeleton = GetComponent<OVRSkeleton>();
		_skeletonGO = new GameObject("SkeletonRenderer");
		_skeletonGO.transform.SetParent(transform, false);

		if (_skeletonMaterial == null)
		{
			_skeletonDefaultMaterial = new Material(Shader.Find("Diffuse"));
			_skeletonMaterial = _skeletonDefaultMaterial;
		}

		if (_capsuleMaterial == null)
		{
			_capsuleDefaultMaterial = new Material(Shader.Find("Diffuse"));
			_capsuleMaterial = _capsuleDefaultMaterial;
		}

		if (_systemGestureMaterial == null)
		{
			_systemGestureDefaultMaterial = new Material(Shader.Find("Diffuse"));
			_systemGestureDefaultMaterial.color = Color.blue;
			_systemGestureMaterial = _systemGestureDefaultMaterial;
		}

		if (_ovrSkeleton.IsInitialized)
		{
			for (int i = 0; i < _ovrSkeleton.Bones.Count; i++)
			{
				var boneVis = new BoneVisualization(
					_skeletonGO,
					_skeletonMaterial,
					_systemGestureMaterial,
					_scale,
					_ovrSkeleton.Bones[i].Transform,
					_ovrSkeleton.Bones[i].Transform.parent);

				_boneVisualizations.Add(boneVis);
			}

			if (_renderPhysicsCapsules && _ovrSkeleton.Capsules != null)
			{
				for (int i = 0; i < _ovrSkeleton.Capsules.Count; i++)
				{
					var capsuleVis = new CapsuleVisualization(
						_skeletonGO,
						_capsuleMaterial,
						_systemGestureMaterial,
						_scale,
						_ovrSkeleton.Capsules[i]);

					_capsuleVisualizations.Add(capsuleVis);
				}
			}

			IsInitialized = true;
		}
	}

	public void Update()
	{
#if UNITY_EDITOR
		if (ShouldInitialize())
		{
			Initialize();
		}
#endif

		IsDataValid = false;
		IsDataHighConfidence = false;
		ShouldUseSystemGestureMaterial = false;

		if (IsInitialized)
		{
			bool shouldRender = false;

			if (_dataProvider != null)
			{
				var data = _dataProvider.GetSkeletonRendererData();

				IsDataValid = data.IsDataValid;
				IsDataHighConfidence = data.IsDataHighConfidence;
				ShouldUseSystemGestureMaterial = data.ShouldUseSystemGestureMaterial;

				shouldRender = data.IsDataValid && data.IsDataHighConfidence;

				if (data.IsDataValid)
				{
					_scale = data.RootScale;
				}
			}

			for (int i = 0; i < _boneVisualizations.Count; i++)
			{
				_boneVisualizations[i].Update(_scale, shouldRender, ShouldUseSystemGestureMaterial, _confidenceBehavior, _systemGestureBehavior);
			}

			for (int i = 0; i < _capsuleVisualizations.Count; i++)
			{
				_capsuleVisualizations[i].Update(_scale, shouldRender, ShouldUseSystemGestureMaterial, _confidenceBehavior, _systemGestureBehavior);
			}
		}
	}

	private void OnDestroy()
	{
		if (_skeletonDefaultMaterial != null)
		{
			DestroyImmediate(_skeletonDefaultMaterial, false);
		}

		if (_capsuleDefaultMaterial != null)
		{
			DestroyImmediate(_capsuleDefaultMaterial, false);
		}

		if (_systemGestureDefaultMaterial != null)
		{
			DestroyImmediate(_systemGestureDefaultMaterial, false);
		}
	}
}
