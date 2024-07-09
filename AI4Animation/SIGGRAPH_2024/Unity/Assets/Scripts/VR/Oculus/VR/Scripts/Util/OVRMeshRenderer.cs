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

public class OVRMeshRenderer : MonoBehaviour
{
	public interface IOVRMeshRendererDataProvider
	{
		MeshRendererData GetMeshRendererData();
	}

	public struct MeshRendererData
	{
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
	private IOVRMeshRendererDataProvider _dataProvider;
	[SerializeField]
	private OVRMesh _ovrMesh;
	[SerializeField]
	private OVRSkeleton _ovrSkeleton;
	[SerializeField]
	private ConfidenceBehavior _confidenceBehavior = ConfidenceBehavior.ToggleRenderer;
	[SerializeField]
	private SystemGestureBehavior _systemGestureBehavior = SystemGestureBehavior.SwapMaterial;
	[SerializeField]
	private Material _systemGestureMaterial = null;
	private Material _originalMaterial = null;

	private SkinnedMeshRenderer _skinnedMeshRenderer;

	public bool IsInitialized { get; private set; }
	public bool IsDataValid { get; private set; }
	public bool IsDataHighConfidence { get; private set; }
	public bool ShouldUseSystemGestureMaterial { get; private set; }

	private void Awake()
	{
		if (_dataProvider == null)
		{
			_dataProvider = GetComponent<IOVRMeshRendererDataProvider>();
		}

		if (_ovrMesh == null)
		{
			_ovrMesh = GetComponent<OVRMesh>();
		}

		if (_ovrSkeleton == null)
		{
			_ovrSkeleton = GetComponent<OVRSkeleton>();
		}
	}

	private void Start()
	{
		if (_ovrMesh == null)
		{
			// disable if no mesh configured
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

		if ((_ovrMesh == null) || ((_ovrMesh != null) && !_ovrMesh.IsInitialized) || ((_ovrSkeleton != null) && !_ovrSkeleton.IsInitialized))
		{
			// do not initialize if mesh or optional skeleton are not initialized
			return false;
		}

		return true;
	}

	private void Initialize()
	{
		_skinnedMeshRenderer = GetComponent<SkinnedMeshRenderer>();
		if (!_skinnedMeshRenderer)
		{
			_skinnedMeshRenderer = gameObject.AddComponent<SkinnedMeshRenderer>();
		}

		_skinnedMeshRenderer.sharedMesh = _ovrMesh.Mesh;
		_originalMaterial = _skinnedMeshRenderer.sharedMaterial;

		if ((_ovrSkeleton != null))
		{
			int numSkinnableBones = _ovrSkeleton.GetCurrentNumSkinnableBones();
			var bindPoses = new Matrix4x4[numSkinnableBones];
			var bones = new Transform[numSkinnableBones];
			var localToWorldMatrix = transform.localToWorldMatrix;
			for (int i = 0; i < numSkinnableBones && i < _ovrSkeleton.Bones.Count; ++i)
			{
				bones[i] = _ovrSkeleton.Bones[i].Transform;
				bindPoses[i] = _ovrSkeleton.BindPoses[i].Transform.worldToLocalMatrix * localToWorldMatrix;
			}
			_ovrMesh.Mesh.bindposes = bindPoses;
			_skinnedMeshRenderer.bones = bones;
			_skinnedMeshRenderer.updateWhenOffscreen = true;
		}

		IsInitialized = true;
	}

	private void Update()
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
				var data = _dataProvider.GetMeshRendererData();

				IsDataValid = data.IsDataValid;
				IsDataHighConfidence = data.IsDataHighConfidence;
				ShouldUseSystemGestureMaterial = data.ShouldUseSystemGestureMaterial;

				shouldRender = data.IsDataValid && data.IsDataHighConfidence;
			}

			if (_confidenceBehavior == ConfidenceBehavior.ToggleRenderer)
			{
				if (_skinnedMeshRenderer != null && _skinnedMeshRenderer.enabled != shouldRender)
				{
					_skinnedMeshRenderer.enabled = shouldRender;
				}
			}

			if (_systemGestureBehavior == SystemGestureBehavior.SwapMaterial)
			{
				if (_skinnedMeshRenderer != null)
				{
					if (ShouldUseSystemGestureMaterial && _systemGestureMaterial != null && _skinnedMeshRenderer.sharedMaterial != _systemGestureMaterial)
					{
						_skinnedMeshRenderer.sharedMaterial = _systemGestureMaterial;
					}
					else if (!ShouldUseSystemGestureMaterial && _originalMaterial != null && _skinnedMeshRenderer.sharedMaterial != _originalMaterial)
					{
						_skinnedMeshRenderer.sharedMaterial = _originalMaterial;
					}
				}
			}
		}
	}
}
