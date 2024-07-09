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

/// <summary>
/// A <see cref="OVRSceneAnchor"/> that has a 3D bounds associated with it.
/// </summary>
[DisallowMultipleComponent]
[RequireComponent(typeof(OVRSceneAnchor))]
public class OVRSceneVolume : MonoBehaviour, IOVRSceneComponent
{
	/// <summary>
	/// The width (in the local X-direction), in meters.
	/// </summary>
	public float Width { get; private set; }

	/// <summary>
	/// The height (in the local Y-direction), in meters.
	/// </summary>
	public float Height { get; private set; }

	/// <summary>
	/// The depth (in the local Z-direction), in meters.
	/// </summary>
	public float Depth { get; private set; }

	/// <summary>
	/// The dimensions of the volume.
	/// </summary>
	/// <remarks>
	/// This property corresponds to a Vector whose components are
	/// (<see cref="Width"/>, <see cref="Height"/>, <see cref="Depth"/>).
	/// </remarks>
	public Vector3 Dimensions => new Vector3(Width, Height, Depth);

	/// <summary>
	/// Whether the child transforms will be scaled according to the dimensions of this volume.
	/// </summary>
	/// <remarks>If set to True, all the child transforms will be scaled to the dimensions of this volume immediately.
	/// And, if it's set to False, dimensions of this volume will no longer affect the child transforms, and child
	/// transforms will retain their current scale.</remarks>
	public bool ScaleChildren
	{
		get => _scaleChildren;
		set {
			_scaleChildren = value;
			if(_scaleChildren && _sceneAnchor.Space.Valid)
			{
				SetChildScale(transform, Dimensions);
			}
		}
	}

	[Tooltip("When enabled, scales the child transforms according to the dimensions of this volume")]
	[SerializeField]
	private bool _scaleChildren = true;

	private OVRSceneAnchor _sceneAnchor;

	private void Awake()
	{
		_sceneAnchor = GetComponent<OVRSceneAnchor>();
		if (_sceneAnchor.Space.Valid)
		{
			((IOVRSceneComponent)this).Initialize();
		}
	}

	void IOVRSceneComponent.Initialize()
	{
		if (OVRPlugin.GetSpaceBoundingBox3D(_sceneAnchor.Space, out var bounds))
		{
			Width = bounds.Size.w;
			Height = bounds.Size.h;
			Depth = bounds.Size.d;

			var dimensions = Dimensions;
			OVRSceneManager.Development.Log(nameof(OVRSceneVolume),
				$"[{_sceneAnchor.Uuid}] Volume has dimensions {dimensions}.");

			if (ScaleChildren)
			{
				SetChildScale(transform, dimensions);
			}
		}
		else
		{
			OVRSceneManager.Development.LogError(nameof(OVRSceneVolume),
				$"[{_sceneAnchor.Space}] Failed to retrieve volume's dimensions.");
		}
	}

	private static void SetChildScale(Transform parentTransform, Vector3 dimensions)
	{
		for (var i = 0; i < parentTransform.childCount; i++)
		{
			parentTransform.GetChild(i).localScale = dimensions;
		}
	}
}
