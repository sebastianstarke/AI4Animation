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

namespace OculusSampleFramework
{
	public class SelectionCylinder : MonoBehaviour
	{
		public enum SelectionState
		{
			Off = 0,
			Selected,
			Highlighted
		}

		[SerializeField] private MeshRenderer _selectionMeshRenderer = null;

		private static int _colorId = Shader.PropertyToID("_Color");
		private Material[] _selectionMaterials;
		private Color[] _defaultSelectionColors = null, _highlightColors = null;

		private SelectionState _currSelectionState = SelectionState.Off;

		public SelectionState CurrSelectionState
		{
			get { return _currSelectionState; }
			set
			{
				var oldState = _currSelectionState;
				_currSelectionState = value;

				if (oldState != _currSelectionState)
				{
					if (_currSelectionState > SelectionState.Off)
					{
						_selectionMeshRenderer.enabled = true;
						AffectSelectionColor(_currSelectionState == SelectionState.Selected
							? _defaultSelectionColors
							: _highlightColors);
					}
					else
					{
						_selectionMeshRenderer.enabled = false;
					}
				}
			}
		}

		private void Awake()
		{
			_selectionMaterials = _selectionMeshRenderer.materials;
			int numColors = _selectionMaterials.Length;
			_defaultSelectionColors = new Color[numColors];
			_highlightColors = new Color[numColors];
			for (int i = 0; i < numColors; i++)
			{
				_defaultSelectionColors[i] = _selectionMaterials[i].GetColor(_colorId);
				_highlightColors[i] = new Color(1.0f, 1.0f, 1.0f, _defaultSelectionColors[i].a);
			}

			CurrSelectionState = SelectionState.Off;
		}

		private void OnDestroy()
		{
			if (_selectionMaterials != null)
			{
				foreach (var selectionMaterial in _selectionMaterials)
				{
					if (selectionMaterial != null)
					{
						Destroy(selectionMaterial);
					}
				}
			}
		}

		private void AffectSelectionColor(Color[] newColors)
		{
			int numColors = newColors.Length;
			for (int i = 0; i < numColors; i++)
			{
				_selectionMaterials[i].SetColor(_colorId, newColors[i]);
			}
		}
	}
}
