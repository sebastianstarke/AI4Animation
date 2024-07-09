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

namespace OculusSampleFramework
{
	/// <summary>
	/// Visual portion of ray tool.
	/// </summary>
	public class RayToolView : MonoBehaviour, InteractableToolView
	{
		private const int NUM_RAY_LINE_POSITIONS = 25;
		private const float DEFAULT_RAY_CAST_DISTANCE = 3.0f;

		[SerializeField] private Transform _targetTransform = null;
		[SerializeField] private LineRenderer _lineRenderer = null;

		public bool EnableState
		{
			get
			{
				return _lineRenderer.enabled;
			}
			set
			{
				_targetTransform.gameObject.SetActive(value);
				_lineRenderer.enabled = value;
			}
		}

		private bool _toolActivateState = false;

		public bool ToolActivateState
		{
			get { return _toolActivateState; }
			set
			{
				_toolActivateState = value;
				_lineRenderer.colorGradient = _toolActivateState ? _highLightColorGradient : _oldColorGradient;
			}
		}

		private Transform _focusedTransform = null;
		private Vector3[] linePositions = new Vector3[NUM_RAY_LINE_POSITIONS];
		private Gradient _oldColorGradient, _highLightColorGradient;

		private void Awake()
		{
			Assert.IsNotNull(_targetTransform);
			Assert.IsNotNull(_lineRenderer);
			_lineRenderer.positionCount = NUM_RAY_LINE_POSITIONS;

			_oldColorGradient = _lineRenderer.colorGradient;
			_highLightColorGradient = new Gradient();
			_highLightColorGradient.SetKeys(
			  new GradientColorKey[] { new GradientColorKey(new Color(0.90f, 0.90f, 0.90f), 0.0f),
		  new GradientColorKey(new Color(0.90f, 0.90f, 0.90f), 1.0f) },
			  new GradientAlphaKey[] { new GradientAlphaKey(1.0f, 0.0f), new GradientAlphaKey(1.0f, 1.0f) }
			);
		}

		public InteractableTool InteractableTool { get; set; }

		public void SetFocusedInteractable(Interactable interactable)
		{
			if (interactable == null)
			{
				_focusedTransform = null;
			}
			else
			{
				_focusedTransform = interactable.transform;
			}
		}

		private void Update()
		{
			var myPosition = InteractableTool.ToolTransform.position;
			var myForward = InteractableTool.ToolTransform.forward;

			var targetPosition = _focusedTransform != null
			  ? _focusedTransform.position
			  : myPosition + myForward * DEFAULT_RAY_CAST_DISTANCE;
			var targetVector = targetPosition - myPosition;
			var targetDistance = targetVector.magnitude;
			var p0 = myPosition;
			// make points in between based on my forward as opposed to targetvector
			// this way the curve "bends" toward to target
			var p1 = myPosition + myForward * targetDistance * 0.3333333f;
			var p2 = myPosition + myForward * targetDistance * 0.6666667f;
			var p3 = targetPosition;
			for (int i = 0; i < NUM_RAY_LINE_POSITIONS; i++)
			{
				linePositions[i] = GetPointOnBezierCurve(p0, p1, p2, p3, i / 25.0f);
			}

			_lineRenderer.SetPositions(linePositions);
			_targetTransform.position = targetPosition;
		}

		/// <summary>
		/// Returns point on four-point Bezier curve.
		/// </summary>
		/// <param name="p0">Beginning point.</param>
		/// <param name="p1">t=1/3 point.</param>
		/// <param name="p2">t=2/3 point.</param>
		/// <param name="p3">End point.</param>
		/// <param name="t">Interpolation parameter.</param>
		/// <returns>Point along Bezier curve.</returns>
		public static Vector3 GetPointOnBezierCurve(Vector3 p0, Vector3 p1, Vector3 p2, Vector3 p3, float t)
		{
			t = Mathf.Clamp01(t);
			var oneMinusT = 1f - t;
			var oneMinusTSqr = oneMinusT * oneMinusT;
			var tSqr = t * t;
			return oneMinusT * oneMinusTSqr * p0 + 3f * oneMinusTSqr * t * p1 + 3f * oneMinusT * tSqr * p2 +
				t * tSqr * p3;
		}
	}
}

