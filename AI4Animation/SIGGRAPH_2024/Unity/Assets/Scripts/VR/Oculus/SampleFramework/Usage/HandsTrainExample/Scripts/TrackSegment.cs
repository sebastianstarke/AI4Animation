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
	public class TrackSegment : MonoBehaviour
	{
		public enum SegmentType
		{
			Straight = 0,
			LeftTurn,
			RightTurn,
			Switch
		}

		[SerializeField] private SegmentType _segmentType = SegmentType.Straight;
		[SerializeField] private MeshFilter _straight = null;
		[SerializeField] private MeshFilter _leftTurn = null;
		[SerializeField] private MeshFilter _rightTurn = null;
		private float _gridSize = 0.8f;
		private int _subDivCount = 20;
		private const float _originalGridSize = 0.8f;
		private const float _trackWidth = 0.15f;
		private GameObject _mesh = null;
		public float StartDistance { get; set; }

		// create variables here to avoid realtime allocation
		private Pose _p1 = new Pose();
		private Pose _p2 = new Pose();

		public float GridSize
		{
			get
			{
				return _gridSize;
			}
			private set
			{
				_gridSize = value;
			}
		}

		public int SubDivCount
		{
			get
			{
				return _subDivCount;
			}
			set
			{
				_subDivCount = value;
			}
		}

		public SegmentType Type
		{
			get
			{
				return _segmentType;
			}
		}

		private Pose _endPose = new Pose();
		public Pose EndPose
		{
			get
			{
				UpdatePose(SegmentLength, _endPose);
				return _endPose;
			}
		}
		public float Radius
		{
			get
			{
				return 0.5f * GridSize;
			}
		}

		public float setGridSize(float size)
		{
			GridSize = size;
			return GridSize / _originalGridSize;
		}

		public float SegmentLength
		{
			get
			{
				switch (Type)
				{
					case SegmentType.Straight:
						return GridSize;
					case SegmentType.LeftTurn:
					case SegmentType.RightTurn:
						// return quarter of circumference.
						return 0.5f * Mathf.PI * Radius;
				}

				return 1f;
			}
		}

		private void Awake()
		{
			Assert.IsNotNull(_straight);
			Assert.IsNotNull(_leftTurn);
			Assert.IsNotNull(_rightTurn);
		}

		/// <summary>
		/// Updates pose given distance into segment. While this mutates a value,
		/// it avoids generating a new object.
		/// </summary>
		public void UpdatePose(float distanceIntoSegment, Pose pose)
		{
			if (Type == SegmentType.Straight)
			{
				pose.Position = transform.position + distanceIntoSegment * transform.forward;
				pose.Rotation = transform.rotation;
			}
			else if (Type == SegmentType.LeftTurn)
			{
				float normalizedDistanceIntoSegment = distanceIntoSegment / SegmentLength;
				// the turn is 90 degrees, so find out how far we are into it
				float angle = 0.5f * Mathf.PI * normalizedDistanceIntoSegment;
				// unity is left handed so the rotations go the opposite directions in X for left turns --
				// invert that by subtracting by radius. also note the angle negation below
				Vector3 localPosition = new Vector3(Radius * Mathf.Cos(angle) - Radius, 0, Radius * Mathf.Sin(angle));
				Quaternion localRotation = Quaternion.Euler(0, -angle * Mathf.Rad2Deg, 0);
				pose.Position = transform.TransformPoint(localPosition);
				pose.Rotation = transform.rotation * localRotation;
			}
			else if (Type == SegmentType.RightTurn)
			{
				// when going to right, start from PI (180) and go toward 90
				float angle = Mathf.PI - 0.5f * Mathf.PI * distanceIntoSegment / SegmentLength;
				// going right means we start at radius distance away, and decrease toward zero
				Vector3 localPosition = new Vector3(Radius * Mathf.Cos(angle) + Radius, 0, Radius * Mathf.Sin(angle));
				Quaternion localRotation = Quaternion.Euler(0, (Mathf.PI - angle) * Mathf.Rad2Deg, 0);
				pose.Position = transform.TransformPoint(localPosition);
				pose.Rotation = transform.rotation * localRotation;
			}
			else
			{
				pose.Position = Vector3.zero;
				pose.Rotation = Quaternion.identity;
			}
		}

		private void Update()
		{
			// uncomment to debug the track path
			//DrawDebugLines();
		}

		private void OnDisable()
		{
			Destroy(_mesh);
		}

		private void DrawDebugLines()
		{
			for (int i = 1; i < SubDivCount + 1; i++)
			{
				float len = SegmentLength / SubDivCount;
				UpdatePose((i - 1) * len, _p1);
				UpdatePose(i * len, _p2);
				// right segment from p1 to p2
				var halfTrackWidth = 0.5f * _trackWidth;
				Debug.DrawLine(_p1.Position + halfTrackWidth * (_p1.Rotation * Vector3.right),
					_p2.Position + halfTrackWidth * (_p2.Rotation * Vector3.right));
				// left segment from p1 to p2
				Debug.DrawLine(_p1.Position - halfTrackWidth * (_p1.Rotation * Vector3.right),
					_p2.Position - halfTrackWidth * (_p2.Rotation * Vector3.right));
			}

			// bottom bound
			Debug.DrawLine(transform.position - 0.5f * GridSize * transform.right,
				transform.position + 0.5f * GridSize * transform.right, Color.yellow);
			// left bound
			Debug.DrawLine(transform.position - 0.5f * GridSize * transform.right,
				transform.position - 0.5f * GridSize * transform.right + GridSize * transform.forward, Color.yellow);
			// right bound
			Debug.DrawLine(transform.position + 0.5f * GridSize * transform.right,
				transform.position + 0.5f * GridSize * transform.right + GridSize * transform.forward, Color.yellow);
			// top bound
			Debug.DrawLine(transform.position - 0.5f * GridSize * transform.right +
				GridSize * transform.forward, transform.position +
				0.5f * GridSize * transform.right + GridSize * transform.forward,
				Color.yellow);
		}

		public void RegenerateTrackAndMesh()
		{
			if (transform.childCount > 0 && !_mesh)
			{
				_mesh = transform.GetChild(0).gameObject;
			}

			if (_mesh)
			{
				DestroyImmediate(_mesh);
			}

			if (_segmentType == SegmentType.LeftTurn)
			{
				_mesh = Instantiate(_leftTurn.gameObject);
			}
			else if (_segmentType == SegmentType.RightTurn)
			{
				_mesh = Instantiate(_rightTurn.gameObject);
			}
			else
			{
				_mesh = Instantiate(_straight.gameObject);
			}

			_mesh.transform.SetParent(transform, false);
			_mesh.transform.position += GridSize / 2.0f * transform.forward;
			_mesh.transform.localScale = new Vector3(GridSize / _originalGridSize, GridSize / _originalGridSize,
			  GridSize / _originalGridSize);
		}
	}
}
