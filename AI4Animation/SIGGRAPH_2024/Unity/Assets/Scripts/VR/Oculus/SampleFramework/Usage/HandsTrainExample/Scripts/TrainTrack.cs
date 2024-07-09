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
	public class TrainTrack : MonoBehaviour
	{
		[SerializeField] private float _gridSize = 0.5f;
		[SerializeField] private int _subDivCount = 20;
		[SerializeField] private Transform _segmentParent = null;
		[SerializeField] private Transform _trainParent = null;
		// regeneration is optional
		[SerializeField] private bool _regnerateTrackMeshOnAwake = false;

		private float _trainLength = -1.0f;
		private TrackSegment[] _trackSegments = null;

		public float TrackLength
		{
			get
			{
				return _trainLength;
			}
			private set
			{
				_trainLength = value;
			}
		}

		private void Awake()
		{
			Assert.IsNotNull(_segmentParent);
			Assert.IsNotNull(_trainParent);
			Regenerate();
		}

		public TrackSegment GetSegment(float distance)
		{
			int childCount = _segmentParent.childCount;

			for (int i = 0; i < childCount; i++)
			{
				var segment = _trackSegments[i];
				var nextSegment = _trackSegments[(i + 1) % childCount];
				if (distance >= segment.StartDistance && (distance < nextSegment.StartDistance || i == childCount - 1))
				{
					return segment;
				}
			}

			return null;
		}

		public void Regenerate()
		{
			_trackSegments = _segmentParent.GetComponentsInChildren<TrackSegment>();
			TrackLength = 0;
			int childCount = _segmentParent.childCount;
			TrackSegment lastSegment = null;

			var ratio = 0.0f;
			for (int i = 0; i < childCount; i++)
			{
				var segment = _trackSegments[i];
				segment.SubDivCount = _subDivCount;
				ratio = segment.setGridSize(_gridSize);
				if (lastSegment != null)
				{
					var endPose = lastSegment.EndPose;
					segment.transform.position = endPose.Position;
					segment.transform.rotation = endPose.Rotation;
					segment.StartDistance = TrackLength;
				}

				if (_regnerateTrackMeshOnAwake)
				{
					segment.RegenerateTrackAndMesh();
				}

				TrackLength += segment.SegmentLength;
				lastSegment = segment;
			}

			SetScale(ratio);
		}

		private void SetScale(float ratio)
		{
			_trainParent.localScale = new Vector3(ratio, ratio, ratio);
			var cars = _trainParent.GetComponentsInChildren<TrainCar>();
			var locomotive = _trainParent.GetComponentInChildren<TrainLocomotive>();
			locomotive.Scale = ratio;
			foreach (var car in cars)
			{
				car.Scale = ratio;
			}
		}
	}
}
