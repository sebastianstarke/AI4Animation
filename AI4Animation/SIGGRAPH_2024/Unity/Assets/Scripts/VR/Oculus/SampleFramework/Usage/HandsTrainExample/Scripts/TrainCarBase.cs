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
	public abstract class TrainCarBase : MonoBehaviour
	{
		private static Vector3 OFFSET = new Vector3(0f, 0.0195f, 0f);
		private const float WHEEL_RADIUS = 0.027f;
		private const float TWO_PI = Mathf.PI * 2.0f;

		[SerializeField] protected Transform _frontWheels = null;
		[SerializeField] protected Transform _rearWheels = null;
		[SerializeField] protected TrainTrack _trainTrack = null;
		[SerializeField] protected Transform[] _individualWheels = null;

		public float Distance { get; protected set; }
		protected float scale = 1.0f;

		private Pose _frontPose = new Pose(), _rearPose = new Pose();

		public float Scale
		{
			get { return scale; }
			set { scale = value; }
		}

		protected virtual void Awake()
		{
			Assert.IsNotNull(_frontWheels);
			Assert.IsNotNull(_rearWheels);
			Assert.IsNotNull(_trainTrack);
			Assert.IsNotNull(_individualWheels);
		}

		public void UpdatePose(float distance, TrainCarBase train, Pose pose)
		{
			// distance could be negative; add track length to it in case that happens
			distance = (train._trainTrack.TrackLength + distance) % train._trainTrack.TrackLength;
			if (distance < 0)
			{
				distance += train._trainTrack.TrackLength;
			}

			var currentSegment = train._trainTrack.GetSegment(distance);
			var distanceInto = distance - currentSegment.StartDistance;

			currentSegment.UpdatePose(distanceInto, pose);
		}

		protected void UpdateCarPosition()
		{
			UpdatePose(Distance + _frontWheels.transform.localPosition.z * scale,
			  this, _frontPose);
			UpdatePose(Distance + _rearWheels.transform.localPosition.z * scale,
			  this, _rearPose);

			var midPoint = 0.5f * (_frontPose.Position + _rearPose.Position);
			var carLookDirection = _frontPose.Position - _rearPose.Position;

			transform.position = midPoint + OFFSET;
			transform.rotation = Quaternion.LookRotation(carLookDirection, transform.up);
			_frontWheels.transform.rotation = _frontPose.Rotation;
			_rearWheels.transform.rotation = _rearPose.Rotation;
		}

		protected void RotateCarWheels()
		{
			// divide by radius to get angle
			float angleOfRot = (Distance / WHEEL_RADIUS) % TWO_PI;

			foreach (var individualWheel in _individualWheels)
			{
				individualWheel.localRotation = Quaternion.AngleAxis(Mathf.Rad2Deg * angleOfRot,
				  Vector3.right);
			}
		}

		public abstract void UpdatePosition();
	}
}
