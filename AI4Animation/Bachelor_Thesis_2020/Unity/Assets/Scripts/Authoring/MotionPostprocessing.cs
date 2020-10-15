using UnityEngine;
using System;
using System.Collections.Generic;

namespace SIGGRAPH_2018 {
	[RequireComponent(typeof(Actor))]
	[RequireComponent(typeof(Runtime))]
	public class MotionPostprocessing : MonoBehaviour {

		public FootIK[] LegSolvers;
		public SerialIK TailSolver;

		private Actor Actor;
		private Runtime Animation;

		private Actor GetActor() {
			if(Actor == null) {
				Actor = GetComponent<Actor>();
			}
			return Actor;
		}

		private Runtime GetAnimation() {
			if(Animation == null) {
				Animation = GetComponent<Runtime>();
			}
			return Animation;
		}

		public void Process(float[] contacts) {
			GetAnimation().GetTrajectory().Postprocess();
			transform.position = GetAnimation().GetTrajectory().Points[60].GetPosition();

			Vector3[] pivotPositions = new Vector3[LegSolvers.Length];
			Quaternion[] pivotRotations = new Quaternion[LegSolvers.Length];
			for(int i=0; i<LegSolvers.Length; i++) {
				pivotPositions[i] = LegSolvers[i].GetPivotPosition();
				pivotRotations[i] = LegSolvers[i].GetPivotRotation();
			}

			Transform hips = Array.Find(GetActor().Bones, x => x.Transform.name == "Hips").Transform;
			Transform spine = Array.Find(GetActor().Bones, x => x.Transform.name == "Spine1").Transform;
			Transform neck = Array.Find(GetActor().Bones, x => x.Transform.name == "Neck").Transform;
			Transform leftShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftShoulder").Transform;
			Transform rightShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "RightShoulder").Transform;

			Vector3 hipsPosition = hips.position;
			Vector3 spinePosition = spine.position;
			Vector3 neckPosition = neck.position;
			Vector3 leftShoulderPosition = leftShoulder.position;
			Vector3 rightShoulderPosition = rightShoulder.position;

			ControlPoint p = GetAnimation().AnimationAuthoring.GetControlPoint(GetAnimation().AnimationAuthoring.RefTimestamp, 0);
			
			float spineHeight = Utility.GetHeight(spine.position, p.Ground);
			float neckHeight = Utility.GetHeight(neck.position, p.Ground);
			float leftShoulderHeight = Utility.GetHeight(leftShoulder.position, p.Ground);
			float rightShoulderHeight = Utility.GetHeight(rightShoulder.position, p.Ground);
			hips.rotation = Quaternion.Slerp(hips.rotation, Quaternion.FromToRotation(neckPosition - hipsPosition, new Vector3(neckPosition.x, neckHeight + (neckPosition.y - transform.position.y), neckPosition.z) - hipsPosition) * hips.rotation, 0.5f);
			spine.rotation = Quaternion.Slerp(spine.rotation, Quaternion.FromToRotation(neckPosition - spinePosition, new Vector3(neckPosition.x, neckHeight + (neckPosition.y - transform.position.y), neckPosition.z) - spinePosition) * spine.rotation, 0.5f);
			spine.position = new Vector3(spinePosition.x, spineHeight + (spinePosition.y - transform.position.y), spinePosition.z);
			neck.position = new Vector3(neckPosition.x, neckHeight + (neckPosition.y - transform.position.y), neckPosition.z);
			leftShoulder.position = new Vector3(leftShoulderPosition.x, leftShoulderHeight + (leftShoulderPosition.y - transform.position.y), leftShoulderPosition.z);
			rightShoulder.position = new Vector3(rightShoulderPosition.x, rightShoulderHeight + (rightShoulderPosition.y - transform.position.y), rightShoulderPosition.z);

			for(int i=0; i<LegSolvers.Length; i++) {
				LegSolvers[i].Solve(pivotPositions[i], pivotRotations[i], contacts[i]);
			}
			TailSolver.TargetPosition = TailSolver.EndEffector.position;
			TailSolver.TargetPosition.y = Utility.GetHeight(TailSolver.TargetPosition, p.Ground) + Mathf.Max(0f, (TailSolver.TargetPosition.y - transform.position.y));
			TailSolver.Solve();
		}

	}
}