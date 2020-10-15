using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace SIGGRAPH_2018
{	

	[RequireComponent(typeof(Actor))]
	public class IKStyle : MonoBehaviour
	{
		private Actor Actor;
		public Transform Hips;

		public Transform LeftHand;
		public Transform RightHand;
		public Transform LeftFoot;
		public Transform RightFoot;
		public Transform Target;
		public Transform Target_Leg;

		private UltimateIK.UltimateIK.Model ModelLH;
		private UltimateIK.UltimateIK.Model ModelRH;
		private UltimateIK.UltimateIK.Model ModelLF;
		private UltimateIK.UltimateIK.Model ModelRF;

		private UltimateIK.UltimateIK.Model ModelHead;

		private Matrix4x4[] Bones;

		void Awake()
		{
			//Bones = GetActor().GetPosture();
			//Sneak(1);
			//Hydrate();
		}

		void Update()
		{
			//GetActor().SetPosture(Bones);
			//Hydrate();
		}
		public void Sneak(float intensity)
		{
			if ((Hips.position.y - (intensity / 8)) < Utility.ProjectGround(Hips.position,LayerMask.GetMask("Ground")).y + 0.07f) return;
			Transform leftUpLeg = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftUpLeg").Transform;
			Transform rightUpLeg = Array.Find(GetActor().Bones, x => x.Transform.name == "RightUpLeg").Transform;
			Transform leftShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftShoulder").Transform;
			Transform rightShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "RightShoulder").Transform;

			ModelLF = UltimateIK.UltimateIK.BuildModel(leftUpLeg, LeftFoot);

			ModelRF = UltimateIK.UltimateIK.BuildModel(rightUpLeg, RightFoot);

			ModelLH = UltimateIK.UltimateIK.BuildModel(leftShoulder, LeftHand);

			ModelRH = UltimateIK.UltimateIK.BuildModel(rightShoulder, RightHand);
			ModelLF.Objectives[0].SetTarget(LeftFoot.position);
			ModelRF.Objectives[0].SetTarget(RightFoot.position);
			ModelLH.Objectives[0].SetTarget(LeftHand.position);
			ModelRH.Objectives[0].SetTarget(RightHand.position);

			Hips.position = new Vector3(Hips.position.x, Hips.position.y - (intensity / 8), Hips.position.z);
			//hips.rotation = Quaternion.Slerp(hips.rotation, Quaternion.FromToRotation(neckPosition - hipsPosition, new Vector3(neckPosition.x, neckHeight + (neckPosition.y - transform.position.y), neckPosition.z) - hipsPosition) * hips.rotation, 0.5f);

			ModelLH.Solve();
			ModelRH.Solve();
			ModelLF.Solve();
			ModelRF.Solve();

			Transform neck = Array.Find(GetActor().Bones, x => x.Transform.name == "Neck").Transform;
			Transform head = Array.Find(GetActor().Bones, x => x.Transform.name == "Head").Transform;
			Transform tail = Array.Find(GetActor().Bones, x => x.Transform.name == "Tail").Transform;

			
			
			neck.rotation *= Quaternion.AngleAxis(20f, Vector3.forward);
			head.rotation *= Quaternion.AngleAxis(10f, Vector3.forward);
			tail.rotation *= Quaternion.AngleAxis(25f, Vector3.forward);
				
			

		}

		public void Eat()
		{
			Transform Spine = Array.Find(GetActor().Bones, x => x.Transform.name == "Spine").Transform;
			Transform head = Array.Find(GetActor().Bones, x => x.Transform.name == "Head").Transform;

			Vector3 playerPos = GetActor().transform.position;
			Vector3 playerDirection = GetActor().transform.forward;
			Quaternion playerRotation = GetActor().transform.rotation;
			float spawnDistance = 0.55f;

			Vector3 pos = Utility.ProjectGround(playerPos+playerDirection*spawnDistance, LayerMask.GetMask("Ground"));
			Target.position = pos;
			//if(Target.position.y < Utility.ProjectGround(new Vector3(LeftFoot.position.x, LeftFoot.position.y, LeftFoot.position.z + 0.5f), LayerMask.GetMask("Ground")).y) Utility.ProjectGround(new Vector3(LeftFoot.position.x, LeftFoot.position.y, LeftFoot.position.z + 0.5f), LayerMask.GetMask("Ground"));


			/*
			LeftHand.position = Utility.ProjectGround(LeftHand.position, LayerMask.GetMask("Ground"));
			RightHand.position = Utility.ProjectGround(RightHand.position, LayerMask.GetMask("Ground"));
			LeftHand.position = new Vector3(LeftHand.position.x,LeftHand.position.y + 0.08f, LeftHand.position.z);
			RightHand.position = new Vector3(RightHand.position.x, RightHand.position.y + 0.08f, RightHand.position.z);
			*/


			//if ((Hips.position.y - (intensity / 8)) < Utility.ProjectGround(Hips.position, LayerMask.GetMask("Ground")).y + 0.07f) return;
			Transform neck = Array.Find(GetActor().Bones, x => x.Transform.name == "Neck").Transform;
			Transform headSite = Array.Find(GetActor().Bones, x => x.Transform.name == "HeadSite").Transform;
			Transform leftShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftShoulder").Transform;
			Transform rightShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "RightShoulder").Transform;
			
			//head.LookAt(Target.transform, Vector3.forward);

			ModelHead = UltimateIK.UltimateIK.BuildModel(Spine, new Transform[]{ headSite, LeftHand, RightHand } );
			ModelLH = UltimateIK.UltimateIK.BuildModel(leftShoulder, LeftHand);
			ModelRH = UltimateIK.UltimateIK.BuildModel(rightShoulder, RightHand);

			ModelHead.Objectives[0].SetTarget(Target.position);
			//ModelLH.Objectives[0].SetTarget(LeftHand.position);
			//ModelRH.Objectives[0].SetTarget(RightHand.position);



			ModelHead.Solve();
		}

		public void Hydrate()
		{
			{
				Transform leftFootSite = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftFoot").Transform;
				Transform rightFootSite = Array.Find(GetActor().Bones, x => x.Transform.name == "RightFoot").Transform;
				Transform leftHandSite = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftHand").Transform;
				Transform rightHandSite = Array.Find(GetActor().Bones, x => x.Transform.name == "RightHand").Transform;

				Matrix4x4 leftFoot = leftFootSite.GetWorldMatrix(true);
				Matrix4x4 rightFoot = rightFootSite.GetWorldMatrix(true);
				Matrix4x4 leftHand = leftHandSite.GetWorldMatrix(true);
				Matrix4x4 rightHand = rightHandSite.GetWorldMatrix(true);

				Hips.position = new Vector3(Hips.position.x, Hips.position.y - (1f / 24f), Hips.position.z);
				//Hips.RotateAround(Utility.ProjectGround(Hips.position, LayerMask.GetMask("Ground")), Hips.right, -10);
				Hips.rotation *= Quaternion.AngleAxis(-25f, Vector3.right);


				Transform leftUpLeg = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftUpLeg").Transform;
				Transform rightUpLeg = Array.Find(GetActor().Bones, x => x.Transform.name == "RightUpLeg").Transform;
				Transform leftShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftShoulder").Transform;
				Transform rightShoulder = Array.Find(GetActor().Bones, x => x.Transform.name == "RightShoulder").Transform;

				ModelLF = UltimateIK.UltimateIK.BuildModel(leftUpLeg, leftFootSite);

				ModelRF = UltimateIK.UltimateIK.BuildModel(rightUpLeg, rightFootSite);

				ModelLH = UltimateIK.UltimateIK.BuildModel(leftShoulder, leftHandSite);

				ModelRH = UltimateIK.UltimateIK.BuildModel(rightShoulder, rightHandSite);

				ModelLF.Objectives[0].SetTarget(leftFoot);
				ModelRF.Objectives[0].SetTarget(rightFoot);
				ModelLH.Objectives[0].SetTarget(leftHand);
				ModelRH.Objectives[0].SetTarget(rightHand);

				ModelLH.Solve();
				ModelRH.Solve();
				ModelLF.Solve();
				ModelRF.Solve();
			}
			{
				Transform leftFoot = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftFoot").Transform;
				Transform leftFootSite = Array.Find(GetActor().Bones, x => x.Transform.name == "LeftFootSite").Transform;
				/*
				Vector3 legPos = new Vector3(leftFootSite.position.x, 0.6f * Hips.position.y, leftFootSite.position.z);
				legPos += 0.2f * Vector3.ProjectOnPlane(Hips.forward, Vector3.up).normalized;
				legPos += 0.2f * Vector3.ProjectOnPlane(-Hips.right, Vector3.up).normalized;
				Quaternion legRot = leftFoot.rotation * Quaternion.AngleAxis(-25f, new Vector3(1f, 0f, 0f));
				*/
				//ModelLF = UltimateIK.UltimateIK.BuildModel(leftUpLeg, leftFootSite);
				Matrix4x4 target = Matrix4x4.TRS(
					new Vector3(0.04700002f, 0.2870001f, 0.1780002f),
					Quaternion.Euler(-27.172f, 1.867f, -1.703f),
					new Vector3(0.1000003f, 0.1000001f, 0.1000003f)
					).GetRelativeTransformationFrom(GetActor().FindTransform("Skeleton").GetWorldMatrix(false));

				ModelLF.Objectives[0].SetTarget(target.GetPosition());
				ModelLF.Objectives[0].SetTarget(target.GetRotation());

				ModelLF.Solve();
			}

			{
				Transform neck = Array.Find(GetActor().Bones, x => x.Transform.name == "Neck").Transform;
				Transform head = Array.Find(GetActor().Bones, x => x.Transform.name == "Head").Transform;
				Transform tail = Array.Find(GetActor().Bones, x => x.Transform.name == "Tail").Transform;

				neck.rotation = Quaternion.AngleAxis(-60f, Vector3.up) * neck.rotation;
				head.rotation = Quaternion.AngleAxis(-30f, Vector3.up) * head.rotation;
				tail.rotation *= Quaternion.AngleAxis(25f, Vector3.forward);
			}

		}


		private Actor GetActor()
		{
			if (Actor == null)
			{
				Actor = GetComponent<Actor>();
			}
			return Actor;
		}


	}
}