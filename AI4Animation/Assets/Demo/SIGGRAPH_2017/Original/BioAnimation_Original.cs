using DeepLearning;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SIGGRAPH_2017 {
	public class BioAnimation_Original : MonoBehaviour {

		public bool Inspect = false;

		public float TargetBlending = 0.25f;
		public float GaitTransition = 0.25f;
		public float TrajectoryCorrection = 1f;

		public Transform Root;
		public Transform[] Joints = new Transform[0];

		public Controller Controller;
		public Character Character;
		public NeuralNetwork NN;

		private Trajectory Trajectory;

		private Vector3 TargetDirection;
		private Vector3 TargetVelocity;
		private Vector3[] Velocities = new Vector3[0];

		//Rescaling for character (cm to m)
		private float UnitScale = 100f;

		//Trajectory for 60 Hz framerate
		private const int PointSamples = 12;
		private const int RootSampleIndex = 6;
		private const int RootPointIndex = 60;
		private const int FuturePoints = 5;
		private const int PreviousPoints = 6;
		private const int PointDensity = 10;

		void Reset() {
			Root = transform;
			Controller = new Controller();
			Character = new Character();
			Character.BuildHierarchy(transform);
			NN = new NeuralNetwork(TYPE.Vanilla);
		}

		void Awake() {
			TargetDirection = new Vector3(Root.forward.x, 0f, Root.forward.z);
			TargetVelocity = Vector3.zero;
			Velocities = new Vector3[Joints.Length];
			Trajectory = new Trajectory(111, Controller.Styles.Length, Root.position, TargetDirection);
			Trajectory.Postprocess();
			NN.Model.LoadParameters();
		}

		void Start() {
			Utility.SetFPS(60);
		}

		public void AutoDetect() {
			SetJointCount(0);
			System.Action<Transform> recursion = null;
			recursion = new System.Action<Transform>((transform) => {
				if(Character.FindSegment(transform.name) != null) {
					AddJoint(transform);
				}
				for(int i=0; i<transform.childCount; i++) {
					recursion(transform.GetChild(i));
				}
			});
			recursion(Root);
		}

		public void AddJoint(Transform joint) {
			System.Array.Resize(ref Joints, Joints.Length+1);
			Joints[Joints.Length-1] = joint;
		}

		void Update() {
			if(NN.Model.Parameters == null) {
				return;
			}
			
			//Update Target Direction / Velocity
			TargetDirection = Vector3.Lerp(TargetDirection, Quaternion.AngleAxis(Controller.QueryTurn()*60f, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection(), TargetBlending);
			TargetVelocity = Vector3.Lerp(TargetVelocity, (Quaternion.LookRotation(TargetDirection, Vector3.up) * Controller.QueryMove()).normalized, TargetBlending);
			
			//Update Gait
			for(int i=0; i<Controller.Styles.Length; i++) {
				Trajectory.Points[RootPointIndex].Styles[i] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[i], Controller.Styles[i].Query() ? 1f : 0f, GaitTransition);
			}
			//For Human Only
			Trajectory.Points[RootPointIndex].Styles[0] = Utility.Interpolate(Trajectory.Points[RootPointIndex].Styles[0], 1.0f - Mathf.Clamp(Vector3.Magnitude(TargetVelocity) / 0.1f, 0.0f, 1.0f), GaitTransition);
			Trajectory.Points[RootPointIndex].Styles[1] = Mathf.Max(Trajectory.Points[RootPointIndex].Styles[1] - Trajectory.Points[RootPointIndex].Styles[2], 0f);
			//

			/*
			//Blend Trajectory Offset
			Vector3 positionOffset = transform.position - Trajectory.Points[RootPointIndex].GetPosition();
			Quaternion rotationOffset = Quaternion.Inverse(Trajectory.Points[RootPointIndex].GetRotation()) * transform.rotation;
			Trajectory.Points[RootPointIndex].SetPosition(Trajectory.Points[RootPointIndex].GetPosition() + positionOffset);
			Trajectory.Points[RootPointIndex].SetDirection(rotationOffset * Trajectory.Points[RootPointIndex].GetDirection());

			for(int i=RootPointIndex; i<Trajectory.Points.Length; i++) {
				float factor = 1f - (i - RootPointIndex)/(RootPointIndex - 1f);
				Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + factor*positionOffset);
			}
			*/

			//Predict Future Trajectory
			Vector3[] trajectory_positions_blend = new Vector3[Trajectory.Points.Length];
			trajectory_positions_blend[RootPointIndex] = Trajectory.Points[RootPointIndex].GetPosition();

			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				float bias_pos = 0.75f;
				float bias_dir = 1.25f;
				float scale_pos = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_pos));
				float scale_dir = (1.0f - Mathf.Pow(1.0f - ((float)(i - RootPointIndex) / (RootPointIndex)), bias_dir));
				float vel_boost = PoolBias();

				float rescale = 1f / (Trajectory.Points.Length - (RootPointIndex + 1f));

				trajectory_positions_blend[i] = trajectory_positions_blend[i-1] + Vector3.Lerp(
					Trajectory.Points[i].GetPosition() - Trajectory.Points[i-1].GetPosition(), 
					vel_boost * rescale * TargetVelocity,
					scale_pos);

				Trajectory.Points[i].SetDirection(Vector3.Lerp(Trajectory.Points[i].GetDirection(), TargetDirection, scale_dir));

				for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
					Trajectory.Points[i].Styles[j] = Trajectory.Points[RootPointIndex].Styles[j];
				}
			}
			
			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				Trajectory.Points[i].SetPosition(trajectory_positions_blend[i]);
			}

			for(int i=RootPointIndex; i<Trajectory.Points.Length; i+=PointDensity) {
				Trajectory.Points[i].Postprocess();
			}

			for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
				//ROOT	1		2		3		4		5
				//.x....x.......x.......x.......x.......x
				Trajectory.Point prev = GetPreviousSample(i);
				Trajectory.Point next = GetNextSample(i);
				float factor = (float)(i % PointDensity) / PointDensity;

				Trajectory.Points[i].SetPosition((1f-factor)*prev.GetPosition() + factor*next.GetPosition());
				Trajectory.Points[i].SetDirection((1f-factor)*prev.GetDirection() + factor*next.GetDirection());
				Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
				Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
				Trajectory.Points[i].SetSlope((1f-factor)*prev.GetSlope() + factor*next.GetSlope());
			}

			//Avoid Collisions
			CollisionChecks(RootPointIndex+1);

			if(NN.Model.Parameters != null) {
				//Calculate Root
				Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
				Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex-1].GetTransformation();
					
				//Input Trajectory Positions / Directions
				for(int i=0; i<PointSamples; i++) {
					Vector3 pos = Trajectory.Points[i*PointDensity].GetPosition().GetRelativePositionTo(currentRoot);
					Vector3 dir = Trajectory.Points[i*PointDensity].GetDirection().GetRelativeDirectionTo(currentRoot);
					NN.Model.SetInput(PointSamples*0 + i, UnitScale * pos.x);
					NN.Model.SetInput(PointSamples*1 + i, UnitScale * pos.z);
					NN.Model.SetInput(PointSamples*2 + i, dir.x);
					NN.Model.SetInput(PointSamples*3 + i, dir.z);
				}

				//Input Trajectory Gaits
				for (int i=0; i<PointSamples; i++) {
					for(int j=0; j<Trajectory.Points[i*PointDensity].Styles.Length; j++) {
						NN.Model.SetInput(PointSamples*(4+j) + i, Trajectory.Points[i*PointDensity].Styles[j]);
					}
					//FOR HUMAN ONLY
					NN.Model.SetInput(PointSamples*8 + i, Trajectory.Points[i*PointDensity].GetSlope());
					//
				}

				//Input Previous Bone Positions / Velocities
				for(int i=0; i<Joints.Length; i++) {
					int o = 10*PointSamples;
					Vector3 pos = Joints[i].position.GetRelativePositionTo(previousRoot);
					Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
					NN.Model.SetInput(o + Joints.Length*3*0 + i*3+0, UnitScale * pos.x);
					NN.Model.SetInput(o + Joints.Length*3*0 + i*3+1, UnitScale * pos.y);
					NN.Model.SetInput(o + Joints.Length*3*0 + i*3+2, UnitScale * pos.z);
					NN.Model.SetInput(o + Joints.Length*3*1 + i*3+0, UnitScale * vel.x);
					NN.Model.SetInput(o + Joints.Length*3*1 + i*3+1, UnitScale * vel.y);
					NN.Model.SetInput(o + Joints.Length*3*1 + i*3+2, UnitScale * vel.z);
				}

				//Input Trajectory Heights
				for(int i=0; i<PointSamples; i++) {
					int o = 10*PointSamples + Joints.Length*3*2;
					NN.Model.SetInput(o + PointSamples*0 + i, UnitScale * (Trajectory.Points[i*PointDensity].GetRightSample().y - currentRoot.GetPosition().y));
					NN.Model.SetInput(o + PointSamples*1 + i, UnitScale * (Trajectory.Points[i*PointDensity].GetPosition().y - currentRoot.GetPosition().y));
					NN.Model.SetInput(o + PointSamples*2 + i, UnitScale * (Trajectory.Points[i*PointDensity].GetLeftSample().y - currentRoot.GetPosition().y));
				}

				//Predict
				float rest = Mathf.Pow(1.0f-Trajectory.Points[RootPointIndex].Styles[0], 0.25f);
				((PFNN)NN.Model).SetDamping(1f - (rest * 0.9f + 0.1f));
				NN.Model.Predict();

				//Update Past Trajectory
				for(int i=0; i<RootPointIndex; i++) {
					Trajectory.Points[i].SetPosition(Trajectory.Points[i+1].GetPosition());
					Trajectory.Points[i].SetDirection(Trajectory.Points[i+1].GetDirection());
					Trajectory.Points[i].SetLeftsample(Trajectory.Points[i+1].GetLeftSample());
					Trajectory.Points[i].SetRightSample(Trajectory.Points[i+1].GetRightSample());
					Trajectory.Points[i].SetSlope(Trajectory.Points[i+1].GetSlope());
					for(int j=0; j<Trajectory.Points[i].Styles.Length; j++) {
						Trajectory.Points[i].Styles[j] = Trajectory.Points[i+1].Styles[j];
					}
				}

				//Update Current Trajectory
				Trajectory.Points[RootPointIndex].SetPosition((rest * new Vector3(NN.Model.GetOutput(0) / UnitScale, 0f, NN.Model.GetOutput(1) / UnitScale)).GetRelativePositionFrom(currentRoot));
				Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(rest * Mathf.Rad2Deg * (-NN.Model.GetOutput(2)), Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
				Trajectory.Points[RootPointIndex].Postprocess();
				Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();

				//Update Future Trajectory
				for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
					Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + (rest * new Vector3(NN.Model.GetOutput(0) / UnitScale, 0f, NN.Model.GetOutput(1) / UnitScale)).GetRelativeDirectionFrom(nextRoot));
				}
				for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
					int w = RootSampleIndex;
					float m = Mathf.Repeat(((float)i - (float)RootPointIndex) / (float)PointDensity, 1.0f);
					float posX = (1-m) * NN.Model.GetOutput(8+(w*0)+(i/PointDensity)-w) + m * NN.Model.GetOutput(8+(w*0)+(i/PointDensity)-w+1);
					float posZ = (1-m) * NN.Model.GetOutput(8+(w*1)+(i/PointDensity)-w) + m * NN.Model.GetOutput(8+(w*1)+(i/PointDensity)-w+1);
					float dirX = (1-m) * NN.Model.GetOutput(8+(w*2)+(i/PointDensity)-w) + m * NN.Model.GetOutput(8+(w*2)+(i/PointDensity)-w+1);
					float dirZ = (1-m) * NN.Model.GetOutput(8+(w*3)+(i/PointDensity)-w) + m * NN.Model.GetOutput(8+(w*3)+(i/PointDensity)-w+1);
					Trajectory.Points[i].SetPosition(
						Utility.Interpolate(
							Trajectory.Points[i].GetPosition(),
							new Vector3(posX / UnitScale, 0f, posZ / UnitScale).GetRelativePositionFrom(nextRoot),
							TrajectoryCorrection
							)
						);
					Trajectory.Points[i].SetDirection(
						Utility.Interpolate(
							Trajectory.Points[i].GetDirection(),
							new Vector3(dirX, 0f, dirZ).normalized.GetRelativeDirectionFrom(nextRoot),
							TrajectoryCorrection
							)
						);
				}

				for(int i=RootPointIndex+PointDensity; i<Trajectory.Points.Length; i+=PointDensity) {
					Trajectory.Points[i].Postprocess();
				}

				for(int i=RootPointIndex+1; i<Trajectory.Points.Length; i++) {
					//ROOT	1		2		3		4		5
					//.x....x.......x.......x.......x.......x
					Trajectory.Point prev = GetPreviousSample(i);
					Trajectory.Point next = GetNextSample(i);
					float factor = (float)(i % PointDensity) / PointDensity;

					Trajectory.Points[i].SetPosition((1f-factor)*prev.GetPosition() + factor*next.GetPosition());
					Trajectory.Points[i].SetDirection((1f-factor)*prev.GetDirection() + factor*next.GetDirection());
					Trajectory.Points[i].SetLeftsample((1f-factor)*prev.GetLeftSample() + factor*next.GetLeftSample());
					Trajectory.Points[i].SetRightSample((1f-factor)*prev.GetRightSample() + factor*next.GetRightSample());
					Trajectory.Points[i].SetSlope((1f-factor)*prev.GetSlope() + factor*next.GetSlope());
				}

				//Avoid Collisions
				CollisionChecks(RootPointIndex);
				
				//Compute Posture
				Vector3[] positions = new Vector3[Joints.Length];
				Quaternion[] rotations = new Quaternion[Joints.Length];
				int opos = 8 + 4*RootSampleIndex + Joints.Length*3*0;
				int ovel = 8 + 4*RootSampleIndex + Joints.Length*3*1;
				//int orot = 8 + 4*RootSampleIndex + Joints.Length*3*2;
				for(int i=0; i<Joints.Length; i++) {			
					Vector3 position = new Vector3(NN.Model.GetOutput(opos+i*3+0), NN.Model.GetOutput(opos+i*3+1), NN.Model.GetOutput(opos+i*3+2)) / UnitScale;
					Vector3 velocity = new Vector3(NN.Model.GetOutput(ovel+i*3+0), NN.Model.GetOutput(ovel+i*3+1), NN.Model.GetOutput(ovel+i*3+2)) / UnitScale;
					//Quaternion rotation = new Quaternion(PFNN.GetOutput(orot+i*3+0), PFNN.GetOutput(orot+i*3+1), PFNN.GetOutput(orot+i*3+2), 0f).Exp();
					positions[i] = Vector3.Lerp(Joints[i].position.GetRelativePositionTo(currentRoot) + velocity, position, 0.5f).GetRelativePositionFrom(currentRoot);
					Velocities[i] = velocity.GetRelativeDirectionFrom(currentRoot);
					//rotations[i] = rotation.GetRelativeRotationFrom(currentRoot);
				}
				
				//Update Posture
				Root.position = nextRoot.GetPosition();
				Root.rotation = nextRoot.GetRotation();
				for(int i=0; i<Joints.Length; i++) {
					Joints[i].position = positions[i];
					Joints[i].rotation = rotations[i];
				}

				//Map to Character
				Character.FetchTransformations(Root);
			}
		}

		private float PoolBias() {
			float[] styles = Trajectory.Points[RootPointIndex].Styles;
			float bias = 0f;
			for(int i=0; i<styles.Length; i++) {
				float _bias = Controller.Styles[i].Bias;
				float max = 0f;
				for(int j=0; j<Controller.Styles[i].Multipliers.Length; j++) {
					if(Input.GetKey(Controller.Styles[i].Multipliers[j].Key)) {
						max = Mathf.Max(max, Controller.Styles[i].Bias * Controller.Styles[i].Multipliers[j].Value);
					}
				}
				for(int j=0; j<Controller.Styles[i].Multipliers.Length; j++) {
					if(Input.GetKey(Controller.Styles[i].Multipliers[j].Key)) {
						_bias = Mathf.Min(max, _bias * Controller.Styles[i].Multipliers[j].Value);
					}
				}
				bias += styles[i] * _bias;
			}
			return bias;
		}

		private Trajectory.Point GetSample(int index) {
			return Trajectory.Points[Mathf.Clamp(index*10, 0, Trajectory.Points.Length-1)];
		}

		private Trajectory.Point GetPreviousSample(int index) {
			return GetSample(index / 10);
		}

		private Trajectory.Point GetNextSample(int index) {
			if(index % 10 == 0) {
				return GetSample(index / 10);
			} else {
				return GetSample(index / 10 + 1);
			}
		}

		private void CollisionChecks(int start) {
			for(int i=start; i<Trajectory.Points.Length; i++) {
				float safety = 0.5f;
				Vector3 previousPos = Trajectory.Points[i-1].GetPosition();
				Vector3 currentPos = Trajectory.Points[i].GetPosition();
				Vector3 testPos = previousPos + safety*(currentPos-previousPos).normalized;
				Vector3 projectedPos = Utility.ProjectCollision(previousPos, testPos, LayerMask.GetMask("Obstacles"));
				if(testPos != projectedPos) {
					Vector3 correctedPos = testPos + safety * (previousPos-testPos).normalized;
					Trajectory.Points[i].SetPosition(correctedPos);
				}
			}
		}

		public void SetJoint(int index, Transform t) {
			if(index < 0 || index >= Joints.Length) {
				return;
			}
			Joints[index] = t;
		}

		public void SetJointCount(int count) {
			count = Mathf.Max(0, count);
			if(Joints.Length != count) {
				System.Array.Resize(ref Joints, count);
			}
		}

		void OnGUI() {
			GUI.color = UltiDraw.Mustard;
			GUI.backgroundColor = UltiDraw.Black;
			float height = 0.05f;
			GUI.Box(Utility.GetGUIRect(0.025f, 0.05f, 0.3f, Controller.Styles.Length*height), "");
			for(int i=0; i<Controller.Styles.Length; i++) {
				GUI.Label(Utility.GetGUIRect(0.05f, 0.075f + i*0.05f, 0.025f, height), Controller.Styles[i].Name);
				string keys = string.Empty;
				for(int j=0; j<Controller.Styles[i].Keys.Length; j++) {
					keys += Controller.Styles[i].Keys[j].ToString() + " ";
				}
				GUI.Label(Utility.GetGUIRect(0.075f, 0.075f + i*0.05f, 0.05f, height), keys);
				GUI.HorizontalSlider(Utility.GetGUIRect(0.125f, 0.075f + i*0.05f, 0.15f, height), Trajectory.Points[RootPointIndex].Styles[i], 0f, 1f);
			}
		}

		void OnRenderObject() {
			if(Root == null) {
				Root = transform;
			}

			UltiDraw.Begin();
			UltiDraw.DrawGUICircle(new Vector2(0.5f, 0.85f), 0.075f, UltiDraw.Black.Transparent(0.5f));
			Quaternion rotation = Quaternion.AngleAxis(-360f * ((PFNN)NN.Model).GetPhase() / (2f * Mathf.PI), Vector3.forward);
			Vector2 a = rotation * new Vector2(-0.005f, 0f);
			Vector2 b = rotation *new Vector3(0.005f, 0f);
			Vector3 c = rotation * new Vector3(0f, 0.075f);
			UltiDraw.DrawGUITriangle(new Vector2(0.5f + b.x/Screen.width*Screen.height, 0.85f + b.y), new Vector2(0.5f + a.x/Screen.width*Screen.height, 0.85f + a.y), new Vector2(0.5f + c.x/Screen.width*Screen.height, 0.85f + c.y), UltiDraw.Cyan);
			UltiDraw.End();

			if(Application.isPlaying) {
				UltiDraw.Begin();
				UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));
				UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Green.Transparent(0.75f));
				UltiDraw.End();
				Trajectory.Draw(10);
			}
			
			if(!Application.isPlaying) {
				Character.FetchTransformations(Root);
			}
			Character.Draw();

			if(Application.isPlaying) {
				UltiDraw.Begin();
				for(int i=0; i<Joints.Length; i++) {
					Character.Segment segment = Character.FindSegment(Joints[i].name);
					if(segment != null) {
						UltiDraw.DrawArrow(
							Joints[i].position,
							Joints[i].position + Velocities[i] * 60f,
							0.75f,
							0.0075f,
							0.05f,
							UltiDraw.Purple.Transparent(0.5f)
						);
					}
				}
				UltiDraw.End();
			}
			
		}

		void OnDrawGizmos() {
			if(!Application.isPlaying) {
				OnRenderObject();
			}
		}
	}

	#if UNITY_EDITOR
	[CustomEditor(typeof(BioAnimation_Original))]
	public class BioAnimation_Original_Editor : Editor {

			public BioAnimation_Original Target;

			void Awake() {
				Target = (BioAnimation_Original)target;
			}

			public override void OnInspectorGUI() {
				Undo.RecordObject(Target, Target.name);

				Inspector();
				Target.Controller.Inspector();
				Target.Character.Inspector(Target.transform);
				Target.NN.Inspector();

				if(GUI.changed) {
					EditorUtility.SetDirty(Target);
				}
			}

			private void Inspector() {
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					if(Target.Character.RebuildRequired(Target.Root)) {
						EditorGUILayout.HelpBox("Rebuild required because hierarchy was changed externally.", MessageType.Error);
						if(Utility.GUIButton("Build Hierarchy", Color.grey, Color.white)) {
							Target.Character.BuildHierarchy(Target.Root);
						}
					}

					if(Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White)) {
						Target.Inspect = !Target.Inspect;
					}
					
					if(Target.Inspect) {
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Target.TargetBlending = EditorGUILayout.Slider("Target Blending", Target.TargetBlending, 0f, 1f);
							Target.GaitTransition = EditorGUILayout.Slider("Gait Transition", Target.GaitTransition, 0f, 1f);
							Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);
							EditorGUI.BeginDisabledGroup(true);
							EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true);
							EditorGUI.EndDisabledGroup();
							Target.SetJointCount(EditorGUILayout.IntField("Joint Count", Target.Joints.Length));
							if(Utility.GUIButton("Auto Detect", UltiDraw.DarkGrey, UltiDraw.White)) {
								Target.AutoDetect();
							}
							for(int i=0; i<Target.Joints.Length; i++) {
								if(Target.Joints[i] != null) {
									Utility.SetGUIColor(UltiDraw.Green);
								} else {
									Utility.SetGUIColor(UltiDraw.Red);
								}
								Target.SetJoint(i, (Transform)EditorGUILayout.ObjectField("Joint " + (i+1), Target.Joints[i], typeof(Transform), true));
								Utility.ResetGUIColor();
							}
						}
					}
					
				}
			}
	}
	#endif
}