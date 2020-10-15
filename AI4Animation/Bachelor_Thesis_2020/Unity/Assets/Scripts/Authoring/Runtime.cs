using UnityEngine;
using System;
using System.Collections.Generic;
using DeepLearning;
using KDTree;
using UnityEngine.SceneManagement;
using System.Linq;
using System.IO;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace SIGGRAPH_2018
{
	[RequireComponent(typeof(Actor))]
	public class Runtime : MonoBehaviour
	{
		public bool Inspect = false;

		public bool ShowTrajectory = true;
		public bool ShowVelocities = false;
		public bool ShowAuthoring = true;
		public bool Paused = false;
		public bool DrawLatentSpaces = false;

		public bool TrajectoryControl = true;
		public AnimationAuthoring AnimationAuthoring;
		[Range(0, 1)] public float TrajectoryCorrection = 1f;
		[Range(0, 1)] public float PositionControlFactor = 0.95f;
		[Range(0, 1)] public float DirectionControlFactor = 0.4f;
		[Range(0, 1)] public float StyleControlFactor = 0.95f;
		[Range(0, 1)] public float VelocityControlFactor = 0f;
		[Range(0, 1)] public float SpeedControlFactor = 0.5f;


		private Actor Actor;
		private MANN NN;
		private Trajectory Trajectory;

		private Vector3 TargetDirection;
		private Vector3 TargetVelocity;

		//State
		private Vector3[] Positions = new Vector3[0];
		private Vector3[] Forwards = new Vector3[0];
		private Vector3[] Ups = new Vector3[0];
		private Vector3[] Velocities = new Vector3[0];

		//NN Parameters
		private const int TrajectoryDimIn = 16;
		private const int TrajectoryDimOut = 6;
		private const int JointDimIn = 12;
		private const int JointDimOut = 12;

		//Trajectory for 60 Hz framerate
		private const int Framerate = 60;
		private const int Points = 111;
		private const int PointSamples = 12;
		private const int PastPoints = 60;
		private const int FuturePoints = 50;
		private const int RootPointIndex = 60;
		private const int PointDensity = 10;

		//Post-Processing
		private MotionPostprocessing MotionEditing;

		private string[] StyleNames = new string[] { "Idle", "Move", "Jump", "Sit", "Stand", "Lie", "Sneak", "Eat", "Hydrate" };
		//Post-Processing
		//private MotionEditing MotionEditing;
		
		//Performance
		private float NetworkPredictionTime;

		private float MotionTime = 0f;
		private float[] ContactLabels = new float[4];


		void Awake()
		{

			Actor = GetComponent<Actor>();
			NN = GetComponent<MANN>();
			MotionEditing = GetComponent<MotionPostprocessing>();

			if (AnimationAuthoring == null)
			{
				Debug.Log("No AnimationAuthoring linked!");
				return;
			}
			if (AnimationAuthoring.ControlPoints.Count <= 0)
			{
				Debug.Log("No Controlpoints defined.");
				return;
			}

			AnimationAuthoring.GenerateLookUpPoints(AnimationAuthoring.TimeDelta);

			AnimationAuthoring.CreateCP = false;

			Point p0 = AnimationAuthoring.GetLookUpPoint(0);
			Point p1 = AnimationAuthoring.GetLookUpPoint(AnimationAuthoring.TimeDelta);
			Vector3 direction = (p1.GetPosition() - p0.GetPosition()).normalized;
			TargetDirection = direction;
			TargetVelocity = p0.GetVelocity();
			Positions = new Vector3[Actor.Bones.Length];
			Forwards = new Vector3[Actor.Bones.Length];
			Ups = new Vector3[Actor.Bones.Length];
			Velocities = new Vector3[Actor.Bones.Length];
			StyleNames = new string[] { "Idle", "Move", "Jump", "Sit", "Stand", "Lie", "Sneak", "Eat", "Hydrate" };
			Trajectory = new Trajectory(Points, StyleNames, AnimationAuthoring.ControlPoints[0].Transform.position, TargetDirection, AnimationAuthoring);



			Actor.transform.position = AnimationAuthoring.ControlPoints[0].Transform.position;
			Actor.transform.LookAt(AnimationAuthoring.ControlPoints[0].Transform);
			//Compute Posture
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Positions[i] = Actor.Bones[i].Transform.position;
				Forwards[i] = Actor.Bones[i].Transform.forward;
				Ups[i] = Actor.Bones[i].Transform.up;
				Velocities[i] = Vector3.zero;
			}

			if (StyleNames.Length > 0)
			{
				for (int i = 0; i < Trajectory.Points.Length; i++)
				{
					//IDLE 
					Trajectory.Points[i].Styles[0] = 1f;
				}

				for (int i = 0; i < Trajectory.Points.Length; i++)
				{
					float timestamp = 0 + ((i - RootPointIndex) / 60f);

					//Set Direction to next point
					Point p = AnimationAuthoring.GetLookUpPoint(timestamp);
					Point p2 = AnimationAuthoring.GetLookUpPoint(timestamp + AnimationAuthoring.TimeDelta);
					Vector3 dir = (p2.GetPosition() - p.GetPosition()).normalized;
					p.SetDirection(dir);

					Trajectory.Points[i].SetDirection(p.GetDirection().normalized);
					Trajectory.Points[i].SetVelocity(p.GetVelocity());
					Trajectory.Points[i].SetPosition(p.GetPosition());
					Trajectory.Points[i].SetSpeed(p.GetSpeed());
					Trajectory.Points[i].TerrainHeight = p.GetPosition().y;
				}
			}

			if (NN.Parameters == null)
			{
				Debug.Log("No parameters saved.");
				return;
			}

			//Trigger callback once to init transforms
			for (int i = 0; i < AnimationAuthoring.ControlPoints.Count; i++)
			{
				if (AnimationAuthoring.ControlPoints[i].GetTransform().hasChanged)
				{
					AnimationAuthoring.ControlPoints[i].GetTransform().hasChanged = false;
				}
			}

			NN.LoadParameters();
		}

		void Start()
		{
			//Utility.SetFPS(60);
			Application.targetFrameRate = 60;
		}

		public Trajectory GetTrajectory()
		{
			return Trajectory;
		}

		void Update()
		{
			if (Paused && Time.time > 0.7f)
			{
				return;
			}

			if (NN.Parameters == null)
			{
				return;
			}

			if (AnimationAuthoring == null)
			{
				return;
			}

			if (AnimationAuthoring.ControlPoints.Count <= 0)
			{
				return;
			}


			if (TrajectoryControl)
			{
				if (MotionTime > 0) MotionTime -= Time.deltaTime;
				if (Input.GetMouseButtonUp(0) && ShowAuthoring) CheckChanges();
				PredictTrajectory();
			}

			if (NN.Parameters != null)
			{
				Animate();
			}


			
			if (MotionEditing != null)
			{
				
				MotionEditing.Process(ContactLabels);

			}
			
		}
		
		private float GetDistanceToPath(Vector3 pos)
		{
			return Vector3.Distance(pos,AnimationAuthoring.GetLookUpPoint(AnimationAuthoring.RefTimestamp - AnimationAuthoring.TimeDelta).GetPosition());
		}

		private float SampleAngle(int currentFrame, float sec)
		{
			float sum = 0f;
			int frames = (int)(sec * 60);
			for(int i= currentFrame; i <= currentFrame + frames; i++)
			{
				Vector3 v1 = AnimationAuthoring.LookUpPoints[i].GetPosition() - AnimationAuthoring.LookUpPoints[i-1].GetPosition();
				Vector3 v2 = AnimationAuthoring.LookUpPoints[i+1].GetPosition() - AnimationAuthoring.LookUpPoints[i].GetPosition();
				float angle = Vector3.Angle(v1, v2);

				sum += angle;
			}

			return sum;
		}

		// returns angleturningSpeed in degrees per second
		private float SampleAngle(int currentFrame)
		{
			int i = currentFrame;

			Vector3 v1 = AnimationAuthoring.LookUpPoints[i].GetPosition() - AnimationAuthoring.LookUpPoints[i - 1].GetPosition();
			Vector3 v2 = AnimationAuthoring.LookUpPoints[i + 1].GetPosition() - AnimationAuthoring.LookUpPoints[i].GetPosition();
			float angle = Vector3.Angle(v1, v2) * Framerate;


			return angle;
		}

		private void PredictTrajectory()
		{
			//Predict Future Trajectory
			//aktueller referenzpunkt zum rootpoint
			float tRoot;
			if (Paused)
			{
				tRoot = AnimationAuthoring.RefTimestamp;
			}
			else
			{
				tRoot = AnimationAuthoring.GetClosestPointTimestamp(Trajectory.Points[RootPointIndex].GetTransformation().GetPosition(), AnimationAuthoring.RefTimestamp);
			}
			AnimationAuthoring.RefTimestamp = tRoot;

			float update = Mathf.Min(
				Mathf.Pow(1f - (Trajectory.Points[RootPointIndex].Styles[0]), 0.25f),
				Mathf.Pow(1f - (Trajectory.Points[RootPointIndex].Styles[1] / 2), 1f),
				Mathf.Pow(1f - (Trajectory.Points[RootPointIndex].Styles[3]
								+ Trajectory.Points[RootPointIndex].Styles[4]
								+ Trajectory.Points[RootPointIndex].Styles[5]
								+ Trajectory.Points[RootPointIndex].Styles[7]
								+ Trajectory.Points[RootPointIndex].Styles[8]
							), 0.5f)
			);

			//Debug.Log(tRoot);
			int loopIndex = 0;
			//set future point values
			for (int i = RootPointIndex+1; i < Trajectory.Points.Length; i++)
			{

				float timestamp = tRoot + (float)(loopIndex*AnimationAuthoring.TimeDelta) + AnimationAuthoring.TimeDelta;

				//Set Direction to next point
				Point p = AnimationAuthoring.GetLookUpPoint(timestamp);
				Point p2 = AnimationAuthoring.GetLookUpPoint(timestamp + AnimationAuthoring.TimeDelta);
				Vector3 dir = (p2.GetPosition() - p.GetPosition()).normalized;
				p.SetDirection(dir);


				DirectionControlFactor = update;
				VelocityControlFactor = update;

				Vector3 direction_final = Vector3.Slerp(Trajectory.Points[i].GetDirection(), p.GetDirection().normalized, DirectionControlFactor);
				Trajectory.Points[i].SetDirection(direction_final);
				TargetDirection = direction_final;

				Trajectory.Points[i].SetPosition(Vector3.Lerp(Trajectory.Points[i].GetPosition(), p.GetPosition(), PositionControlFactor));

				Vector3 velNormal = Vector3.Lerp(Trajectory.Points[i].GetVelocity(), p.GetVelocity(), VelocityControlFactor);
				Trajectory.Points[i].SetVelocity(velNormal);
				TargetVelocity = velNormal;

				loopIndex += 1;
			}

			loopIndex = 0;
			for (int i = RootPointIndex ; i < Trajectory.Points.Length; i++)
			{
				float timestamp = tRoot + (float)(loopIndex * AnimationAuthoring.TimeDelta);

				Point p = AnimationAuthoring.GetLookUpPoint(timestamp);

				for (int j = 0; j < Trajectory.Points[i].Styles.Length; j++)
				{
					Trajectory.Points[i].Styles[j] = Utility.Interpolate(Trajectory.Points[i].Styles[j], p.GetStyle()[j], 0.95f);
				}

				Utility.Normalise(ref Trajectory.Points[i].Styles);

				SpeedControlFactor = update;
				Trajectory.Points[i].SetSpeed(Utility.Interpolate(Trajectory.Points[i].GetSpeed(), p.GetSpeed(), SpeedControlFactor));

				//ControlPoint cp = AnimationAuthoring.GetControlPoint(timestamp, 1);
				Trajectory.Points[i].TerrainHeight = p.GetPosition().y;

				loopIndex += 1;
			}
		}


#if UNITY_EDITOR
		void OnEnable()
		{
			SceneView.onSceneGUIDelegate += SceneGUI;
		}

		void SceneGUI(SceneView sceneView)
		{
			Event e = Event.current;

			if (Application.isPlaying && Application.isEditor)
			{
				if (e.type == EventType.MouseUp && e.button == 0){
					CheckChanges();
				}				
			}
		}
#endif
		private void CheckChanges()
		{
			for (int i = 0; i < AnimationAuthoring.ControlPoints.Count; i++)
			{
				if (AnimationAuthoring.ControlPoints[i].GetTransform().hasChanged)
				{
					AnimationAuthoring.UpdateLookUpPoint(i * AnimationAuthoring.TimeInterval);
					AnimationAuthoring.ControlPoints[i].GetTransform().hasChanged = false;
					return;
				}
			}
		}
		private float FootSliding(Actor.Bone foot, Vector3 previousFootPosition)
		{
			//Array.Find(Actor.Bones, x => x.Transform.name == "LeftHandSite").Velocity.x
			Vector3 velocity = (foot.Transform.position - previousFootPosition) / Time.deltaTime;
			
			float magnitude = new Vector3(velocity.x, 0f, velocity.z).magnitude;
			float footHeight = foot.Transform.position.y - Utility.GetHeight(foot.Transform.position, AnimationAuthoring.GetControlPoint(AnimationAuthoring.RefTimestamp, 0).Ground);
			float threshold = 0.025f;
			float s = magnitude*(2f - Mathf.Pow(2, Mathf.Clamp(footHeight / threshold, 0f,1f)));

			return s;
		}

		private void Animate()
		{
			//Calculate Root
			Matrix4x4 currentRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			currentRoot[1, 3] = 0f; //For flat terrain

			int start = 0;
			//Input Trajectory Positions / Directions / Velocities / Styles
			for (int i = 0; i < PointSamples; i++)
			{
				Vector3 pos = GetSample(i).GetPosition().GetRelativePositionTo(currentRoot);
				Vector3 dir = GetSample(i).GetDirection().GetRelativeDirectionTo(currentRoot);
				Vector3 vel = GetSample(i).GetVelocity().GetRelativeDirectionTo(currentRoot);
				float speed = GetSample(i).GetSpeed();

				NN.SetInput(start + i * TrajectoryDimIn + 0, pos.x);
				NN.SetInput(start + i * TrajectoryDimIn + 1, pos.z);
				NN.SetInput(start + i * TrajectoryDimIn + 2, dir.x);
				NN.SetInput(start + i * TrajectoryDimIn + 3, dir.z);
				NN.SetInput(start + i * TrajectoryDimIn + 4, vel.x);
				NN.SetInput(start + i * TrajectoryDimIn + 5, vel.z);
				NN.SetInput(start + i * TrajectoryDimIn + 6, speed);

				for (int j = 0; j < StyleNames.Length; j++)
				{
					NN.SetInput(start + i * TrajectoryDimIn + (TrajectoryDimIn - StyleNames.Length) + j, GetSample(i).Styles[j]);
				}
			}
			start += TrajectoryDimIn * PointSamples;

			Matrix4x4 previousRoot = Trajectory.Points[RootPointIndex - 1].GetTransformation();
			previousRoot[1, 3] = 0f; //For flat terrain

			//Input Previous Bone Positions / Velocities
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Vector3 pos = Positions[i].GetRelativePositionTo(previousRoot);
				Vector3 forward = Forwards[i].GetRelativeDirectionTo(previousRoot);
				Vector3 up = Ups[i].GetRelativeDirectionTo(previousRoot);
				Vector3 vel = Velocities[i].GetRelativeDirectionTo(previousRoot);
				NN.SetInput(start + i * JointDimIn + 0, pos.x);
				NN.SetInput(start + i * JointDimIn + 1, pos.y);
				NN.SetInput(start + i * JointDimIn + 2, pos.z);
				NN.SetInput(start + i * JointDimIn + 3, forward.x);
				NN.SetInput(start + i * JointDimIn + 4, forward.y);
				NN.SetInput(start + i * JointDimIn + 5, forward.z);
				NN.SetInput(start + i * JointDimIn + 6, up.x);
				NN.SetInput(start + i * JointDimIn + 7, up.y);
				NN.SetInput(start + i * JointDimIn + 8, up.z);
				NN.SetInput(start + i * JointDimIn + 9, vel.x);
				NN.SetInput(start + i * JointDimIn + 10, vel.y);
				NN.SetInput(start + i * JointDimIn + 11, vel.z);
			}
			start += JointDimIn * Actor.Bones.Length;

			//Predict
			System.DateTime timestamp = Utility.GetTimestamp();
			NN.Predict();
			NetworkPredictionTime = (float)Utility.GetElapsedTime(timestamp);

			//Update Past Trajectory
			for (int i = 0; i < RootPointIndex; i++)
			{
				Trajectory.Points[i].SetPosition(Trajectory.Points[i + 1].GetPosition());
				Trajectory.Points[i].SetDirection(Trajectory.Points[i + 1].GetDirection());
				Trajectory.Points[i].SetVelocity(Trajectory.Points[i + 1].GetVelocity());
				Trajectory.Points[i].SetSpeed(Trajectory.Points[i + 1].GetSpeed());
				Trajectory.Points[i].TerrainHeight = Trajectory.Points[i + 1].TerrainHeight;
				for (int j = 0; j < Trajectory.Points[i].Styles.Length; j++)
				{
					Trajectory.Points[i].Styles[j] = Trajectory.Points[i + 1].Styles[j];
				}
			}

			//Update Root
			float update = Mathf.Min(
				Mathf.Pow(1f - (Trajectory.Points[RootPointIndex].Styles[0]), 0.25f),
				Mathf.Pow(1f - (Trajectory.Points[RootPointIndex].Styles[3]
								+ Trajectory.Points[RootPointIndex].Styles[4]
								+ Trajectory.Points[RootPointIndex].Styles[5]
								+ Trajectory.Points[RootPointIndex].Styles[7]
								+ Trajectory.Points[RootPointIndex].Styles[8]
							), 0.5f)
			);

			Vector3 rootMotion = update * new Vector3(NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 0), NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 1), NN.GetOutput(TrajectoryDimOut * 6 + JointDimOut * Actor.Bones.Length + 2));

			rootMotion /= Framerate;
			Vector3 translation = new Vector3(rootMotion.x, 0f, rootMotion.z);
			float angle = rootMotion.y;
			Trajectory.Points[RootPointIndex].SetPosition(translation.GetRelativePositionFrom(currentRoot));
			Trajectory.Points[RootPointIndex].SetDirection(Quaternion.AngleAxis(angle, Vector3.up) * Trajectory.Points[RootPointIndex].GetDirection());
			Trajectory.Points[RootPointIndex].SetVelocity(translation.GetRelativeDirectionFrom(currentRoot) * Framerate);
			Matrix4x4 nextRoot = Trajectory.Points[RootPointIndex].GetTransformation();
			nextRoot[1, 3] = 0f; //For flat terrain

			//Update Future Trajectory
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				Trajectory.Points[i].SetPosition(Trajectory.Points[i].GetPosition() + translation.GetRelativeDirectionFrom(nextRoot));
				Trajectory.Points[i].SetDirection(Quaternion.AngleAxis(angle, Vector3.up) * Trajectory.Points[i].GetDirection());
				Trajectory.Points[i].SetVelocity(Trajectory.Points[i].GetVelocity() + translation.GetRelativeDirectionFrom(nextRoot) * Framerate);
			}
			start = 0;
			for (int i = RootPointIndex + 1; i < Trajectory.Points.Length; i++)
			{
				//ROOT	1		2		3		4		5
				//.x....x.......x.......x.......x.......x
				int index = i;
				int prevSampleIndex = GetPreviousSample(index).GetIndex() / PointDensity;
				int nextSampleIndex = GetNextSample(index).GetIndex() / PointDensity;
				float factor = (float)(i % PointDensity) / PointDensity;

				Vector3 prevPos = new Vector3(
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 0),
					0f,
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 1)
				).GetRelativePositionFrom(nextRoot);
				Vector3 prevDir = new Vector3(
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 2),
					0f,
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 3)
				).normalized.GetRelativeDirectionFrom(nextRoot);
				Vector3 prevVel = new Vector3(
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 4),
					0f,
					NN.GetOutput(start + (prevSampleIndex - 6) * TrajectoryDimOut + 5)
				).GetRelativeDirectionFrom(nextRoot);

				Vector3 nextPos = new Vector3(
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 0),
					0f,
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 1)
				).GetRelativePositionFrom(nextRoot);
				Vector3 nextDir = new Vector3(
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 2),
					0f,
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 3)
				).normalized.GetRelativeDirectionFrom(nextRoot);
				Vector3 nextVel = new Vector3(
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 4),
					0f,
					NN.GetOutput(start + (nextSampleIndex - 6) * TrajectoryDimOut + 5)
				).GetRelativeDirectionFrom(nextRoot);

				Vector3 pos = (1f - factor) * prevPos + factor * nextPos;
				Vector3 dir = ((1f - factor) * prevDir + factor * nextDir).normalized;
				Vector3 vel = (1f - factor) * prevVel + factor * nextVel;

				pos = Vector3.Lerp(Trajectory.Points[i].GetPosition() + vel / Framerate, pos, 0.5f);

				Trajectory.Points[i].SetPosition(
					Utility.Interpolate(
						Trajectory.Points[i].GetPosition(),
						pos,
						TrajectoryCorrection
						)
					);
				Trajectory.Points[i].SetDirection(
					Utility.Interpolate(
						Trajectory.Points[i].GetDirection(),
						dir,
						TrajectoryCorrection
						)
					);
				Trajectory.Points[i].SetVelocity(
					Utility.Interpolate(
						Trajectory.Points[i].GetVelocity(),
						vel,
						TrajectoryCorrection
						)
					);
			}
			start += TrajectoryDimOut * 6;

			//Compute Posture
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Vector3 position = new Vector3(NN.GetOutput(start + i * JointDimOut + 0), NN.GetOutput(start + i * JointDimOut + 1), NN.GetOutput(start + i * JointDimOut + 2)).GetRelativePositionFrom(currentRoot);
				Vector3 forward = new Vector3(NN.GetOutput(start + i * JointDimOut + 3), NN.GetOutput(start + i * JointDimOut + 4), NN.GetOutput(start + i * JointDimOut + 5)).normalized.GetRelativeDirectionFrom(currentRoot);
				Vector3 up = new Vector3(NN.GetOutput(start + i * JointDimOut + 6), NN.GetOutput(start + i * JointDimOut + 7), NN.GetOutput(start + i * JointDimOut + 8)).normalized.GetRelativeDirectionFrom(currentRoot);
				Vector3 velocity = new Vector3(NN.GetOutput(start + i * JointDimOut + 9), NN.GetOutput(start + i * JointDimOut + 10), NN.GetOutput(start + i * JointDimOut + 11)).GetRelativeDirectionFrom(currentRoot);

				Positions[i] = Vector3.Lerp(Positions[i] + velocity / Framerate, position, 0.5f);
				Forwards[i] = forward;
				Ups[i] = up;
				Velocities[i] = velocity;
			}
			start += JointDimOut * Actor.Bones.Length;

			ContactLabels[0] = SmoothStep(NN.GetOutput(363), 2f, .25f);
			ContactLabels[1] = SmoothStep(NN.GetOutput(364), 2f, .25f);
			ContactLabels[2] = SmoothStep(NN.GetOutput(365), 2f, .25f);
			ContactLabels[3] = SmoothStep(NN.GetOutput(366), 2f, .25f);

			//Assign Posture
			transform.position = nextRoot.GetPosition();
			transform.rotation = nextRoot.GetRotation();
			for (int i = 0; i < Actor.Bones.Length; i++)
			{
				Actor.Bones[i].Transform.position = Positions[i];
				Actor.Bones[i].Transform.rotation = Quaternion.LookRotation(Forwards[i], Ups[i]);
			}
		}

		private Trajectory.Point GetSample(int index)
		{
			return Trajectory.Points[Mathf.Clamp(index * 10, 0, Trajectory.Points.Length - 1)];
		}

		private Trajectory.Point GetPreviousSample(int index)
		{
			return GetSample(index / 10);
		}

		private Trajectory.Point GetNextSample(int index)
		{
			if (index % 10 == 0)
			{
				return GetSample(index / 10);
			}
			else
			{
				return GetSample(index / 10 + 1);
			}
		}

		//aktivierungsfunktion sigmoid nach rechts verschoben
		// ultidraw plot function
		public float SmoothStep(float x, float power, float threshold)
		{
			//Validate
			x = Mathf.Clamp(x, 0f, 1f);
			power = Mathf.Max(power, 0f);
			threshold = Mathf.Clamp(threshold, 0f, 1f);

			//Skew X
			if (threshold == 0f || threshold == 1f)
			{
				x = 1f - threshold;
			}
			else
			{
				if (threshold < 0.5f)
				{
					x = 1f - Mathf.Pow(1f - x, 0.5f / threshold);
				}
				if (threshold > 0.5f)
				{
					x = Mathf.Pow(x, 0.5f / (1f - threshold));
				}
			}

			//Evaluate Y
			if (x < 0.5f)
			{
				return 0.5f * Mathf.Pow(2f * x, power);
			}
			if (x > 0.5f)
			{
				return 1f - 0.5f * Mathf.Pow(2f - 2f * x, power);
			}
			return 0.5f;
		}

		void OnGUI()
		{
			if (NN.Parameters == null)
			{
				return;
			}

			if (AnimationAuthoring == null)
			{
				return;
			}

			if (AnimationAuthoring.ControlPoints.Count <= 0)
			{
				return;
			}

			GUIStyle style = new GUIStyle();
			int size = Mathf.RoundToInt(0.01f * Screen.width);
			Rect rect = new Rect(10, Screen.height - 10 - size - size, Screen.width - 2f * 10, size);
			style.alignment = TextAnchor.MiddleRight;
			style.fontSize = size;
			style.normal.textColor = Color.black;
			float sec = AnimationAuthoring.RefTimestamp;
			string text = string.Format("{0:0.0}s", sec);
			GUI.Label(rect, text, style);
		}

		void OnRenderObject()
		{
			if (Application.isPlaying)
			{
				if (NN.Parameters == null)
				{
					return;
				}

				if (AnimationAuthoring == null)
				{
					return;
				}

				if (AnimationAuthoring.ControlPoints.Count <= 0)
				{
					return;
				}

				if (ShowAuthoring)
				{
					UltiDraw.SetDepthRendering(true); 
					AnimationAuthoring.StyleColors = new Color[] { AnimationAuthoring.Idle, AnimationAuthoring.Move, AnimationAuthoring.Jump, AnimationAuthoring.Sit, AnimationAuthoring.Stand, AnimationAuthoring.Lie,  AnimationAuthoring.Sneak, AnimationAuthoring.Eat, AnimationAuthoring.Hydrate };

					if (MotionTime <= 0.0f)
					{
						MotionTime = AnimationAuthoring.GetControlPoint(AnimationAuthoring.RefTimestamp, 0).GetMotionTime();
					}
					if(MotionTime > 0.0f)
					{
						GUIStyle style = new GUIStyle();
						style.normal.textColor = UltiDraw.Red;
						style.fontSize = 20;
						Vector3 pos = AnimationAuthoring.GetLookUpPoint(AnimationAuthoring.RefTimestamp).GetPosition();
#if UNITY_EDITOR
						Handles.Label(new Vector3(pos.x,pos.y+1f,pos.z), string.Format("{0:0.0}s", MotionTime), style);
#endif
					}
					
					int countCpInspector = 0;
					foreach (ControlPoint cp in AnimationAuthoring.ControlPoints)
					{
						if (cp.Inspector) countCpInspector++;
						else continue;
						//AnimationAuthoring.LabelCP(cp, (countCpInspector - 1).ToString());
						UltiDraw.Begin();
						UltiDraw.DrawSphere(AnimationAuthoring.GetGroundPosition(cp.GetPosition(), cp.Ground), Quaternion.identity, 0.1f, UltiDraw.Red);
						UltiDraw.End();
					}

					Color[] colors = AnimationAuthoring.StyleColors;

					for (int i = 0; i < AnimationAuthoring.LookUpPoints.Length; i++)
					{
						//draw every 4th point
						if ((i % (4*AnimationAuthoring.TimeInterval) != 0) || !AnimationAuthoring.GetControlPoint(i * AnimationAuthoring.TimeDelta, +1).Inspector)
						{
							continue;
						}

						float r = 0f;
						float g = 0f;
						float b = 0f;

						for (int j = 0; j < AnimationAuthoring.LookUpPoints[i].GetStyle().Length; j++)
						{
							r += AnimationAuthoring.LookUpPoints[i].GetStyle()[j] * colors[j].r;
							g += AnimationAuthoring.LookUpPoints[i].GetStyle()[j] * colors[j].g;
							b += AnimationAuthoring.LookUpPoints[i].GetStyle()[j] * colors[j].b;
						}
						Color color = new Color(r, g, b, 1f);
						UltiDraw.Begin();
						UltiDraw.DrawCube(AnimationAuthoring.GetPointPositon(i*AnimationAuthoring.TimeDelta), Quaternion.identity, 0.05f, color);
						UltiDraw.End();
					}

				}	
			
				if (ShowTrajectory)
				{
					UltiDraw.SetDepthRendering(false);
					UltiDraw.Begin();
					UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetDirection, 0.05f, 0f, UltiDraw.Red.Transparent(0.75f));
					UltiDraw.DrawLine(Trajectory.Points[RootPointIndex].GetPosition(), Trajectory.Points[RootPointIndex].GetPosition() + TargetVelocity, 0.05f, 0f, UltiDraw.Green.Transparent(0.75f));
					UltiDraw.End();
					Trajectory.Draw(10);
				}

				if (ShowVelocities)
				{
					UltiDraw.SetDepthRendering(false);
					UltiDraw.Begin();
					for (int i = 0; i < Actor.Bones.Length; i++)
					{
						UltiDraw.DrawArrow(
							Actor.Bones[i].Transform.position,
							Actor.Bones[i].Transform.position + Velocities[i],
							0.75f,
							0.0075f,
							0.05f,
							UltiDraw.Purple.Transparent(0.5f)
						);
					}
					UltiDraw.End();
				}

				float[] a = new float[100];
				for (int i = 0; i < a.Length; i++)
				{
					float x = (float)i / (float)(a.Length - 1);
					a[i] = SmoothStep(x, 2f, .8f);


				}

				UltiDraw.Begin();
				//UltiDraw.DrawGUIFunction(new Vector2(0.5f, 0.5f), new Vector2(1f, 1f), a, 0f, 1f, Color.white, Color.black);
				UltiDraw.End();
				
				if (DrawLatentSpaces)
				{
					UltiDraw.SetDepthRendering(false);

					UltiDraw.Begin();
					float YMin = 0f;
					float YMax = 0f;
					Tensor t0 = NN.GetTensor("W0");
					Tensor t1 = NN.GetTensor("W1");
					Tensor t2 = NN.GetTensor("W2");

					float[] arr0 = new float[t0.GetCols()];
					float[] arr1 = new float[t1.GetCols()];
					float[] arr2 = new float[t2.GetCols()];
					for (int i = 0; i < t0.GetCols(); i++)
					{
						float x = (float)i / (float)(arr0.Length - 1);
						//arr[i] = SmoothStep(x, 2f, .8f);
						float colMean = 0f;
						for (int j = 0; j < t0.GetRows(); j++)
						{
							colMean += t0.GetValue(j, i);
						}
						arr0[i] = colMean;
					}
					for (int i = 0; i < t1.GetCols(); i++)
					{
						float x = (float)i / (float)(arr1.Length - 1);
						//arr[i] = SmoothStep(x, 2f, .8f);
						float colMean = 0f;
						for (int j = 0; j < t1.GetRows(); j++)
						{
							colMean += t1.GetValue(j, i);
						}
						arr1[i] = colMean;
					}
					for (int i = 0; i < t2.GetCols(); i++)
					{
						float x = (float)i / (float)(arr2.Length - 1);
						//arr[i] = SmoothStep(x, 2f, .8f);
						float colMean = 0f;
						for (int j = 0; j < t2.GetRows(); j++)
						{
							colMean += t2.GetValue(j, i);
						}
						arr2[i] = colMean;
					}
					YMin = -10f;
					YMax = 10f;

					UltiDraw.DrawGUIFunction(new Vector2(0.5f, 0.9f), new Vector2(0.4f, 0.1f), arr0, YMin, YMax, UltiDraw.LightGrey, UltiDraw.Black, 0.002f, UltiDraw.BlackGrey);
					YMin = -16f;
					YMax = 13f;
					UltiDraw.DrawGUIFunction(new Vector2(0.5f, 0.79f), new Vector2(0.4f, 0.1f), arr1, YMin, YMax, UltiDraw.LightGrey, UltiDraw.Black, 0.002f, UltiDraw.BlackGrey);

					YMin = -2f;
					YMax = 2f;
					UltiDraw.DrawGUIFunction(new Vector2(0.5f, 0.68f), new Vector2(0.4f, 0.1f), arr2, YMin, YMax, UltiDraw.LightGrey, UltiDraw.Black, 0.002f, UltiDraw.BlackGrey);

					UltiDraw.End();
				}
				
				
		

			}
		}

		void OnDrawGizmos()
		{
			if (!Application.isPlaying)
			{
				OnRenderObject();
			}
			//Gizmos.color = Color.blue;
			//Gizmos.DrawSphere(AnimationAuthoring.GetPointPositon(GetClosestPointTimestamp(Trajectory.Points[RootPointIndex].GetTransformation().GetPosition(), RefTimestamp)), 0.1f);
		}
		
		#if UNITY_EDITOR
			[CustomEditor(typeof(Runtime))]
			public class Runtime_Editor : Editor
			{

				public Runtime Target;

				void Awake()
				{
					Target = (Runtime)target;
				}

				public override void OnInspectorGUI()
				{
					Undo.RecordObject(Target, Target.name);

					Inspector();
					//Target.Controller.Inspector();

					if (GUI.changed)
					{
						EditorUtility.SetDirty(Target);
					}
				}

				private void Inspector()
				{
					Utility.SetGUIColor(UltiDraw.Grey);
					using (new EditorGUILayout.VerticalScope("Box"))
					{
						Utility.ResetGUIColor();

						if (Utility.GUIButton("Animation", UltiDraw.DarkGrey, UltiDraw.White))
						{
							Target.Inspect = !Target.Inspect;
						}

					if (Target.Inspect)
						{
						DrawDefaultInspector();
						/*
							using (new EditorGUILayout.VerticalScope("Box"))
							{
								//Target.AnimationAuthoring = (AnimationAuthoring)EditorGUILayout.ObjectField("AnimationAuthoring", Target.AnimationAuthoring, typeof(AnimationAuthoring), true);
								Target.ShowTrajectory = EditorGUILayout.Toggle("Show Trajectory", Target.ShowTrajectory);
								Target.ShowVelocities = EditorGUILayout.Toggle("Show Velocities", Target.ShowVelocities);
								Target.TargetGain = EditorGUILayout.Slider("Target Gain", Target.TargetGain, 0f, 1f);
								Target.TargetDecay = EditorGUILayout.Slider("Target Decay", Target.TargetDecay, 0f, 1f);
								Target.TrajectoryControl = EditorGUILayout.Toggle("Trajectory Control", Target.TrajectoryControl);
								Target.TrajectoryCorrection = EditorGUILayout.Slider("Trajectory Correction", Target.TrajectoryCorrection, 0f, 1f);
							}
							*/
						}
					
					if (Utility.GUIButton("Restart", UltiDraw.DarkGrey, UltiDraw.White))
					{
						SceneManager.LoadScene("Demo");
					}
				}
				}
			}
		#endif
		
	}
}


