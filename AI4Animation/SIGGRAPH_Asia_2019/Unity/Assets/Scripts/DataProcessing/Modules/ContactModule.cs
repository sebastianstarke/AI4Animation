#if UNITY_EDITOR
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class ContactModule : Module {

	public float DrawScale = 1f;
	public bool ShowDebug = true;
	public bool ShowSensors = true;
	public bool ShowTolerances = true;
	public bool ShowDistances = false;
	public bool TrueMotionTrajectory = false;
	public bool CorrectedMotionTrajectory = false;
	public bool ShowContacts = false;
	public bool ContactTrajectories = false;

	//public bool ShowSkeletons = false;
	//public int SkeletonStep = 10;

	public bool EditMotion = true;
	public int Step = 10;
	public float CaptureFilter = 0.1f;
	public float EditFilter = 0.1f;
	public Sensor[] Sensors = new Sensor[0];
	public BakedContacts BakedContacts = null;

	public float PastTrajectoryWindow = 1f;
	public float FutureTrajectoryWindow = 1f;

	private float Window = 1f;

	private UltimateIK.Model IK;

	public override ID GetID() {
		return ID.Contact;
	}

	public override Module Initialise(MotionData data) {
		Data = data;
		return this;
	}

	public override void Slice(Sequence sequence) {

	}

	public override void Callback(MotionEditor editor) {
		if(EditMotion) {
			float delta = 1f/editor.TargetFramerate;
			Actor actor = editor.GetActor();
			IK = UltimateIK.BuildModel(IK, actor.Bones[0].Transform, GetObjectives(actor));
			if(IK.Objectives.Length == 0) {
				return;
			}
			IK.Iterations = 50;
			IK.Activation = UltimateIK.ACTIVATION.Constant;
			IK.RootTranslationY = true;
			IK.SetWeights(GetWeights());
			bool[] solvePositions = GetSolvePositions();
			bool[] solveRotations = GetSolveRotations();
			for(int i=0; i<IK.Objectives.Length; i++) {
				IK.Objectives[i].SolvePosition = solvePositions[i];
				IK.Objectives[i].SolveRotation = solveRotations[i];
			}
			
			Frame frame = editor.GetCurrentFrame();
			Frame relative = (frame.Timestamp - delta) < 0f ? Data.GetFrame(frame.Timestamp + delta) : Data.GetFrame(frame.Timestamp - delta);
			actor.WriteTransforms(relative.GetBoneTransformations(editor.Mirror), Data.Source.GetBoneNames());
			IK.Solve(GetTargets(relative, editor.Mirror));
			Matrix4x4[] relativePosture = actor.GetBoneTransformations();
			actor.WriteTransforms(frame.GetBoneTransformations(editor.Mirror), Data.Source.GetBoneNames());
			IK.Solve(GetTargets(frame, editor.Mirror));
			Matrix4x4[] framePosture = actor.GetBoneTransformations();
			
			for(int i=0; i<actor.Bones.Length; i++) {
				actor.Bones[i].Velocity = (frame.Timestamp - delta) < 0f ? 
				(relativePosture[i].GetPosition() - framePosture[i].GetPosition()) / delta:
				(framePosture[i].GetPosition() - relativePosture[i].GetPosition()) / delta;
			}
		}
	}

	public Sensor AddSensor() {
		return AddSensor(Data.Source.Bones[0].Name);
	}

	public Sensor AddSensor(string bone) {
		return AddSensor(bone, Vector3.zero, 0.1f, 0f, 0f);
	}

	public Sensor AddSensor(string bone, Vector3 offset, float threshold, float tolerance, float velocity) {
		return AddSensor(bone, offset, threshold, tolerance, velocity, Sensor.ID.Closest, Sensor.ID.None);
	}

	public Sensor AddSensor(string bone, Vector3 offset, float threshold, float tolerance, float velocity, Sensor.ID capture, Sensor.ID edit) {
		Sensor sensor = new Sensor(this, Data.Source.FindBone(bone).Index, offset, threshold, tolerance, velocity, capture, edit);
		ArrayExtensions.Add(ref Sensors, sensor);
		return sensor;
	}

	public void RemoveSensor(Sensor sensor) {
		if(!ArrayExtensions.Remove(ref Sensors, sensor)) {
			Debug.Log("Sensor could not be found in " + Data.GetName() + ".");
		}
	}

	public void Clear() {
		ArrayExtensions.Clear(ref Sensors);
	}

	public string[] GetNames() {
		string[] names = new string[Sensors.Length];
		for(int i=0; i<Sensors.Length; i++) {
			names[i] = Sensors[i].GetName();
		}
		return names;
	}

	public Sensor GetSensor(string bone) {
		return System.Array.Find(Sensors, x => x.GetName() == bone);
	}

	public float[] GetContacts(Frame frame, bool mirrored) {
		float[] contacts = new float[Sensors.Length];
		for(int i=0; i<Sensors.Length; i++) {
			contacts[i] = Sensors[i].GetContact(frame, mirrored);
		}
		return contacts;
	}

	public float[] GetContacts(Frame frame, bool mirrored, params string[] bones) {
		float[] contacts = new float[bones.Length];
		for(int i=0; i<bones.Length; i++) {
			Sensor sensor = GetSensor(bones[i]);
			if(sensor == null) {
				Debug.Log("Sensor for bone " + bones[i] + " could not be found.");
				contacts[i] = 0f;
			} else {
				contacts[i] = sensor.GetContact(frame, mirrored);
			}
		}
		return contacts;
	}

	public Matrix4x4[] GetTargets(Frame frame, bool mirrored) {
		List<Matrix4x4> targets = new List<Matrix4x4>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				targets.Add(Sensors[i].GetCorrectedTransformation(frame, mirrored));
			}
		}
		return targets.ToArray();
	}

	public Transform[] GetObjectives(Actor actor) {
		List<Transform> objectives = new List<Transform>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				objectives.Add(actor.FindTransform(Data.Source.Bones[Sensors[i].Bone].Name));
			}
		}
		return objectives.ToArray();
	}

	public float[] GetWeights() {
		List<float> weights = new List<float>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				weights.Add(Sensors[i].Weight);
			}
		}
		return weights.ToArray();
	}

	public bool[] GetSolvePositions() {
		List<bool> values = new List<bool>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				values.Add(Sensors[i].SolvePosition);
			}
		}
		return values.ToArray();
	}

	public bool[] GetSolveRotations() {
		List<bool> values = new List<bool>();
		for(int i=0; i<Sensors.Length; i++) {
			if(Sensors[i].Edit != Sensor.ID.None) {
				values.Add(Sensors[i].SolveRotation);
			}
		}
		return values.ToArray();
	}

	public IEnumerator CaptureContacts(MotionEditor editor) {
		bool edit = EditMotion;
		EditMotion = false;
		Frame current = editor.GetCurrentFrame();
		for(int s=0; s<Sensors.Length; s++) {
			Sensors[s].RegularContacts = new float[Data.Frames.Length];
			Sensors[s].InverseContacts = new float[Data.Frames.Length];
			Sensors[s].RegularContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].InverseContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].RegularDistances = new Vector3[Data.Frames.Length];
			Sensors[s].InverseDistances = new Vector3[Data.Frames.Length];
		}
		System.DateTime time = Utility.GetTimestamp();
		//int count = 0;
		for(int i=0; i<Data.Frames.Length; i++) {
			//count += 1;
			Frame frame = Data.Frames[i];
			editor.LoadFrame(frame);
			for(int s=0; s<Sensors.Length; s++) {
				Sensors[s].CaptureContact(frame, editor);
			}
			//if(count > Step) {
			if(Utility.GetElapsedTime(time) > 0.2f) {
				time = Utility.GetTimestamp();
				//count = 0;
				yield return new WaitForSeconds(0f);
			}
			//}
		}
		editor.LoadFrame(current);
		EditMotion = edit;
	}

	public void CaptureContactsNoCoroutine(MotionEditor editor) {
		bool edit = EditMotion;
		EditMotion = false;
		Frame current = editor.GetCurrentFrame();
		for(int s=0; s<Sensors.Length; s++) {
			Sensors[s].RegularContacts = new float[Data.Frames.Length];
			Sensors[s].InverseContacts = new float[Data.Frames.Length];
			Sensors[s].RegularContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].InverseContactPoints = new Vector3[Data.Frames.Length];
			Sensors[s].RegularDistances = new Vector3[Data.Frames.Length];
			Sensors[s].InverseDistances = new Vector3[Data.Frames.Length];
		}
		for(int i=0; i<Data.Frames.Length; i++) {
			Frame frame = Data.Frames[i];
			editor.LoadFrame(frame);
			for(int s=0; s<Sensors.Length; s++) {
				Sensors[s].CaptureContact(frame, editor);
			}
		}
		editor.LoadFrame(current);
		EditMotion = edit;
	}

	public void BakeContacts(MotionEditor editor) {
		if(BakedContacts == null) {
			return;
		}
		BakedContacts.Setup(GetNames(), Data.GetTotalFrames());
		for(int i=0; i<Data.Frames.Length; i++) {
			for(int s=0; s<Sensors.Length; s++) {
				if(Sensors[s].GetContact(Data.Frames[i], false) == 1f) {
					BakedContacts.BakeContact(s, Sensors[s].GetContactPoint(Data.Frames[i], false), Data.Frames[i], false);
				}
				if(Sensors[s].GetContact(Data.Frames[i], true) == 1f) {
					BakedContacts.BakeContact(s, Sensors[s].GetContactPoint(Data.Frames[i], true), Data.Frames[i], true);
				}
			}
		}
	}

	protected override void DerivedDraw(MotionEditor editor) {
		UltiDraw.Begin();
		
		Frame frame = editor.GetCurrentFrame();

		Color[] colors = UltiDraw.GetRainbowColors(Sensors.Length);

		if(ShowDebug) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].GetContact(frame, editor.Mirror) == 1f) {
					Vector3 contact = Sensors[i].GetContactPoint(frame, editor.Mirror);
					Vector3 corrected = Sensors[i].GetCorrectedContactPoint(frame, editor.Mirror);
					UltiDraw.DrawArrow(contact, corrected, 0.8f, 0.01f, DrawScale*0.025f, colors[i].Transparent(0.5f));
					UltiDraw.DrawSphere(contact, Quaternion.identity, DrawScale*0.025f, UltiDraw.Yellow);
					UltiDraw.DrawSphere(corrected, Quaternion.identity, DrawScale*0.05f, UltiDraw.Gold.Transparent(0.5f));
				}
			}
			for(int i=0; i<Sensors.Length; i++) {
				Matrix4x4 bone = frame.GetBoneTransformation(Sensors[i].Bone, editor.Mirror);
				Matrix4x4 corrected = Sensors[i].GetCorrectedTransformation(frame, editor.Mirror);
				UltiDraw.DrawCube(bone, DrawScale*0.025f, UltiDraw.DarkRed.Transparent(0.5f));
				UltiDraw.DrawLine(bone.GetPosition(), corrected.GetPosition(), colors[i].Transparent(0.5f));
				UltiDraw.DrawCube(corrected, DrawScale*0.025f, UltiDraw.DarkGreen.Transparent(0.5f));
			}
		}

		if(ShowSensors) {
			for(int i=0; i<Sensors.Length; i++) {
				Quaternion rot = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetRotation();
				Vector3 pos = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetPosition() + rot * Sensors[i].Offset;
				UltiDraw.DrawCube(pos, rot, DrawScale*0.025f, UltiDraw.Black);
				UltiDraw.DrawWireSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Transparent(0.25f));
				if(Sensors[i].GetContact(frame, editor.Mirror) == 1f) {
					UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i]);
				} else {
					UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Transparent(0.125f));
				}
			}
		}

		if(ShowTolerances) {
			for(int i=0; i<Sensors.Length; i++) {
				Quaternion rot = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetRotation();
				Vector3 pos = editor.GetActor().GetBoneTransformation(Sensors[i].GetName()).GetPosition() + rot * Sensors[i].Offset;
				UltiDraw.DrawWireSphere(pos, rot, 2f*(Sensors[i].Tolerance+Sensors[i].Threshold), UltiDraw.DarkGrey.Transparent(0.25f));
			}
		}

		if(ShowContacts) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					for(float j=0f; j<=Data.GetTotalTime(); j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						if(Sensors[i].GetContact(reference, editor.Mirror) == 1f) {
							UltiDraw.DrawSphere(Sensors[i].GetContactPoint(reference, editor.Mirror), Quaternion.identity, DrawScale*0.025f, colors[i]);
						}
					}
				}
			}
		}

		/*
		if(ShowSkeletons) {
			UltiDraw.End();
			float start = Mathf.Clamp(frame.Timestamp-Window, 0f, Data.GetTotalTime());
			float end = Mathf.Clamp(frame.Timestamp+Window, 0f, Data.GetTotalTime());
			float inc = Mathf.Max(SkeletonStep, 1)/Data.Framerate;
			for(float j=start; j<=end; j+=inc) {
				Frame reference = Data.GetFrame(j);
				float weight = (j-start+inc) / (end-start+inc);
				editor.GetActor().Sketch(reference.GetBoneTransformations(editor.GetActor().GetBoneNames(), editor.Mirror), Color.Lerp(UltiDraw.Cyan, UltiDraw.Magenta, weight).Transparent(weight));
			}
			UltiDraw.Begin();
		}
		*/

		if(TrueMotionTrajectory || CorrectedMotionTrajectory) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					Vector3 previousPos = Vector3.zero;
					Vector3 previousCorrected = Vector3.zero;
					float start = Mathf.Clamp(frame.Timestamp-PastTrajectoryWindow, 0f, Data.GetTotalTime());
					float end = Mathf.Clamp(frame.Timestamp+FutureTrajectoryWindow, 0f, Data.GetTotalTime());
					for(float j=start; j<=end; j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						Matrix4x4 bone = reference.GetBoneTransformation(Sensors[i].Bone, editor.Mirror);
						Matrix4x4 corrected = Sensors[i].GetCorrectedTransformation(reference, editor.Mirror);
						if(j > start) {
							if(TrueMotionTrajectory) {
								UltiDraw.DrawArrow(previousPos, bone.GetPosition(), 0.8f, DrawScale*0.005f, DrawScale*0.025f, UltiDraw.DarkRed.Lighten(0.5f).Transparent(0.5f));
							}
							if(CorrectedMotionTrajectory) {
								UltiDraw.DrawArrow(previousCorrected, corrected.GetPosition(), 0.8f, DrawScale*0.005f, DrawScale*0.025f, UltiDraw.DarkGreen.Lighten(0.5f).Transparent(0.5f));
							}
							//UltiDraw.DrawLine(previousPos, bone.GetPosition(), UltiDraw.DarkRed.Transparent(0.5f));
							//UltiDraw.DrawLine(previousCorrected, corrected.GetPosition(), UltiDraw.DarkGreen.Transparent(0.5f));
						}
						previousPos = bone.GetPosition();
						previousCorrected = corrected.GetPosition();
						if(TrueMotionTrajectory) {
							UltiDraw.DrawCube(bone, DrawScale*0.025f, UltiDraw.DarkRed.Transparent(0.5f));
						}
						//UltiDraw.DrawLine(bone.GetPosition(), corrected.GetPosition(), colors[i].Transparent(0.5f));
						if(CorrectedMotionTrajectory) {
							UltiDraw.DrawCube(corrected, DrawScale*0.025f, UltiDraw.DarkGreen.Transparent(0.5f));
						}
					}
				}
			}
		}

		if(ContactTrajectories) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					float start = Mathf.Clamp(frame.Timestamp-Window, 0f, Data.GetTotalTime());
					float end = Mathf.Clamp(frame.Timestamp+Window, 0f, Data.GetTotalTime());
					for(float j=0f; j<=Data.GetTotalTime(); j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						if(Sensors[i].GetContact(reference, editor.Mirror) == 1f) {
							Vector3 contact = Sensors[i].GetContactPoint(reference, editor.Mirror);
							Vector3 corrected = Sensors[i].GetCorrectedContactPoint(reference, editor.Mirror);
							UltiDraw.DrawArrow(contact, corrected, 0.8f, Vector3.Distance(contact, corrected)*DrawScale*0.025f, Vector3.Distance(contact, corrected)*DrawScale*0.1f, colors[i].Lighten(0.5f).Transparent(0.5f));
							UltiDraw.DrawSphere(contact, Quaternion.identity, DrawScale*0.0125f, colors[i].Transparent(0.5f));
							UltiDraw.DrawSphere(corrected, Quaternion.identity, DrawScale*0.05f, colors[i]);
						}
					}
				}
			}
		}

		if(ShowDistances) {
			for(int i=0; i<Sensors.Length; i++) {
				if(Sensors[i].Edit != Sensor.ID.None) {
					for(float j=frame.Timestamp-PastTrajectoryWindow; j<=frame.Timestamp+FutureTrajectoryWindow; j+=Mathf.Max(Step, 1)/Data.Framerate) {
						Frame reference = Data.GetFrame(j);
						if(Sensors[i].GetContact(reference, editor.Mirror) == 1f) {
							UltiDraw.DrawArrow(Sensors[i].GetContactPoint(reference, editor.Mirror), Sensors[i].GetContactPoint(reference, editor.Mirror) - Sensors[i].GetContactDistance(reference, editor.Mirror), 0.8f, DrawScale*0.0025f, DrawScale*0.01f, colors[i].Transparent(0.5f));
						}
					}
				}
			}
		}

		UltiDraw.End();
	}

	protected override void DerivedInspector(MotionEditor editor) {
		if(Utility.GUIButton("Capture Contacts", UltiDraw.DarkGrey, UltiDraw.White)) {
			EditorCoroutines.StartCoroutine(CaptureContacts(editor), this);
		}
		EditorGUILayout.BeginHorizontal();
		BakedContacts = (BakedContacts)EditorGUILayout.ObjectField(BakedContacts, typeof(BakedContacts), true);
		EditorGUI.BeginDisabledGroup(BakedContacts == null || editor.Mirror);
		if(Utility.GUIButton("Bake", UltiDraw.DarkGrey, UltiDraw.White)) {
			BakeContacts(editor);
		}
		EditorGUI.EndDisabledGroup();
		EditorGUILayout.EndHorizontal();
		DrawScale = EditorGUILayout.FloatField("Draw Scale", DrawScale);
		EditMotion = EditorGUILayout.Toggle("Edit Motion", EditMotion);
		ShowDebug = EditorGUILayout.Toggle("Show Debug", ShowDebug);
		ShowSensors = EditorGUILayout.Toggle("Show Sensors", ShowSensors);
		ShowTolerances = EditorGUILayout.Toggle("Show Tolerances", ShowTolerances);
		ShowDistances = EditorGUILayout.Toggle("Show Distances", ShowDistances);
		TrueMotionTrajectory = EditorGUILayout.Toggle("True Motion Trajectory", TrueMotionTrajectory);
		CorrectedMotionTrajectory = EditorGUILayout.Toggle("Corrected Motion Trajectory", CorrectedMotionTrajectory);
		PastTrajectoryWindow = EditorGUILayout.FloatField("Past Trajectory Window", PastTrajectoryWindow);
		FutureTrajectoryWindow = EditorGUILayout.FloatField("Future Trajectory Window" , FutureTrajectoryWindow);
		//ShowSkeletons = EditorGUILayout.Toggle("Show Skeletons", ShowSkeletons);
		//SkeletonStep = EditorGUILayout.IntField("Skeleton Step", SkeletonStep);
		ShowContacts = EditorGUILayout.Toggle("Show Contacts", ShowContacts);
		ContactTrajectories = EditorGUILayout.Toggle("Contact Trajectories", ContactTrajectories);
		Step = EditorGUILayout.IntField("Step", Step);
		CaptureFilter = EditorGUILayout.Slider("Capture Filter", CaptureFilter, 0f, 1f);
		EditFilter = EditorGUILayout.Slider("Edit Filter", EditFilter, 0f, 1f);
		for(int i=0; i<Sensors.Length; i++) {
			EditorGUILayout.BeginHorizontal();
			Sensors[i].Inspector(editor);
			EditorGUILayout.BeginVertical();
			if(Utility.GUIButton("-", UltiDraw.DarkRed, UltiDraw.White, 28f, 18f)) {
				RemoveSensor(Sensors[i]);
			}
			EditorGUILayout.EndVertical();
			EditorGUILayout.EndHorizontal();
		}
		if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
			AddSensor();
		}
	}

	[System.Serializable]
	public class Sensor {
		public enum ID {
			None, 
			Closest, 
			RayTopDown, RayCenterDown, RayBottomUp, RayCenterUp, 
			SphereTopDown, SphereCenterDown, SphereBottomUp, SphereCenterUp, 
			RayXPositive, RayXNegative, RayYPositive, RayYNegative, RayZPositive, RayZNegative,
			Identity
			};
		public ContactModule Module = null;
		public int Bone = 0;
		public Vector3 Offset = Vector3.zero;
		public float Threshold = 0.1f;
		public float Tolerance = 0f;
		public float Velocity = 0f;
		public bool SolvePosition = true;
		public bool SolveRotation = true;
		public bool SolveDistance = true;
		public LayerMask Mask = -1;
		public ID Capture = ID.Closest;
		public ID Edit = ID.None;
		public float Weight = 1f;

		public float[] RegularContacts, InverseContacts = new float[0];
		public Vector3[] RegularContactPoints, InverseContactPoints = new Vector3[0];
		public Vector3[] RegularDistances, InverseDistances = new Vector3[0];

		public Sensor(ContactModule module, int bone, Vector3 offset, float threshold, float tolerance, float velocity, ID capture, ID edit) {
			Module = module;
			Bone = bone;
			Offset = offset;
			Threshold = threshold;
			Tolerance = tolerance;
			Velocity = velocity;
			Capture = capture;
			Edit = edit;
			RegularContacts = new float[Module.Data.Frames.Length];
			InverseContacts = new float[Module.Data.Frames.Length];
			RegularContactPoints = new Vector3[Module.Data.Frames.Length];
			InverseContactPoints = new Vector3[Module.Data.Frames.Length];
			RegularDistances = new Vector3[Module.Data.Frames.Length];
			InverseDistances = new Vector3[Module.Data.Frames.Length];
		}

		public string GetName() {
			return Module.Data.Source.Bones[Bone].Name;
		}

		public int GetIndex() {
			return System.Array.FindIndex(Module.Sensors, x => x==this);
		}

		public Vector3 GetPivot(Frame frame, bool mirrored) {
			return Offset.GetRelativePositionFrom(frame.GetBoneTransformation(Bone, mirrored));
		}

		public float GetContact(Frame frame, bool mirrored) {
			return mirrored ? InverseContacts[frame.Index-1] : RegularContacts[frame.Index-1];
		}

		public Vector3 GetContactDistance(Frame frame, bool mirrored) {
			return mirrored ? InverseDistances[frame.Index-1] : RegularDistances[frame.Index-1];
		}

		public Vector3 GetContactPoint(Frame frame, bool mirrored) {
			return mirrored ? InverseContactPoints[frame.Index-1] : RegularContactPoints[frame.Index-1];
		}

		public Vector3 GetCorrectedContactDistance(Frame frame, bool mirrored) {
			Matrix4x4 bone = frame.GetBoneTransformation(Bone, mirrored);
			if(SolveDistance) {
				return GetCorrectedContactPoint(frame, mirrored) - GetContactDistance(frame, mirrored) - bone.GetPosition();
			} else {
				return GetCorrectedContactPoint(frame, mirrored) - bone.GetRotation()*Offset - bone.GetPosition();
			}
		}

		public Vector3 GetCorrectedContactPoint(Frame frame, bool mirrored) {
			Collider collider = null;
			Vector3 point = DetectCollision(frame, mirrored, Edit, GetPivot(frame, mirrored), Tolerance+Threshold, out collider);
			if(collider != null) {
				Interaction annotated = collider.GetComponentInParent<Interaction>();
				if(annotated != null) {
					if(annotated.ContainsContact(GetName())) {
						point = annotated.GetContact(GetName(), frame, mirrored).GetPosition();
					}
					// Transform t = annotated.GetContactTransform(GetName());
					// if(t != null) {
					// 	if(mirrored) {
					// 		point = t.parent.position + t.parent.rotation * Vector3.Scale(t.parent.lossyScale.GetMirror(Module.Data.MirrorAxis), t.localPosition);
					// 	} else {
					// 		point = t.position;
					// 	}
					// }
				}
				BakedContacts baked = collider.GetComponentInParent<BakedContacts>();
				if(baked != null) {
					return baked.GetContactPoint(GetName(), frame, mirrored);
				}
			}
			return point;
		}

		public Matrix4x4 GetCorrectedTransformation(Frame frame, bool mirrored) {
			Matrix4x4 bone = frame.GetBoneTransformation(Bone, mirrored);
			if(Edit == ID.None) {
				return bone;
			}
			if(Edit == ID.Identity) {
				return bone;
			}
			if(GetContact(frame, mirrored) == 1f) {
				//Gaussian smoothing filter along contact points
				int width = Mathf.RoundToInt(Module.EditFilter * Module.Data.Framerate);
				bool[] contacts = new bool[2*width + 1];
				Vector3[] distances = new Vector3[2*width + 1];
				contacts[width] = true;
				distances[width] = GetCorrectedContactDistance(frame, mirrored);
				for(int i=1; i<=width; i++) {
					int left = frame.Index - i;
					int right = frame.Index + i;
					if(left > 1 && right <= Module.Data.GetTotalFrames()) {
						if(GetContact(Module.Data.GetFrame(left), mirrored) == 1f && GetContact(Module.Data.GetFrame(right), mirrored) == 1f) {
							contacts[width-i] = true;
							contacts[width+i] = true;
							distances[width-i] = GetCorrectedContactDistance(Module.Data.GetFrame(left), mirrored);
							distances[width+i] = GetCorrectedContactDistance(Module.Data.GetFrame(right), mirrored);
						} else {
							break;
						}
					} else {
						break;
					}
				}
				return Matrix4x4.TRS(bone.GetPosition() + Utility.FilterGaussian(distances, contacts), bone.GetRotation(), Vector3.one);
			} else {
				//Interpolation between ground truth and contact points
				float min = Mathf.Clamp(frame.Timestamp-Module.Window, 0f, Module.Data.GetTotalTime());
				float max = Mathf.Clamp(frame.Timestamp+Module.Window, 0f, Module.Data.GetTotalTime());
				Frame start = null;
				Frame end = null;
				for(float j=frame.Timestamp; j>=min; j-=1f/Module.Data.Framerate) {
					Frame reference = Module.Data.GetFrame(j);
					if(GetContact(reference, mirrored) == 1f) {
						start = reference;
						break;
					}
				}
				for(float j=frame.Timestamp; j<=max; j+=1f/Module.Data.Framerate) {
					Frame reference = Module.Data.GetFrame(j);
					if(GetContact(reference, mirrored) == 1f) {
						end = reference;
						break;
					}
				}
				if(start != null && end == null) {
					float weight = 1f - (frame.Timestamp - start.Timestamp) / (frame.Timestamp - min);
					return Matrix4x4.TRS(bone.GetPosition() + weight*GetCorrectedContactDistance(start, mirrored), bone.GetRotation(), Vector3.one);
				}
				if(start == null && end != null) {
					float weight = 1f - (end.Timestamp - frame.Timestamp) / (max - frame.Timestamp);
					return Matrix4x4.TRS(bone.GetPosition() + weight*GetCorrectedContactDistance(end, mirrored), bone.GetRotation(), Vector3.one);
				}
				if(start != null && end != null) {
					float weight = (frame.Timestamp - start.Timestamp) / (end.Timestamp - start.Timestamp);
					return Matrix4x4.TRS(
						bone.GetPosition() + Vector3.Lerp(GetCorrectedContactDistance(start, mirrored), GetCorrectedContactDistance(end, mirrored), weight), 
						bone.GetRotation(), 
						Vector3.one
					);
				}
				return bone;
			}
		}
		
		public Vector3 DetectCollision(Frame frame, bool mirrored, Sensor.ID mode, Vector3 pivot, float radius, out Collider collider) {
			if(mode == ID.Closest) {
				return Utility.GetClosestPointOverlapSphere(pivot, radius, Mask, out collider);
			}

			if(mode == ID.RayTopDown) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot + new Vector3(0f, radius, 0f), Vector3.down, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayCenterDown) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot, Vector3.down, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayBottomUp) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - new Vector3(0f, radius, 0f), Vector3.up, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayCenterUp) {
				RaycastHit info;
				bool hit = Physics.Raycast(pivot, Vector3.up, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereTopDown) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot + new Vector3(0f, radius+Threshold, 0f), Threshold, Vector3.down, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereCenterDown) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot + new Vector3(0f, radius, 0f), Threshold, Vector3.down, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereBottomUp) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot - new Vector3(0f, radius+Threshold, 0f), Threshold, Vector3.up, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.SphereCenterUp) {
				RaycastHit info;
				bool hit = Physics.SphereCast(pivot - new Vector3(0f, radius, 0f), Threshold, Vector3.up, out info, radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayXPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetRight();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayXNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetRight();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayYPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetUp();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayYNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetUp();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayZPositive) {
				Vector3 dir = frame.GetBoneTransformation(Bone, mirrored).GetForward();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			if(mode == ID.RayZNegative) {
				Vector3 dir = -frame.GetBoneTransformation(Bone, mirrored).GetForward();
				RaycastHit info;
				bool hit = Physics.Raycast(pivot - radius*dir, dir, out info, 2f*radius, Mask);
				if(hit) {
					collider = info.collider;
					return info.point;
				}
			}

			collider = null;
			return pivot;
		}

		//TODO: FilterGaussian here has problems at the boundary of the file since the pivot point is not centered.
		public void CaptureContact(Frame frame, MotionEditor editor) {
			int width = Mathf.RoundToInt(Module.CaptureFilter * Module.Data.Framerate);
			Frame[] frames = Module.Data.GetFrames(Mathf.Clamp(frame.Index-width, 1, Module.Data.GetTotalFrames()), Mathf.Clamp(frame.Index+width, 1, Module.Data.GetTotalFrames()));
			{
				bool[] contacts = new bool[frames.Length];
				Vector3[] contactPoints = new Vector3[frames.Length];
				Vector3[] distances = new Vector3[frames.Length];
				for(int i=0; i<frames.Length; i++) {
					Frame f = frames[i];
					Vector3 bone = editor.Mirror ? f.GetBoneTransformation(Bone, false).GetPosition().GetMirror(f.Data.MirrorAxis) : f.GetBoneTransformation(Bone, false).GetPosition();
					Vector3 pivot = editor.Mirror ? GetPivot(f, false).GetMirror(f.Data.MirrorAxis) : GetPivot(f, false);
					Collider collider;
					Vector3 collision = DetectCollision(frame, false, Capture, pivot, Threshold, out collider);
					contacts[i] = collider != null;
					if(collider != null) {
						Vector3 distance = collision - bone;
						contactPoints[i] = editor.Mirror ? collision.GetMirror(f.Data.MirrorAxis) : collision;
						distances[i] = editor.Mirror ? distance.GetMirror(f.Data.MirrorAxis) : distance;
					}
				}
				bool hit = Utility.GetMostCommonItem(contacts);
				if(hit) {
					RegularContacts[frame.Index-1] = 1f;
					RegularDistances[frame.Index-1] = Utility.GetMostCenteredVector(distances, contacts);
					RegularContactPoints[frame.Index-1] = Utility.GetMostCenteredVector(contactPoints, contacts);

				} else {
					RegularContacts[frame.Index-1] = 0f;
					RegularDistances[frame.Index-1] = Vector3.zero;
					RegularContactPoints[frame.Index-1] = Vector3.zero;
				}				
			}
			{
				bool[] contacts = new bool[frames.Length];
				Vector3[] distances = new Vector3[frames.Length];
				Vector3[] contactPoints = new Vector3[frames.Length];
				for(int i=0; i<frames.Length; i++) {
					Frame f = frames[i];
					Vector3 bone = editor.Mirror ? f.GetBoneTransformation(Bone, true).GetPosition() : f.GetBoneTransformation(Bone, true).GetPosition().GetMirror(f.Data.MirrorAxis);
					Vector3 pivot = editor.Mirror ? GetPivot(f, true) : GetPivot(f, true).GetMirror(f.Data.MirrorAxis);
					Collider collider;
					Vector3 collision = DetectCollision(frame, true, Capture, pivot, Threshold, out collider);
					contacts[i] = collider != null;
					if(collider != null) {
						Vector3 distance = collision - bone;
						distances[i] = editor.Mirror ? distance : distance.GetMirror(f.Data.MirrorAxis);
						contactPoints[i] = editor.Mirror ? collision : collision.GetMirror(f.Data.MirrorAxis);
					}
				}
				bool hit = Utility.GetMostCommonItem(contacts);
				if(hit) {
					InverseContacts[frame.Index-1] = 1f;
					InverseDistances[frame.Index-1] = Utility.GetMostCenteredVector(distances, contacts);
					InverseContactPoints[frame.Index-1] = Utility.GetMostCenteredVector(contactPoints, contacts);
				} else {
					InverseContacts[frame.Index-1] = 0f;
					InverseDistances[frame.Index-1] = Vector3.zero;
					InverseContactPoints[frame.Index-1] = Vector3.zero;
				}
			}
			if(Velocity > 0f) {
				if(GetContact(frame, false) == 1f) {
					if(frame.GetBoneVelocity(Bone, false, 1f/Module.Data.Framerate).magnitude > Velocity) {
						RegularContacts[frame.Index-1] = 0f;
						RegularContactPoints[frame.Index-1] = GetPivot(frame, false);
						RegularDistances[frame.Index-1] = Vector3.zero;
					}
				}
				if(GetContact(frame, true) == 1f) {
					if(frame.GetBoneVelocity(Bone, true, 1f/Module.Data.Framerate).magnitude > Velocity) {
						InverseContacts[frame.Index-1] = 0f;
						InverseContactPoints[frame.Index-1] = GetPivot(frame, true);
						InverseDistances[frame.Index-1] = Vector3.zero;
					}
				}
			}
		}

		public void Inspector(MotionEditor editor) {
			UltiDraw.Begin();
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Bone", GUILayout.Width(40f));
				Bone = EditorGUILayout.Popup(Bone, editor.GetData().Source.GetBoneNames(), GUILayout.Width(80f));
				EditorGUILayout.LabelField("Mask", GUILayout.Width(30));
				Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers, GUILayout.Width(75f)));
				EditorGUILayout.LabelField("Capture", GUILayout.Width(50));
				Capture = (ID)EditorGUILayout.EnumPopup(Capture, GUILayout.Width(75f));
				EditorGUILayout.LabelField("Edit", GUILayout.Width(30));
				Edit = (ID)EditorGUILayout.EnumPopup(Edit, GUILayout.Width(75f));
				EditorGUILayout.LabelField("Solve Position", GUILayout.Width(80f));
				SolvePosition = EditorGUILayout.Toggle(SolvePosition, GUILayout.Width(20f));
				EditorGUILayout.LabelField("Solve Rotation", GUILayout.Width(80f));
				SolveRotation = EditorGUILayout.Toggle(SolveRotation, GUILayout.Width(20f));
				EditorGUILayout.LabelField("Solve Distance", GUILayout.Width(80f));
				SolveDistance = EditorGUILayout.Toggle(SolveDistance, GUILayout.Width(20f));
				EditorGUILayout.EndHorizontal();

				EditorGUILayout.BeginHorizontal();
				EditorGUILayout.LabelField("Offset", GUILayout.Width(40f));
				Offset = EditorGUILayout.Vector3Field("", Offset, GUILayout.Width(180f));
				EditorGUILayout.LabelField("Threshold", GUILayout.Width(70f));
				Threshold = EditorGUILayout.FloatField(Threshold, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Tolerance", GUILayout.Width(70f));
				Tolerance = EditorGUILayout.FloatField(Tolerance, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Velocity", GUILayout.Width(70f));
				Velocity = EditorGUILayout.FloatField(Velocity, GUILayout.Width(50f));
				EditorGUILayout.LabelField("Weight", GUILayout.Width(60f));
				Weight = EditorGUILayout.FloatField(Weight, GUILayout.Width(50f));
				EditorGUILayout.EndHorizontal();

				Frame frame = editor.GetCurrentFrame();
				MotionData data = editor.GetData();

				EditorGUILayout.BeginVertical(GUILayout.Height(10f));
				Rect ctrl = EditorGUILayout.GetControlRect();
				Rect rect = new Rect(ctrl.x, ctrl.y, ctrl.width, 10f);
				EditorGUI.DrawRect(rect, UltiDraw.Black);

				float startTime = frame.Timestamp-editor.GetWindow()/2f;
				float endTime = frame.Timestamp+editor.GetWindow()/2f;
				if(startTime < 0f) {
					endTime -= startTime;
					startTime = 0f;
				}
				if(endTime > data.GetTotalTime()) {
					startTime -= endTime-data.GetTotalTime();
					endTime = data.GetTotalTime();
				}
				startTime = Mathf.Max(0f, startTime);
				endTime = Mathf.Min(data.GetTotalTime(), endTime);
				int start = data.GetFrame(startTime).Index;
				int end = data.GetFrame(endTime).Index;
				int elements = end-start;

				Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
				Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

				start = Mathf.Clamp(start, 1, Module.Data.Frames.Length);
				end = Mathf.Clamp(end, 1, Module.Data.Frames.Length);

				//Contacts
				for(int i=start; i<=end; i++) {
					if((editor.Mirror ? InverseContacts[i-1] : RegularContacts[i-1]) == 1f) {
						float left = rect.xMin + (float)(i-start)/(float)elements * rect.width;
						float right = left;
						while(i<end && (editor.Mirror ? InverseContacts[i-1] : RegularContacts[i-1]) != 0f) {
							right = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							i++;
						}
						if(left != right) {
							Vector3 a = new Vector3(left, rect.y, 0f);
							Vector3 b = new Vector3(right, rect.y, 0f);
							Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
							Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
							UltiDraw.DrawTriangle(a, c, b, UltiDraw.Green);
							UltiDraw.DrawTriangle(b, c, d, UltiDraw.Green);
						}
					}
				}

				//Current Pivot
				top.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				bottom.x = rect.xMin + (float)(frame.Index-start)/elements * rect.width;
				top.y = rect.yMax - rect.height;
				bottom.y = rect.yMax;
				UltiDraw.DrawLine(top, bottom, UltiDraw.Yellow);

				Handles.DrawLine(Vector3.zero, Vector3.zero); //Somehow needed to get it working...

				EditorGUILayout.EndVertical();
			}
			UltiDraw.End();
		}
	}

}
#endif