using UnityEngine;
using System.Collections.Generic;
using System.Linq;


#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif

namespace AI4Animation {
	public class ContactModule : Module {
		public enum ColliderType {Sphere, Capsule, Box};
		public enum ContactType {Translational, Rotational}
		public bool ShowSensors = true;

		public Sensor[] Sensors = new Sensor[0];

		private LayerMask CopyMask = ~0;

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series instance = new Series(global, GetNames());
			for(int i=0; i<instance.Samples.Length; i++) {
                float t = timestamp + instance.Samples[i].Timestamp;
				instance.Values[i] = GetContacts(t, mirrored);
			}
			return instance;
		}
		#if UNITY_EDITOR
		protected override void DerivedInitialize() {

		}

		protected override void DerivedLoad(MotionEditor editor) {
			
		}

		protected override void DerivedUnload(MotionEditor editor) {

		}
		
		protected override void DerivedCallback(MotionEditor editor) {

		}

		protected override void DerivedGUI(MotionEditor editor) {

		}

		protected override void DerivedDraw(MotionEditor editor) {
			UltiDraw.Begin();
			Color[] colors = UltiDraw.GetRainbowColors(Sensors.Length);
			if(ShowSensors) {
				for(int i=0; i<Sensors.Length; i++) {
					if(Sensors[i].SensorType == ColliderType.Sphere) {
						Quaternion rot = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).rotation;
						Vector3 pos = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).position;
						pos += rot * Sensors[i].Offset;
						rot *= Quaternion.Euler(Sensors[i].Rotation);
						UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
						UltiDraw.DrawWireSphere(pos, rot, 2f*Sensors[i].Thresholds.x, colors[i].Opacity(0.25f));
						if(Sensors[i].GetContact(editor.GetCurrentFrame(), editor.Mirror) == 1f) {
							UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Thresholds.x, colors[i]);
						} else {
							UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Thresholds.x, colors[i].Opacity(0.125f));
						}
					}
					if(Sensors[i].SensorType == ColliderType.Capsule) {
						Quaternion rot = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).rotation;
						Vector3 pos = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).position;
						pos += rot * Sensors[i].Offset;
						rot *= Quaternion.Euler(Sensors[i].Rotation);
						UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
						UltiDraw.DrawCapsule(pos, rot, 2f*Sensors[i].Thresholds.x, 2f*Sensors[i].Thresholds.y, colors[i].Opacity(0.25f));
						if(Sensors[i].GetContact(editor.GetCurrentFrame(), editor.Mirror) == 1f) {
							UltiDraw.DrawCapsule(pos, rot, 2f*Sensors[i].Thresholds.x, 2f*Sensors[i].Thresholds.y, colors[i]);
						} else {
							UltiDraw.DrawCapsule(pos, rot, 2f*Sensors[i].Thresholds.x, 2f*Sensors[i].Thresholds.y, colors[i].Opacity(0.125f));
						}
					}
					if(Sensors[i].SensorType == ColliderType.Box) {
						Quaternion rot = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).rotation;
						Vector3 pos = editor.GetSession().GetActor().FindTransform(Sensors[i].GetName()).position;
						pos += rot * Sensors[i].Offset;
						rot *= Quaternion.Euler(Sensors[i].Rotation);
						UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
						UltiDraw.DrawCuboid(pos, rot, Sensors[i].Thresholds, colors[i].Opacity(0.25f));
						if(Sensors[i].GetContact(editor.GetCurrentFrame(), editor.Mirror) == 1f) {
							UltiDraw.DrawCuboid(pos, rot, Sensors[i].Thresholds, colors[i]);
						} else {
							UltiDraw.DrawCuboid(pos, rot, Sensors[i].Thresholds, colors[i].Opacity(0.125f));
						}
					}
				}
			}
			UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
			if(Utility.GUIButton("Capture Contacts", UltiDraw.DarkGrey, UltiDraw.White)) {
				CaptureContacts(editor);
			};
			ShowSensors = EditorGUILayout.Toggle("Show Sensors", ShowSensors);
			for(int i=0; i<Sensors.Length; i++) {
				Sensors[i].Inspector(editor);
			}
			if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
				AddSensor();
			}
			if(Utility.GUIButton("Compute Sliding", UltiDraw.DarkGrey, UltiDraw.White)) {
				ComputeSliding();
			};
		}
		#endif

		public Sensor AddSensor() {
			return AddSensor(Asset.Source.Bones[0].GetName(), Vector3.zero, Vector3.zero, 0.1f*Vector3.one, 1f, ~0, ContactType.Translational, ColliderType.Sphere);
		}

		public Sensor AddSensor(string bone, Vector3 offset, Vector3 rotation, Vector3 thresholds, float cutoffVelocity, LayerMask mask, ContactType contactType, ColliderType colliderType) {
			Sensor sensor = new Sensor(this, Asset.Source.FindBone(bone).Index, offset, rotation, thresholds, cutoffVelocity, mask, contactType, colliderType);
			ArrayExtensions.Append(ref Sensors, sensor);
			return sensor;
		}

		public void RemoveSensor(Sensor sensor) {
			if(!ArrayExtensions.Remove(ref Sensors, sensor)) {
				Debug.Log("Sensor could not be found in " + Asset.name + ".");
			}
		}

		public void Clear() {
			ArrayExtensions.Clear(ref Sensors);
		}

		public Sensor GetSensor(string bone) {
			return System.Array.Find(Sensors, x => x.GetName() == bone);
		}

		public Sensor GetSensor(string bone, Sensor reference) {
			return System.Array.Find(Sensors, x => x.GetName() == bone && x.Mask == reference.Mask && x.ContactType == reference.ContactType && x.SensorType == reference.SensorType);
		}

		public Sensor[] GetSensors(params string[] bones) {
			Sensor[] sensors = new Sensor[bones.Length];
			for(int i=0; i<sensors.Length; i++) {
				sensors[i] = GetSensor(bones[i]);
			}
			return sensors;
		}

		public string[] GetNames() {
			string[] names = new string[Sensors.Length];
			for(int i=0; i<Sensors.Length; i++) {
				names[i] = Sensors[i].GetName();
			}
			return names;
		}

		public float GetContact(float timestamp, bool mirrored, string bone) {
			Sensor sensor = GetSensor(bone);
			if(sensor == null) {
				//Debug.Log("Sensor for bone " + bones + " could not be found.");
				return 0f;
			}
			return sensor.GetContact(timestamp, mirrored);
		}

		public float[] GetContacts(float timestamp, bool mirrored) {
			float[] contacts = new float[Sensors.Length];
			for(int i=0; i<Sensors.Length; i++) {
				contacts[i] = Sensors[i].GetContact(timestamp, mirrored);
			}
			return contacts;
		}

		public float[] GetContacts(float timestamp, bool mirrored, params string[] bones) {
			float[] contacts = new float[bones.Length];
			for(int i=0; i<bones.Length; i++) {
				contacts[i] = GetSensor(bones[i]).GetContact(timestamp, mirrored);
			}
			return contacts;
		}

		public string[][] GetContactNames(float timestamp, bool mirrored, params string[] bones) {
			string[][] names = new string[bones.Length][];
			for(int i=0; i<bones.Length; i++) {
				names[i] = GetContactNames(timestamp, mirrored, bones[i]);
			}
			return names;
		}

		public string[] GetContactNames(float timestamp, bool mirrored, string bone) {
			Sensor sensor = GetSensor(bone);
			if(sensor == null) {
				//Debug.Log("Sensor for bone " + bones + " could not be found.");
				return new string[0];
			}
			return sensor.GetContactData(timestamp, mirrored).ObjectNames;
		}

		public string[] GetAllContactNames(float timestamp, bool mirrored) {
			List<string> result = new List<string>();
			for(int i=0; i<Sensors.Length; i++) {
				string[] names = Sensors[i].GetContactData(timestamp, mirrored).ObjectNames;
				for (int j = 0; j < names.Length; j++)
				{
					result.Add(names[j]);
				}
			}
			return result.Distinct().ToArray();
		}

		#if UNITY_EDITOR
		public void CaptureContacts(MotionEditor editor) {
			// editor.StartCoroutine(Process());
			// IEnumerator Process() {
				foreach(Sensor s in Sensors) {				
					s.InitContacts(Asset.Frames.Length);
				}
				// System.DateTime time = Utility.GetTimestamp();
				
				float framerate = editor.TargetFramerate;
				editor.SetTargetFramerate(Asset.Framerate);
				bool mirror = editor.Mirror;
				editor.SetMirror(false);
				float current = editor.GetCurrentFrame().Timestamp;

				foreach(Frame frame in Asset.Frames) {
					editor.LoadFrame(frame.Timestamp);
					for(int s=0; s<Sensors.Length; s++) {
						Sensors[s].CaptureContact(frame);
					}
					// if(step != 0f && Utility.GetElapsedTime(time) > step) {
					// 	time = Utility.GetTimestamp();
					// 	yield return new WaitForSeconds(0f);
					// }
				}

				editor.SetTargetFramerate(framerate);
				editor.SetMirror(mirror);
				editor.LoadFrame(current);
			// }
		}
		#endif

		public bool HasContact(Frame frame, bool mirrored, params Sensor[] sensors) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(frame, mirrored) == 1f) {
					return true;
				}
			}
			return false;
		}

		public Frame GetClosestContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
			if(HasContact(frame, mirrored, sensors)) {
				return frame;
			}
			int offset = 1;
			while(true) {
				if(frame.Index - offset < 1 && frame.Index + offset > Asset.GetTotalFrames()) {
					return null;
				}
				if(frame.Index - offset >= 1) {
					if(HasContact(Asset.GetFrame(frame.Index - offset), mirrored, sensors)) {
						return Asset.GetFrame(frame.Index - offset);
					}
				}
				if(frame.Index + offset <= Asset.GetTotalFrames()) {
					if(HasContact(Asset.GetFrame(frame.Index + offset), mirrored, sensors)) {
						return Asset.GetFrame(frame.Index + offset);
					}
				}
				offset += 1;
			}
		}

		public Frame GetNextContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index+1; i<=Asset.Frames.Last().Index; i++) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 1f) {
						return Asset.GetFrame(i);
					}
				}
			}
			return null;
		}

		public Frame GetNextContactStart(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index; i<=Asset.Frames.Last().Index-1; i++) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 0f && s.GetContact(Asset.GetFrame(i+1), mirrored) == 1f) {
						return Asset.GetFrame(i+1);
					}
				}
			}
			return null;
		}

		public Frame GetNextContactEnd(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index; i<=Asset.Frames.Last().Index-1; i++) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 1f && s.GetContact(Asset.GetFrame(i+1), mirrored) == 0f) {
						return Asset.GetFrame(i);
					}
				}
			}
			return null;
		}

		public Frame GetPreviousContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index-1; i>=Asset.Frames.First().Index; i--) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 1f) {
						return Asset.GetFrame(i);
					}
				}
			}
			return null;
		}

		public Frame GetPreviousContactStart(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index; i>=Asset.Frames.First().Index+1; i--) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 1f && s.GetContact(Asset.GetFrame(i-1), mirrored) == 0f) {
						return Asset.GetFrame(i);
					}
				}
			}
			return null;
		}

		public Frame GetPreviousContactEnd(Frame frame, bool mirrored, params Sensor[] sensors) {
			for(int i=frame.Index; i>=Asset.Frames.First().Index+1; i--) {
				foreach(Sensor s in sensors) {
					if(s.GetContact(Asset.GetFrame(i), mirrored) == 0f && s.GetContact(Asset.GetFrame(i-1), mirrored) == 1f) {
						return Asset.GetFrame(i-1);
					}
				}
			}
			return null;
		}

		public void ComputeSliding() {
			float sliding = 0f;
			int count = 0;
			for(int i=0; i<Asset.Frames.Length; i++) {
				float skating = 0f;
				for(int j=0; j<Sensors.Length; j++) {
					Vector3 velocity = Asset.Frames[i].GetBoneVelocity(Sensors[j].GetName(), false);
					float weight = Sensors[j].GetContact(Asset.Frames[i].Timestamp, false);
					skating += weight * velocity.magnitude;
				}
				sliding += skating;
				count += 1;
			}
			sliding /= count;
			Debug.Log("Sliding for asset " + Asset.name + ": " + sliding.Round(3).ToString());
		}

		[System.Serializable]
		public class Sensor {
			[System.Serializable]
			public class ContactData {
				public string[] ObjectNames = new string[0];
				public float Value { get { return ObjectNames.Length > 0 ? 1f : 0f; } }
			}
			public ContactData[] Contacts = new ContactData[0];
			public ContactType ContactType = ContactType.Translational;
			public ColliderType SensorType = ColliderType.Sphere;
			public ContactModule Module = null;
			public int Bone = 0;
			public Vector3 Offset = Vector3.zero;
			public Vector3 Rotation = Vector3.zero;
			public Vector3 Thresholds = new Vector3(0.1f, 0.1f, 0.1f);
			public float CutoffVelocity = 1f;
			public LayerMask Mask = -1;
			
			private bool Inspect = false;
			private Sensor Mirror = null;

			public Sensor(ContactModule module, int bone, Vector3 offset, Vector3 rotation, Vector3 thresholds, float cutoffVelocity, LayerMask mask, ContactType contactType, ColliderType colliderType) {
				Module = module;
				Bone = bone;
				Offset = offset;
				Rotation = rotation;
				Thresholds = thresholds;
				CutoffVelocity = cutoffVelocity;
				Mask = mask;
				ContactType = contactType;
				SensorType = colliderType;
				InitContacts(Module.Asset.Frames.Length);
			}
			
			public void InitContacts(int length){
				Contacts = new ContactData[length];
				for (int i = 0; i < Contacts.Length; i++)
				{
					Contacts[i] = new ContactData();
				}
			}

			public float GetContact(float timestamp, bool mirrored) {
				return GetContactData(timestamp, mirrored).Value;
			}

			public float GetContact(Frame frame, bool mirrored) {
				return GetContactData(frame, mirrored).Value;
			}

			public ContactData GetContactData(Frame frame, bool mirrored) {
				if(Contacts == null || Contacts.Length != Module.Asset.Frames.Length) {
					Debug.Log("Restoring missing contacts for sensor " + GetName() + " in asset " + Module.Asset.name + ".");
					InitContacts(Module.Asset.Frames.Length);
				}
				if(!mirrored) {
					return Contacts[frame.Index - 1];
				}
				if(Mirror == null) {
					Mirror = Module.GetSensor(Module.Asset.Source.Bones[Module.Asset.Symmetry[Bone]].GetName(), this);
				}
				return Mirror == null ? null : Mirror.Contacts[frame.Index - 1];
			}

			public ContactData GetContactData(float timestamp, bool mirrored) {
				float start = Module.Asset.Frames.First().Timestamp;
				float end = Module.Asset.Frames.Last().Timestamp;
				if(timestamp < start || timestamp > end) {
					float boundary = Mathf.Clamp(timestamp, start, end);
					float pivot = 2f*boundary - timestamp;
					float clamped = Mathf.Clamp(pivot, start, end);
					return GetContactData(Module.Asset.GetFrame(clamped), mirrored);
				} else {
					return GetContactData(Module.Asset.GetFrame(timestamp), mirrored);
				}
			}

			public string GetName() {
				return Module.Asset.Source.Bones[Bone].GetName();
			}

			public void SetBone(int bone) {
				Bone = bone;
			}

			public int GetIndex() {
				return System.Array.FindIndex(Module.Sensors, x => x==this);
			}

			public int GetStateWindow(Frame frame, bool mirrored) {
				float state = GetContact(frame, mirrored);
				int window = 1;
				for(int i=frame.Index-1; i>=1; i--) {
					if(GetContact(Module.Asset.GetFrame(i), mirrored) != state) {
						break;
					}
					window += 1;
				}
				for(int i=frame.Index+1; i<=Module.Asset.Frames.Length; i++) {
					if(GetContact(Module.Asset.GetFrame(i), mirrored) != state) {
						break;
					}
					window += 1;
				}
				return window;
			}

			public int GetStateWindow(Frame frame, bool mirrored, float state) {
				if(state != GetContact(frame, mirrored)) {
					return 0;
				}
				int window = 1;
				for(int i=frame.Index-1; i>=1; i--) {
					if(GetContact(Module.Asset.GetFrame(i), mirrored) != state) {
						break;
					}
					window += 1;
				}
				for(int i=frame.Index+1; i<=Module.Asset.Frames.Length; i++) {
					if(GetContact(Module.Asset.GetFrame(i), mirrored) != state) {
						break;
					}
					window += 1;
				}
				return window;
			}

			public void CaptureContact(Frame frame) {
				Collider[] GetColliders() {
					Matrix4x4 matrix = frame.GetBoneTransformation(Bone, false);
					Vector3 pos = matrix.GetPosition() + matrix.GetRotation() * Offset;
					Quaternion rot = matrix.GetRotation() * Quaternion.Euler(Rotation);
					if(SensorType == ColliderType.Sphere) {
						return Physics.OverlapSphere(pos, Thresholds.x, Mask);
					}
					if(SensorType == ColliderType.Capsule) {
						Vector3 start = pos - rot * new Vector3(0f, Thresholds.y, 0f);
						Vector3 end = pos + rot * new Vector3(0f, Thresholds.y, 0f);
						return Physics.OverlapCapsule(start, end, Thresholds.x, Mask);
					}
					if(SensorType == ColliderType.Box) {
						return Physics.OverlapBox(pos, Thresholds/2f, rot, Mask);
					}
					return null;
				}
				Collider[] colliders = GetColliders();
				if(colliders.Length > 0) {
					string[] names = colliders.Select(x => x.gameObject.name).Distinct().ToArray();
					if(ContactType == ContactType.Translational) {
						Vector3 velocity = frame.GetBoneVelocity(Bone, false);
						float threshold = CutoffVelocity;
						if(velocity.magnitude <= threshold) {
							Contacts[frame.Index-1].ObjectNames = names;
						}
					}
					if(ContactType == ContactType.Rotational) {
						float velocity = frame.GetAngularVelocity(Bone, false);
						float threshold = CutoffVelocity;
						if(velocity <= threshold) {
							Contacts[frame.Index-1].ObjectNames = names;
						}
					}
				}
			}
#if UNITY_EDITOR
			public void Inspector(MotionEditor editor) {
				UltiDraw.Begin();
				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Bone", GUILayout.Width(40f));
					SetBone(EditorGUILayout.Popup(Bone, Module.Asset.Source.GetBoneNames(), GUILayout.Width(150f)));
					EditorGUILayout.LabelField("Mask", GUILayout.Width(40f));
					Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers, GUILayout.Width(100f)));
					EditorGUILayout.LabelField("Contact Type", GUILayout.Width(40f));
					ContactType = (ContactType)EditorGUILayout.EnumPopup(ContactType, GUILayout.Width(100f));
					EditorGUILayout.LabelField("Sensor Type", GUILayout.Width(40f));
					SensorType = (ColliderType)EditorGUILayout.EnumPopup(SensorType, GUILayout.Width(100f));
					EditorGUILayout.LabelField("Cutoff Velocity", GUILayout.Width(90f));
					CutoffVelocity = EditorGUILayout.FloatField(CutoffVelocity, GUILayout.Width(60f));
					if(Utility.GUIButton("Inspect", Inspect ? UltiDraw.Cyan : UltiDraw.DarkGrey, Inspect ? UltiDraw.Black : UltiDraw.White, 80f, 20f)) {
						Inspect = !Inspect;
					}
					if(Utility.GUIButton("-", UltiDraw.DarkRed, UltiDraw.White, 28f, 18f)) {
						Module.RemoveSensor(this);
						EditorGUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					Frame frame = editor.GetCurrentFrame();

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
					if(endTime > Module.Asset.GetTotalTime()) {
						startTime -= endTime-Module.Asset.GetTotalTime();
						endTime = Module.Asset.GetTotalTime();
					}
					startTime = Mathf.Max(0f, startTime);
					endTime = Mathf.Min(Module.Asset.GetTotalTime(), endTime);
					int start = Module.Asset.GetFrame(startTime).Index;
					int end = Module.Asset.GetFrame(endTime).Index;
					int elements = end-start;

					Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
					Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

					start = Mathf.Clamp(start, 1, Module.Asset.Frames.Length);
					end = Mathf.Clamp(end, 1, Module.Asset.Frames.Length);

					//Contacts
					for(int i=start; i<=end; i++) {
						if(GetContact(Module.Asset.GetFrame(i), editor.Mirror) == 1f) {
							float left = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							float right = left;
							while(i<end && GetContact(Module.Asset.GetFrame(i), editor.Mirror) == 1f) {
								i++;
								right = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							}
							if(left != right) {
								Vector3 a = new Vector3(left, rect.y, 0f);
								Vector3 b = new Vector3(right, rect.y, 0f);
								Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
								Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
								UltiDraw.DrawTriangle(a, c, b, ContactType == ContactType.Translational ? UltiDraw.Green : UltiDraw.Magenta);
								UltiDraw.DrawTriangle(b, c, d, ContactType == ContactType.Translational ? UltiDraw.Green : UltiDraw.Magenta);
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

					string[] names = GetContactData(frame, editor.Mirror).ObjectNames;
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("[ " + names.Length + " ] Objects:");
					for (int i = 0; i < names.Length; i++)
					{
						EditorGUILayout.LabelField(i+1 + ": " + names[i]);
					}
					EditorGUILayout.EndHorizontal();
					
					if(Inspect) {
						Utility.SetGUIColor(UltiDraw.White);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							switch(SensorType) {
								case ColliderType.Sphere:
								Offset = EditorGUILayout.Vector3Field("Offset", Offset);
								Thresholds.x = EditorGUILayout.FloatField("Radius", Thresholds.x);
								Thresholds.y = Thresholds.x;
								Thresholds.z = Thresholds.x;
								break;

								case ColliderType.Capsule:
								Offset = EditorGUILayout.Vector3Field("Offset", Offset);
								Rotation = EditorGUILayout.Vector3Field("Rotation", Rotation);
								Thresholds.x = EditorGUILayout.FloatField("Radius", Thresholds.x);
								Thresholds.y = EditorGUILayout.FloatField("Length", Thresholds.y);
								Thresholds.z = Thresholds.x;
								break;
							}
						}
					}
				}
				UltiDraw.End();
			}
#endif
		}
		public class Series : TimeSeries.Component {
			public string[] Bones;
			public float[][] Values;
			
			// private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.7f, 0.2f, 0.1f);
			private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.2f, 0.8f, 0.3f, 0.15f);

			public Series(TimeSeries global, params string[] bones) : base(global) {
				Bones = bones;
				Values = new float[Samples.Length][];
				for(int i=0; i<Values.Length; i++) {
					Values[i] = new float[Bones.Length];
				}
			}

			public float[] GetContacts(int index) {
				return Values[index];
			}

			public float[] GetContacts(int index, params string[] bones) {
				float[] values = new float[bones.Length];
				for(int i=0; i<bones.Length; i++) {
					values[i] = GetContact(index, bones[i]);
				}
				return values;
			}

			public float GetContact(int index, string bone) {
				int idx = ArrayExtensions.FindIndex(ref Bones, bone);
				if(idx == -1) {
					Debug.Log("Contact " + bone + " could not be found.");
					return 0f;
				}
				return Values[index][idx];
			}

			public override void Increment(int start, int end) {
				for(int i=start; i<end; i++) {
					for(int j=0; j<Bones.Length; j++) {
						Values[i][j] = Values[i+1][j];
					}
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.GUIRect area = rect == null ? Rect : rect;
					UltiDraw.Begin();
					UltiDraw.OnGUILabel(area.GetCenter() + new Vector2(0f, 0.5f*area.H + 0.025f), area.GetSize(), 0.0175f, "Contacts", UltiDraw.White);
					UltiDraw.End();
				}
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.GUIRect area = rect == null ? Rect : rect;
					UltiDraw.Begin();
					float[] function = new float[Samples.Length];
					Color[] colors = UltiDraw.GetRainbowColors(Bones.Length);
					for(int i=0; i<Bones.Length; i++) {
						for(int j=0; j<function.Length; j++) {
							function[j] = Values[j][Bones.Length-1-i];
						}
						float ratio = i.Ratio(Bones.Length-1, 0);
						float itemSize = area.H / Bones.Length;
						UltiDraw.PlotBars(new Vector2(area.X, ratio.Normalize(0f, 1f, area.Y + area.H/2f - itemSize/2f, area.Y - area.H/2f + itemSize/2f)), new Vector2(area.W, itemSize), function, yMin: 0f, yMax: 1f, barColor : colors[i]);
					}
					UltiDraw.End();
				}
			}
		}
	}
}
