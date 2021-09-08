#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

public class ContactModule : Module {

	public bool ShowSensors = true;

	public Sensor[] Sensors = new Sensor[0];

	private string[] Names = null;

	private Precomputable<float[]>[] PrecomputedRegularContacts = null;
	private Precomputable<float[]>[] PrecomputedInverseContacts = null;

	public override ID GetID() {
		return ID.Contact;
	}

	public override void DerivedResetPrecomputation() {
		PrecomputedRegularContacts = Data.ResetPrecomputable(PrecomputedRegularContacts);
		PrecomputedInverseContacts = Data.ResetPrecomputable(PrecomputedInverseContacts);
	}

	public override ComponentSeries DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored) {
		ContactSeries instance = new ContactSeries(global, GetNames());
		for(int i=0; i<instance.Samples.Length; i++) {
			instance.Values[i] = GetContacts(timestamp + instance.Samples[i].Timestamp, mirrored);
		}
		return instance;
	}

	protected override void DerivedInitialize() {

	}

	protected override void DerivedLoad(MotionEditor editor) {
		
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
				for(int j=0; j<Sensors[i].Bones.Length; j++) {
					Quaternion rot = editor.GetActor().FindTransform(Sensors[i].GetBoneName(j)).rotation;
					Vector3 pos = editor.GetActor().FindTransform(Sensors[i].GetBoneName(j)).position + rot * Sensors[i].Offset;
					UltiDraw.DrawCube(pos, rot, 0.025f, UltiDraw.Black);
					UltiDraw.DrawWireSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Opacity(0.25f));
					if(Sensors[i].GetContact(editor.GetCurrentFrame(), editor.Mirror) == 1f) {
						UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i]);
					} else {
						UltiDraw.DrawSphere(pos, rot, 2f*Sensors[i].Threshold, colors[i].Opacity(0.125f));
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

	public Sensor AddSensor() {
		return AddSensor(string.Empty, new string[0], Vector3.zero, 0f, 0f, ~0);
	}

	public Sensor AddSensor(string id, string bone, Vector3 offset, float threshold, float velocity, LayerMask mask) {
		return AddSensor(id, new string[]{bone}, offset, threshold, velocity, mask);
	}

	public Sensor AddSensor(string id, string[] bones, Vector3 offset, float threshold, float velocity, LayerMask mask) {
		Names = null;
		Sensor sensor = new Sensor(this, id, Data.Source.GetBoneIndices(bones), offset, threshold, velocity);
		sensor.Mask = mask;
		ArrayExtensions.Append(ref Sensors, sensor);
		return sensor;
	}

	public void RemoveSensor(Sensor sensor) {
		if(!ArrayExtensions.Remove(ref Sensors, sensor)) {
			Debug.Log("Sensor could not be found in " + Data.GetName() + ".");
		} else {
			Names = null;
		}
	}

	public void Clear() {
		ArrayExtensions.Clear(ref Sensors);
	}

	public Sensor GetSensor(string name) {
		return System.Array.Find(Sensors, x => x.ID == name);
	}

	public Sensor[] GetSensors(params string[] names) {
		Sensor[] sensors = new Sensor[names.Length];
		for(int i=0; i<sensors.Length; i++) {
			sensors[i] = GetSensor(names[i]);
		}
		return sensors;
	}

	public string[] GetNames() {
		if(Names == null || !Names.Verify(Sensors.Length)) {
			Names = new string[Sensors.Length];
			for(int i=0; i<Sensors.Length; i++) {
				Names[i] = Sensors[i].ID;
			}
		}
		return Names;
	}

	public float[] GetContacts(float timestamp, bool mirrored) {
		if(Data.IsPrecomputable(timestamp)) {
			int index = Data.GetPrecomputedIndex(timestamp);
			if(mirrored && PrecomputedInverseContacts[index] == null) {
				PrecomputedInverseContacts[index] = new Precomputable<float[]>(Compute());
			}
			if(!mirrored && PrecomputedRegularContacts[index] == null) {
				PrecomputedRegularContacts[index] = new Precomputable<float[]>(Compute());
			}
			return mirrored ? PrecomputedInverseContacts[index].Value : PrecomputedRegularContacts[index].Value;
		}

		return Compute();
		float[] Compute() {
			float[] contacts = new float[Sensors.Length];
			for(int i=0; i<Sensors.Length; i++) {
				contacts[i] = Sensors[i].GetContact(timestamp, mirrored);
			}
			return contacts;
		}
	}

	// public float[] GetContacts(float timestamp, bool mirrored, params string[] bones) {
	// 	float[] contacts = new float[bones.Length];
	// 	for(int i=0; i<bones.Length; i++) {
	// 		Sensor sensor = GetSensor(bones[i]);
	// 		if(sensor == null) {
	// 			Debug.Log("Sensor for bone " + bones[i] + " could not be found.");
	// 			contacts[i] = 0f;
	// 		} else {
	// 			contacts[i] = sensor.GetContact(timestamp, mirrored);
	// 		}
	// 	}
	// 	return contacts;
	// }

	public void CaptureContacts(MotionEditor editor) {
		// editor.StartCoroutine(Process());
		// IEnumerator Process() {
			foreach(Sensor s in Sensors) {
				s.Contacts = new float[Data.Frames.Length];
			}
			// System.DateTime time = Utility.GetTimestamp();
			float framerate = editor.TargetFramerate;
			
			Frame current = editor.GetCurrentFrame();
			editor.TargetFramerate = Data.Framerate;
			foreach(Frame frame in Data.Frames) {
				editor.LoadFrame(frame);
				for(int s=0; s<Sensors.Length; s++) {
					Sensors[s].CaptureContact(frame, editor);
				}
				// if(step != 0f && Utility.GetElapsedTime(time) > step) {
				// 	time = Utility.GetTimestamp();
				// 	yield return new WaitForSeconds(0f);
				// }
			}
			editor.TargetFramerate = framerate;
			editor.LoadFrame(current);
		// }
	}

	public bool HasContact(Frame frame, bool mirrored, params Sensor[] sensors) {
		foreach(Sensor s in sensors) {
			if(s.GetContact(frame, mirrored) == 1f) {
				return true;
			}
		}
		return false;
	}

	public bool HasContacts(Frame frame, bool mirrored, params Sensor[] sensors) {
		foreach(Sensor s in sensors) {
			if(s.GetContact(frame, mirrored) == 0f) {
				return false;
			}
		}
		return true;
	}

	public Frame GetClosestAnyContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
		if(HasContact(frame, mirrored, sensors)) {
			return frame;
		}
		int offset = 1;
		while(true) {
			if(frame.Index - offset < 1 && frame.Index + offset > Data.GetTotalFrames()) {
				return null;
			}
			if(frame.Index - offset >= 1) {
				if(HasContact(Data.GetFrame(frame.Index - offset), mirrored, sensors)) {
					return Data.GetFrame(frame.Index - offset);
				}
			}
			if(frame.Index + offset <= Data.GetTotalFrames()) {
				if(HasContact(Data.GetFrame(frame.Index + offset), mirrored, sensors)) {
					return Data.GetFrame(frame.Index + offset);
				}
			}
			offset += 1;
		}
	}

	public Frame GetClosestGroupContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
		if(HasContact(frame, mirrored, sensors)) {
			return frame;
		}
		int offset = 1;
		while(true) {
			if(frame.Index - offset < 1 && frame.Index + offset > Data.GetTotalFrames()) {
				return null;
			}
			if(frame.Index - offset >= 1) {
				if(HasContacts(Data.GetFrame(frame.Index - offset), mirrored, sensors)) {
					return Data.GetFrame(frame.Index - offset);
				}
			}
			if(frame.Index + offset <= Data.GetTotalFrames()) {
				if(HasContacts(Data.GetFrame(frame.Index + offset), mirrored, sensors)) {
					return Data.GetFrame(frame.Index + offset);
				}
			}
			offset += 1;
		}
	}

	public Frame GetNextContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index+1; i<=Data.Frames.Last().Index; i++) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 1f) {
					return Data.GetFrame(i);
				}
			}
		}
		return null;
	}

	public Frame GetNextContactStart(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index; i<=Data.Frames.Last().Index-1; i++) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 0f && s.GetContact(Data.GetFrame(i+1), mirrored) == 1f) {
					return Data.GetFrame(i+1);
				}
			}
		}
		return null;
	}

	public Frame GetNextContactEnd(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index; i<=Data.Frames.Last().Index-1; i++) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 1f && s.GetContact(Data.GetFrame(i+1), mirrored) == 0f) {
					return Data.GetFrame(i);
				}
			}
		}
		return null;
	}

	public Frame GetPreviousContactFrame(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index-1; i>=Data.Frames.First().Index; i--) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 1f) {
					return Data.GetFrame(i);
				}
			}
		}
		return null;
	}

	public Frame GetPreviousContactStart(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index; i>=Data.Frames.First().Index+1; i--) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 1f && s.GetContact(Data.GetFrame(i-1), mirrored) == 0f) {
					return Data.GetFrame(i);
				}
			}
		}
		return null;
	}

	public Frame GetPreviousContactEnd(Frame frame, bool mirrored, params Sensor[] sensors) {
		for(int i=frame.Index; i>=Data.Frames.First().Index+1; i--) {
			foreach(Sensor s in sensors) {
				if(s.GetContact(Data.GetFrame(i), mirrored) == 0f && s.GetContact(Data.GetFrame(i-1), mirrored) == 1f) {
					return Data.GetFrame(i-1);
				}
			}
		}
		return null;
	}

	[System.Serializable]
	public class Sensor {
		public ContactModule Module = null;
		public string ID = string.Empty;
		public string Mirror = string.Empty;
		public int[] Bones = new int[0];
		public Vector3 Offset = Vector3.zero;
		public float Threshold = 0.1f;
		public float Velocity = 0f;
		public LayerMask Mask = -1;

		public float[] Contacts = new float[0];

		private Sensor MirrorSensor = null;

		public Sensor(ContactModule module, string id, int bone, Vector3 offset, float threshold, float velocity) {
			Module = module;
			ID = id;
			Bones = new int[1]{bone};
			Offset = offset;
			Threshold = threshold;
			Velocity = velocity;
			Contacts = new float[Module.Data.Frames.Length];
		}

		public Sensor(ContactModule module, string id, int[] bones, Vector3 offset, float threshold, float velocity) {
			Module = module;
			ID = id;
			Bones = bones;
			Offset = offset;
			Threshold = threshold;
			Velocity = velocity;
			Contacts = new float[Module.Data.Frames.Length];
		}

		public float GetContact(float timestamp, bool mirrored) {
			float start = Module.Data.GetFirstValidFrame().Timestamp;
			float end = Module.Data.GetLastValidFrame().Timestamp;
			if(timestamp < start || timestamp > end) {
				float boundary = Mathf.Clamp(timestamp, start, end);
				float pivot = 2f*boundary - timestamp;
				float clamped = Mathf.Clamp(pivot, start, end);
				return GetContact(Module.Data.GetFrame(clamped), mirrored);
			} else {
				return GetContact(Module.Data.GetFrame(timestamp), mirrored);
			}
		}

		public float GetContact(Frame frame, bool mirrored) {
			if(Contacts == null || Contacts.Length != Module.Data.Frames.Length) {
				Debug.Log("Restoring missing contacts for sensor " + ID + " in asset " + Module.Data.GetName() + ".");
				Contacts = new float[Module.Data.Frames.Length];
			}
			if(!mirrored) {
				return Contacts[frame.Index-1];
			}
			if(MirrorSensor == null) {
				MotionData.Hierarchy.Bone bone = Module.Data.Source.FindBone(ID);
				if(bone != null) {
					MirrorSensor = Module.GetSensor(Module.Data.Source.Bones[Module.Data.Symmetry[bone.Index]].Name);
				}
			}
			return MirrorSensor == null ? 0f : MirrorSensor.Contacts[frame.Index-1];
		}

		public void SetName(string name) {
			if(ID != name) {
				ID = name;
				foreach(Sensor s in Module.Sensors) {
					s.MirrorSensor = null;
				}
			}
		}

		public void SetBone(int index, int bone) {
			Bones[index] = bone;
		}

		public string GetBoneName(int index) {
			return Module.Data.Source.Bones[Bones[index]].Name;
		}

		public int GetIndex() {
			return System.Array.FindIndex(Module.Sensors, x => x==this);
		}

		public void AddBone(int index) {
			ArrayExtensions.Append(ref Bones, index);
		}

		public void RemoveBone(int index) {
			ArrayExtensions.RemoveAt(ref Bones, index);
		}

		// public float GetStateVariation(Frame frame, bool mirrored, int window) {
		// 	float[] timestamps = Module.Data.SimulateTimestamps(frame, window);
		// 	float sum = 0f;
		// 	for(int i=0; i<timestamps.Length; i++) {
		// 		sum += Mathf.Abs(GetContact(frame, mirrored) - GetContact(timestamps[i], mirrored));
		// 	}
		// 	return sum / timestamps.Length;
		// }

		public int GetStateWindow(Frame frame, bool mirrored) {
			float state = GetContact(frame, mirrored);
			int window = 1;
			for(int i=frame.Index-1; i>=1; i--) {
				if(GetContact(Module.Data.GetFrame(i), mirrored) != state) {
					break;
				}
				window += 1;
			}
			for(int i=frame.Index+1; i<=Module.Data.Frames.Length; i++) {
				if(GetContact(Module.Data.GetFrame(i), mirrored) != state) {
					break;
				}
				window += 1;
			}
			return window;
		}

		public void CaptureContact(Frame frame, MotionEditor editor) {
			for(int i=0; i<Bones.Length; i++) {
				Matrix4x4 matrix = frame.GetBoneTransformation(Bones[i], false);
				Vector3 velocity = frame.GetBoneVelocity(Bones[i], false);
				Contacts[frame.Index-1] = Physics.CheckSphere(matrix.GetPosition() + matrix.GetRotation() * Offset, Threshold, Mask) ? 1f : 0f;
				if(Velocity > 0f) {
					if(GetContact(frame, false) == 1f && velocity.magnitude > Velocity) {
						Contacts[frame.Index-1] = 0f;
					}
				}
				if(Contacts[frame.Index-1] == 1f) {
					return;
				}
			}
		}

		public void Inspector(MotionEditor editor) {
			UltiDraw.Begin();
			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White, 20f, 20f)) {
					AddBone(0);
				}
				EditorGUILayout.LabelField("Group", GUILayout.Width(40f));
				SetName(EditorGUILayout.TextField(ID, GUILayout.Width(100f)));
				EditorGUILayout.LabelField("Mask", GUILayout.Width(40));
				Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers, GUILayout.Width(100f)));
				EditorGUILayout.LabelField("Offset", GUILayout.Width(40f));
				Offset = EditorGUILayout.Vector3Field("", Offset, GUILayout.Width(125f));
				EditorGUILayout.LabelField("Threshold", GUILayout.Width(60f));
				Threshold = EditorGUILayout.FloatField(Threshold, GUILayout.Width(30f));
				EditorGUILayout.LabelField("Velocity", GUILayout.Width(60f));
				Velocity = EditorGUILayout.FloatField(Velocity, GUILayout.Width(30f));
				EditorGUILayout.EndHorizontal();

				for(int i=0; i<Bones.Length; i++) {
					EditorGUILayout.BeginHorizontal();
					if(Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White, 20f, 20f)) {
						RemoveBone(i);
						i--;
					} else {
						SetBone(i, EditorGUILayout.Popup(Bones[i], editor.GetAsset().Source.GetBoneNames()));
					}
					EditorGUILayout.EndHorizontal();
				}

				Frame frame = editor.GetCurrentFrame();
				MotionData data = editor.GetAsset();

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
					if(GetContact(data.GetFrame(i), editor.Mirror) == 1f) {
						float left = rect.xMin + (float)(i-start)/(float)elements * rect.width;
						float right = left;
						while(i<end && GetContact(data.GetFrame(i), editor.Mirror) == 1f) {
							i++;
							right = rect.xMin + (float)(i-start)/(float)elements * rect.width;
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