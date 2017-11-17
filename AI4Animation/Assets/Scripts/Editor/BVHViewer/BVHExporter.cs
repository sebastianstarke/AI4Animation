using UnityEngine;
using UnityEditor;
using System.IO;

public class BVHExporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public bool[] Use = new bool[0];
	public BVHAnimation[] Animations = new BVHAnimation[0];

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Addons/BVH Exporter")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHExporter));
		Scroll = Vector3.zero;
	}

	void OnGUI() {
		Utility.SetGUIColor(Utility.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(Utility.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(Utility.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Exporter");
				}

				if(Utility.GUIButton("Export Labels", Utility.DarkGrey, Utility.White)) {
					ExportLabels();
				}
				if(Utility.GUIButton("Export Data", Utility.DarkGrey, Utility.White)) {
					ExportData();
				}

				Scroll = EditorGUILayout.BeginScrollView(Scroll);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					SetDirectory(EditorGUILayout.TextField(Directory));
					EditorGUILayout.EndHorizontal();
					for(int i=0; i<Animations.Length; i++) {
						if(Use[i]) {
							Utility.SetGUIColor(Utility.DarkGreen);
						} else {
							Utility.SetGUIColor(Utility.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							Use[i] = EditorGUILayout.Toggle(Use[i], GUILayout.Width(20f));
							Animations[i] = (BVHAnimation)EditorGUILayout.ObjectField(Animations[i], typeof(BVHAnimation), true);
							EditorGUILayout.EndHorizontal();
						}
					}
				}

				EditorGUILayout.EndScrollView();
			}
		}
	}

	private void SetDirectory(string dir) {
		if(Directory != dir) {
			Directory = dir;
			Animations = new BVHAnimation[0];
			Use = new bool[0];
			string path = "Assets/"+Directory;
			if(AssetDatabase.IsValidFolder(path)) {
				string[] elements = AssetDatabase.FindAssets("t:BVHAnimation", new string[1]{path});
				Animations = new BVHAnimation[elements.Length];
				Use = new bool[elements.Length];
				for(int i=0; i<elements.Length; i++) {
					Animations[i] = (BVHAnimation)AssetDatabase.LoadAssetAtPath(AssetDatabase.GUIDToAssetPath(elements[i]), typeof(BVHAnimation));
					Use[i] = true;
				}
			}
		}
	}
	
	private void ExportLabels() {
		if(Animations.Length == 0) {
			Debug.Log("No animations specified.");
			return;
		}
		
		string name = "Labels";
		string filename = string.Empty;
		if(!File.Exists(Application.dataPath+"/Animation/Project/"+name+".txt")) {
			filename = Application.dataPath+"/Animation/Project/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Animation/Project/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Animation/Project/"+name+" ("+i+")";
		}

		StreamWriter labels = File.CreateText(filename+".txt");
		int index = 0;
		labels.WriteLine(index + " " + "Sequence"); index += 1;
		labels.WriteLine(index + " " + "Frame"); index += 1;
		labels.WriteLine(index + " " + "Timestamp"); index += 1;
		for(int i=1; i<=Animations[0].Character.Bones.Length; i++) {
			labels.WriteLine(index + " " + "BonePositionX"+i); index += 1;
			labels.WriteLine(index + " " + "BonePositionY"+i); index += 1;
			labels.WriteLine(index + " " + "BonePositionZ"+i); index += 1;
			labels.WriteLine(index + " " + "BoneVelocityX"+i); index += 1;
			labels.WriteLine(index + " " + "BoneVelocityY"+i); index += 1;
			labels.WriteLine(index + " " + "BoneVelocityZ"+i); index += 1;
		}
		for(int i=1; i<=12; i++) {
			labels.WriteLine(index + " " + "TrajectoryPositionX"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryPositionY"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryPositionZ"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryDirectionX"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryDirectionY"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryDirectionZ"+i); index += 1;
		}
		for(int i=1; i<=12; i++) {
			labels.WriteLine(index + " " + "TrajectoryLeftHeight"+i); index += 1;
			labels.WriteLine(index + " " + "TrajectoryRightHeight"+i); index += 1;
		}
		for(int i=1; i<=12; i++) {
			for(int j=1; j<=Animations[0].StyleFunction.Styles.Length; j++) {
				labels.WriteLine(index + " " + Animations[0].StyleFunction.Styles[j-1].Name + i); index += 1;
			}
		}
		labels.WriteLine(index + " " + "Phase"); index += 1;
		labels.WriteLine(index + " " + "RootTranslationalVelocityX"); index += 1;
		labels.WriteLine(index + " " + "RootTranslationalVelocityZ"); index += 1;
		labels.WriteLine(index + " " + "RootAngularVelocity"); index += 1;
		labels.WriteLine(index + " " + "PhaseChange"); index += 1;
		labels.Close();
	}

	//TODO: EVERYTHING ASSUMES 60HZ!!!
	private void ExportData() {
		if(Animations.Length == 0) {
			Debug.Log("No animations specified.");
			return;
		}
		
		string name = "Data";
		string filename = string.Empty;
		if(!File.Exists(Application.dataPath+"/Animation/Project/"+name+".txt")) {
			filename = Application.dataPath+"/Animation/Project/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Animation/Project/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Animation/Project/"+name+" ("+i+")";
		}

		StreamWriter data = File.CreateText(filename+".txt");
		int sequence = 0;
		for(int i=0; i<Animations.Length; i++) {
			if(Use[i]) {
				sequence += 1;
				//float timeStart = Animations[i].GetFrame(Animations[i].SequenceStart).Timestamp;
				//float timeEnd = Animations[i].GetFrame(Animations[i].SequenceEnd).Timestamp;
				//for(float j=timeStart; j<=timeEnd; j+=1f/60f) {
				int startIndex = Animations[i].SequenceStart;
				int endIndex = Animations[i].SequenceEnd;
				for(int j=startIndex; j<=endIndex; j++) {
					//Get frame
					BVHAnimation.BVHFrame frame = Animations[i].GetFrame(j);
					//BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1f/60f, 0f, Animations[i].TotalTime));
					BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1, 1, Animations[i].TotalFrames));

					//j = frame.Timestamp;

					//Sequence number
					string line = sequence + Separator;

					//Frame index
					line += frame.Index + Separator;

					//Frame time
					line += frame.Timestamp + Separator;

					Trajectory trajectory = Animations[i].ExtractTrajectory(frame);
					Transformation root = new Transformation(trajectory.Points[6].GetPosition(), trajectory.Points[6].GetRotation());
					//Bone data
					for(int k=0; k<Animations[i].Character.Bones.Length; k++) {
						//Position
						line += FormatVector3(frame.Positions[k].RelativePositionTo(root));

						//Velocity
						line += FormatVector3(frame.SmoothTranslationalVelocityVector(k, 0.1f).RelativeDirectionTo(root));
					}
					
					//Trajectory data
					for(int k=0; k<12; k++) {
						line += FormatVector3(trajectory.Points[k].GetPosition().RelativePositionTo(root));
						line += FormatVector3(trajectory.Points[k].GetDirection().RelativeDirectionTo(root));
					}

					for(int k=0; k<12; k++) {
						line += FormatValue(trajectory.Points[k].GetRightSample().y - root.Position.y);
						line += FormatValue(trajectory.Points[k].GetLeftSample().y - root.Position.y);
					}

					for(int k=0; k<12; k++) {
						line += FormatArray(trajectory.Points[k].Styles);
					}

					//Phase
					line += FormatValue(Animations[i].PhaseFunction.GetPhase(frame));

					//ADDITIONAL
					Vector3 position = Animations[i].GetRootPosition(frame);
					Vector3 prevPosition = Animations[i].GetRootPosition(prevFrame);
					Vector3 direction = Animations[i].GetRootDirection(frame);
					Vector3 prevDirection = Animations[i].GetRootDirection(prevFrame);

					//Translational root velocity
					Vector3 translationOffset = Quaternion.Inverse(Quaternion.LookRotation(prevDirection, Vector3.up)) * (position - prevPosition);
					line += FormatValue(translationOffset.x);
					line += FormatValue(translationOffset.z);

					//Angular root velocity
					float rotationOffset = Vector3.Angle(prevDirection, direction);
					line += FormatValue(rotationOffset);

					//Phase change
					line += FormatValue(GetPhaseChange(Animations[i].PhaseFunction.GetPhase(prevFrame), Animations[i].PhaseFunction.GetPhase(frame)));

					//Postprocess
					line = line.Remove(line.Length-1);
					line = line.Replace(",",".");

					//Write
					data.WriteLine(line);
				}
			}
		}
		data.Close();
	}

	private float GetPhaseChange(float prev, float next) {
		return Mathf.Repeat(((next-prev) + 1f), 1f);
	}

	private string FormatString(string value) {
		return value + Separator;
	}

	private string FormatValue(float value) {
		return value.ToString(Accuracy) + Separator;
	}

	private string FormatArray(float[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			format += array[i].ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string FormatVector3(Vector3 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator + vector.z.ToString(Accuracy) + Separator;
	}

	/*
	private string FormatQuaternion(Quaternion quaternion) {
		return quaternion.x + Separator + quaternion.y + Separator + quaternion.z + Separator + quaternion.w + Separator;
	}
	*/

}
