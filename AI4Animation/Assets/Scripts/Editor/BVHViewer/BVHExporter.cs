using UnityEngine;
using UnityEditor;
using System.IO;

public class BVHExporter : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public int Files = 0;
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

				Scroll = EditorGUILayout.BeginScrollView(Scroll);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					SetFiles(EditorGUILayout.IntField("Files", Files));
					for(int i=0; i<Animations.Length; i++) {
						if(Animations[i] == null) {
							Utility.SetGUIColor(Utility.DarkRed);
						} else {
							Utility.SetGUIColor(Utility.DarkGreen);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							Animations[i] = (BVHAnimation)EditorGUILayout.ObjectField((i+1).ToString(), Animations[i], typeof(BVHAnimation), true);
						}
					}
				}

				if(Utility.GUIButton("Export Labels", Utility.DarkGrey, Utility.White)) {
					ExportLabels();
				}
				if(Utility.GUIButton("Export Data", Utility.DarkGrey, Utility.White)) {
					ExportData();
				}
				EditorGUILayout.EndScrollView();
			}
		}
	}

	private void SetFiles(int files) {
		files = Mathf.Max(files, 0);
		if(Files != files) {
			Files = files;
			System.Array.Resize(ref Animations, files);
		}
	}

	private void ExportLabels() {
		if(Animations.Length == 0) {
			Debug.Log("No animations specified.");
			return;
		}
		
		string name = "Labels";
		string filename = string.Empty;
		if(!File.Exists(Application.dataPath+"/Animation/"+name+".txt")) {
			filename = Application.dataPath+"/Animation/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Animation/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Animation/"+name+" ("+i+")";
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
		if(!File.Exists(Application.dataPath+"/Animation/"+name+".txt")) {
			filename = Application.dataPath+"/Animation/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Animation/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Animation/"+name+" ("+i+")";
		}

		StreamWriter data = File.CreateText(filename+".txt");
		for(int i=0; i<Animations.Length; i++) {
			if(Animations[i] != null) {
				//float timeStart = Animations[i].GetFrame(Animations[i].SequenceStart).Timestamp;
				//float timeEnd = Animations[i].GetFrame(Animations[i].SequenceEnd).Timestamp;
				//for(float j=timeStart; j<=timeEnd; j+=1f/60f) {
				int startIndex = Animations[i].SequenceStart;
				int endIndex = Animations[i].SequenceEnd;
				for(int j=startIndex; j<=endIndex; j++) {
					//Get frame
					BVHAnimation.BVHFrame frame = Animations[i].GetFrame(j);

					//j = frame.Timestamp;

					//File number
					string line = (i+1) + Separator;

					//Frame index
					line += frame.Index + Separator;

					//Frame time
					line += frame.Timestamp + Separator;

					Trajectory trajectory = Animations[i].GenerateTrajectory(frame);
					Transformation root = new Transformation(trajectory.GetRoot().GetPosition(), trajectory.GetRoot().GetRotation());
					//Bone data
					for(int k=0; k<Animations[i].Character.Bones.Length; k++) {
						//Position
						line += FormatVector3(frame.Positions[k].RelativePositionTo(root));
						//Velocity
						//line += FormatVector3(frame.SmoothTranslationalVelocityVector(k, 0.25f).RelativeDirectionTo(root));
						line += FormatVector3(frame.Velocities[k].RelativePositionTo(root));
					}
					
					//Trajectory data
					for(int k=0; k<trajectory.GetSampleCount(); k++) {
						line += FormatVector3(trajectory.GetSample(k).GetPosition().RelativePositionTo(root));
						line += FormatVector3(trajectory.GetSample(k).GetDirection().RelativeDirectionTo(root));
					}

					for(int k=0; k<trajectory.GetSampleCount(); k++) {
						line += FormatValue(trajectory.GetSample(k).SampleSide(-trajectory.Width/0.25f).y - root.Position.y);
						line += FormatValue(trajectory.GetSample(k).SampleSide(trajectory.Width/0.25f).y - root.Position.y);
					}

					for(int k=0; k<trajectory.GetSampleCount(); k++) {
						line += FormatArray(trajectory.GetSample(k).Styles);
					}

					//Phase
					//line += FormatValue(Animations[i].PhaseFunction.GetPhase(frame));
					line += FormatValue(0.5f*Mathf.Sin(Animations[i].PhaseFunction.GetPhase(frame)*2f*Mathf.PI) + 0.5f);

					//ADDITIONAL
					//Get previous frame
					//BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1f/60f, 0f, Animations[i].TotalTime));
					BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1, 1, Animations[i].TotalFrames));
					Vector3 position = Animations[i].GetRootPosition(frame);
					Vector3 prevPosition = Animations[i].GetRootPosition(prevFrame);
					Vector3 direction = Animations[i].GetRootDirection(frame);
					Vector3 prevDirection = Animations[i].GetRootDirection(prevFrame);
					//Offsets
					Vector3 translationOffset = Quaternion.Inverse(Quaternion.LookRotation(prevDirection, Vector3.up)) * (position - prevPosition);
					line += FormatValue(translationOffset.x);
					line += FormatValue(translationOffset.z);
					float rotationOffset = Vector3.Angle(prevDirection, direction);
					line += FormatValue(rotationOffset);
					float phaseChange = Animations[i].PhaseFunction.GetPhase(frame) - Animations[i].PhaseFunction.GetPhase(prevFrame);
					line += FormatValue(phaseChange);

					line = line.Remove(line.Length-1);
					
					line = line.Replace(",",".");

					//Write
					data.WriteLine(line);
				}
			}
		}
		data.Close();
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
