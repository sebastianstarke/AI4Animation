using UnityEngine;
using UnityEditor;
using System.IO;

public class BVHProcessor : EditorWindow {

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public string Directory = string.Empty;
	public bool[] Use = new bool[0];
	public BVHAnimation[] Animations = new BVHAnimation[0];

	private static string Separator = " ";
	private static string Accuracy = "F5";

	[MenuItem ("Addons/BVH Processor")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(BVHProcessor));
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
					EditorGUILayout.LabelField("Processor");
				}

				if(Utility.GUIButton("Export Labels", Utility.DarkGrey, Utility.White)) {
					ExportLabels();
				}
				if(Utility.GUIButton("Export Data", Utility.DarkGrey, Utility.White)) {
					ExportData();
				}
				if(Utility.GUIButton("Style Distribution", Utility.DarkGrey, Utility.White)) {
					PrintStyleDistribution();
				}

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Enable All", Utility.Grey, Utility.White)) {
					for(int i=0; i<Use.Length; i++) {
						Use[i] = true;
					}
				}
				if(Utility.GUIButton("Disable All", Utility.Grey, Utility.White)) {
					for(int i=0; i<Use.Length; i++) {
						Use[i] = false;
					}
				}
				EditorGUILayout.EndHorizontal();

				/*
                if(Utility.GUIButton("Fix Data", Utility.DarkGreen, Utility.White)) {
                    for(int i=0; i<Animations.Length; i++) {
						for(int j=0; j<Animations[i].Sequences.Length; j++) {
							Animations[i].Sequences[j].Export = 1;
						}
						//Animations[i].StyleFunction.SetTransition(0.1f);
                        EditorUtility.SetDirty(Animations[i]);
                    }
                    AssetDatabase.SaveAssets();
                    AssetDatabase.Refresh();
                }
				*/
				
				Scroll = EditorGUILayout.BeginScrollView(Scroll);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("Assets/", GUILayout.Width(45f));
					LoadDirectory(EditorGUILayout.TextField(Directory));
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

	private void LoadDirectory(string dir) {
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
		if(!File.Exists(Application.dataPath+"/Project/"+name+".txt")) {
			filename = Application.dataPath+"/Project/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Project/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Project/"+name+" ("+i+")";
		}

		StreamWriter labels = File.CreateText(filename+".txt");
		int index = 0;
		labels.WriteLine(index + " " + "Sequence"); index += 1;
		labels.WriteLine(index + " " + "Frame"); index += 1;
		labels.WriteLine(index + " " + "Timestamp"); index += 1;
		for(int i=1; i<=Animations[0].Character.Hierarchy.Length; i++) {
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "PositionX"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "PositionY"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "PositionZ"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "ForwardX"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "ForwardY"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "ForwardZ"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "UpX"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "UpY"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "UpZ"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "VelocityX"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "VelocityY"+i); index += 1;
			labels.WriteLine(index + " " + Animations[0].Character.Hierarchy[i-1].GetName() + "VelocityZ"+i); index += 1;
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
		if(!File.Exists(Application.dataPath+"/Project/"+name+".txt")) {
			filename = Application.dataPath+"/Project/"+name;
		} else {
			int i = 1;
			while(File.Exists(Application.dataPath+"/Project/"+name+" ("+i+").txt")) {
				i += 1;
			}
			filename = Application.dataPath+"/Project/"+name+" ("+i+")";
		}

		StreamWriter data = File.CreateText(filename+".txt");
		int sequence = 0;
		WriteAnimations(ref data, ref sequence, false);
		WriteAnimations(ref data, ref sequence, true);
		data.Close();
	}

	private void WriteAnimations(ref StreamWriter data, ref int sequence, bool mirrored) {
		for(int i=0; i<Animations.Length; i++) {
			if(Use[i]) {
				for(int s=0; s<Animations[i].Sequences.Length; s++) {
					for(int e=0; e<Animations[i].Sequences[s].Export; e++) {
						sequence += 1;
						//float timeStart = Animations[i].GetFrame(Animations[i].SequenceStart).Timestamp;
						//float timeEnd = Animations[i].GetFrame(Animations[i].SequenceEnd).Timestamp;
						//for(float j=timeStart; j<=timeEnd; j+=1f/60f) {
						int startIndex = Animations[i].Sequences[s].Start;
						int endIndex = Animations[i].Sequences[s].End;
						for(int j=startIndex; j<=endIndex; j++) {
							//Get frame
							BVHAnimation.BVHFrame frame = Animations[i].GetFrame(j);
							//BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1f/60f, 0f, Animations[i].TotalTime));
							BVHAnimation.BVHFrame prevFrame = Animations[i].GetFrame(Mathf.Clamp(j-1, 1, Animations[i].GetTotalFrames()));

							//j = frame.Timestamp;

							//Sequence number
							string line = sequence + Separator;

							//Frame index
							line += frame.Index + Separator;

							//Frame time
							line += frame.Timestamp + Separator;

							//Extract data
							Matrix4x4[] transformations = Animations[i].ExtractTransformations(frame, mirrored);
							Vector3[] velocities = Animations[i].ExtractVelocities(frame, mirrored, 0.1f);
							Trajectory trajectory = Animations[i].ExtractTrajectory(frame, mirrored);
							Trajectory prevTrajectory = Animations[i].ExtractTrajectory(prevFrame, mirrored);
							
							//Get root transformation
							Matrix4x4 root = trajectory.Points[6].GetTransformation();

							//Bone data
							for(int k=0; k<Animations[i].Character.Hierarchy.Length; k++) {
								//Position
								line += FormatVector3(transformations[k].GetPosition().GetRelativePositionTo(root));

								//Rotation
								line += FormatVector3(transformations[k].GetForward().GetRelativeDirectionTo(root));
								line += FormatVector3(transformations[k].GetUp().GetRelativeDirectionTo(root));

								//Velocity
								line += FormatVector3(velocities[k].GetRelativeDirectionTo(root));
							}
							
							//Trajectory data
							for(int k=0; k<12; k++) {
								line += FormatVector3(trajectory.Points[k].GetPosition().GetRelativePositionTo(root));
								line += FormatVector3(trajectory.Points[k].GetDirection().GetRelativeDirectionTo(root));
							}

							for(int k=0; k<12; k++) {
								line += FormatValue(trajectory.Points[k].GetLeftSample().y - root.GetPosition().y);
								line += FormatValue(trajectory.Points[k].GetRightSample().y - root.GetPosition().y);
							}

							for(int k=0; k<12; k++) {
								line += FormatArray(trajectory.Points[k].Styles);
							}

							//Phase
							if(mirrored) {
								line += FormatValue(Animations[i].MirroredPhaseFunction.GetPhase(frame));
							} else {
								line += FormatValue(Animations[i].PhaseFunction.GetPhase(frame));
							}

							//Translational and angular root velocity
							Vector3 position = trajectory.Points[6].GetPosition();
							Vector3 direction = trajectory.Points[6].GetDirection();
							Vector3 prevPosition = prevTrajectory.Points[6].GetPosition();
							Vector3 prevDirection = prevTrajectory.Points[6].GetDirection();

							Vector3 translationOffset = Quaternion.Inverse(Quaternion.LookRotation(prevDirection, Vector3.up)) * (position - prevPosition);
							line += FormatValue(translationOffset.x);
							line += FormatValue(translationOffset.z);

							float rotationOffset = Vector3.SignedAngle(prevDirection, direction, Vector3.up);
							line += FormatValue(rotationOffset);

							//Phase change
							if(mirrored) {
								line += FormatValue(GetPhaseChange(Animations[i].MirroredPhaseFunction.GetPhase(prevFrame), Animations[i].MirroredPhaseFunction.GetPhase(frame)));
							} else {
								line += FormatValue(GetPhaseChange(Animations[i].PhaseFunction.GetPhase(prevFrame), Animations[i].PhaseFunction.GetPhase(frame)));
							}

							//Feet offset function
							//TODO

							//Postprocess
							line = line.Remove(line.Length-1);
							line = line.Replace(",",".");

							//Write
							data.WriteLine(line);
						}
					}
				}
			}
		}
	}

	private void PrintStyleDistribution() {
		if(Animations.Length == 0) {
			return;
		}
		string[] names = new string[Animations[0].StyleFunction.Styles.Length];
		for(int i=0; i<names.Length; i++) {
			names[i] = Animations[0].StyleFunction.Styles[i].Name;
		}
		int[] distribution = new int[Animations[0].StyleFunction.Styles.Length];
		int totalFrames = 0;
		for(int i=0; i<Animations.Length; i++) {
			if(Use[i]) {
				for(int s=0; s<Animations[i].Sequences.Length; s++) {
					for(int e=0; e<Animations[i].Sequences[s].Export; e++) {
						int startIndex = Animations[i].Sequences[s].Start-1;
						int endIndex = Animations[i].Sequences[s].End-1;
						for(int j=startIndex; j<=endIndex; j++) {
							totalFrames += 1;
							for(int d=0; d<distribution.Length; d++) {
								distribution[d] += Animations[i].StyleFunction.Styles[d].Flags[j] ? 1 : 0;
							}
						}
					}
				}
			}
		}
		
		for(int i=0; i<names.Length; i++) {
			Debug.Log("Name: " + names[i] + " Count: " + distribution[i] + " Ratio: " + (float)distribution[i] / (float)totalFrames);
		}
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

	private string FormatArray(bool[] array) {
		string format = string.Empty;
		for(int i=0; i<array.Length; i++) {
			float value = array[i] ? 1f : 0f;
			format += value.ToString(Accuracy) + Separator;
		}
		return format;
	}

	private string FormatVector3(Vector3 vector) {
		return vector.x.ToString(Accuracy) + Separator + vector.y.ToString(Accuracy) + Separator + vector.z.ToString(Accuracy) + Separator;
	}

	private string FormatQuaternion(Quaternion quaternion, bool imaginary, bool real) {
		string output = string.Empty;
		if(imaginary) {
			output += quaternion.x + Separator + quaternion.y + Separator + quaternion.z + Separator;
		}
		if(real) {
			output += quaternion.w + Separator;
		}
		return output;
	}

}
