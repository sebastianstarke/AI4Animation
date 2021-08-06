#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using System.Collections.Generic;

public class MotionProcessor : EditorWindow {

	public enum PIPELINE {Basketball, Quadruped};

	[System.Serializable]
	public class Asset {
		public string GUID = string.Empty;
		public bool Selected = true;
		public bool Processed = false;
	}

	public static EditorWindow Window;
	public static Vector2 Scroll;

	public PIPELINE Pipeline = PIPELINE.Quadruped; 

	public bool OfflineProcessing = false;
	public bool SaveAfterProcessing = true;

	private string Filter = string.Empty;
	private Asset[] Assets = new Asset[0];
	[NonSerialized] private Asset[] Instances = null;

	private static bool Processing = false;

	[NonSerialized] private Sequence Sequence;
	private int Page = 0;
	private int Items = 25;

	private MotionEditor Editor = null;

	[MenuItem ("AI4Animation/Tools/Motion Processor")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MotionProcessor));
		Scroll = Vector3.zero;
	}

	public void OnInspectorUpdate() {
		Repaint();
	}
	
	public void Refresh() {
		if(Editor == null) {
			Editor = GameObject.FindObjectOfType<MotionEditor>();
		}
		if(Editor != null && Assets.Length != Editor.Assets.Length) {
			Assets = new Asset[Editor.Assets.Length];
			for(int i=0; i<Editor.Assets.Length; i++) {
				Assets[i] = new Asset();
				Assets[i].GUID = Editor.Assets[i];
				Assets[i].Selected = true;
				Assets[i].Processed = false;
			}
			Processing = false;
			ApplyFilter(string.Empty);
			LoadPage(1);
		}
		if(Instances == null) {
			ApplyFilter(string.Empty);
			LoadPage(1);
		}
		if(Sequence == null) {
			Sequence = new Sequence(1, Assets.Length);
		}
	}

	public void ApplySequence() {
		for(int i=0; i<Assets.Length; i++) {
			Assets[i].Selected = Sequence.Contains(i+1);
		}
	}

	public void ApplyFilter(string filter) {
		Filter = filter;
		if(Filter == string.Empty) {
			Instances = Assets;
		} else {
			List<Asset> instances = new List<Asset>();
			for(int i=0; i<Assets.Length; i++) {
				if(Utility.GetAssetName(Assets[i].GUID).ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
					instances.Add(Assets[i]);
				}
			}
			Instances = instances.ToArray();
		}
		LoadPage(1);
	}

	public void LoadPage(int page) {
		Page = Mathf.Clamp(page, 1, GetPages());
		int start = GetStart();
		int end = GetEnd();
	}

	public int GetPages() {
		return Mathf.CeilToInt(Instances.Length/Items)+1;
	}

	public int GetStart() {
		return (Page-1)*Items;
	}

	public int GetEnd() {
		return Mathf.Min(Page*Items, Instances.Length);
	}

	void OnGUI() {
		Refresh();

		if(Editor == null) {
			EditorGUILayout.LabelField("No 'Motion Editor' component available in the loaded scene.");
			return;
		}

		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Motion Processor");
				}

				Pipeline = (PIPELINE)EditorGUILayout.EnumPopup("Pipeline", Pipeline);
				OfflineProcessing = EditorGUILayout.Toggle("Offline Processing", OfflineProcessing);
				SaveAfterProcessing = EditorGUILayout.Toggle("Save After Processing", SaveAfterProcessing);

				EditorGUILayout.BeginHorizontal();
				if(Utility.GUIButton("Current", UltiDraw.DarkGrey, UltiDraw.White)) {
					int index = Editor.GetAssetIndex() + 1;
					Sequence.Start = index;
					Sequence.End = index;
					ApplySequence();
				}
				if(Utility.GUIButton("All", UltiDraw.DarkGrey, UltiDraw.White)) {
					Sequence.Start = 1;
					Sequence.End = Assets.Length;
					ApplySequence();
				}
				EditorGUILayout.EndHorizontal();

				if(Sequence != null) {
					if(Sequence.Inspector(1, Assets.Length)) {
						ApplySequence();
					}
				}

				if(Processing) {
					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
							this.StopAllCoroutines();
							Processing = false;
						}
					}
					EditorGUI.DrawRect(new Rect(EditorGUILayout.GetControlRect().x, EditorGUILayout.GetControlRect().y, (float)(Editor.GetAssetIndex()+1) / (float)Assets.Length * EditorGUILayout.GetControlRect().width, 25f), UltiDraw.Green.Opacity(0.75f));
				} else {
					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
							this.StartCoroutine(Process());
						}
					}
				}

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();

					EditorGUILayout.LabelField("Page", GUILayout.Width(40f));
					EditorGUI.BeginChangeCheck();
					int page = EditorGUILayout.IntField(Page, GUILayout.Width(40f));
					if(EditorGUI.EndChangeCheck()) {
						LoadPage(page);
					}
					EditorGUILayout.LabelField("/" + GetPages());
					
					EditorGUILayout.LabelField("Filter", GUILayout.Width(40f));
					EditorGUI.BeginChangeCheck();
					string filter = EditorGUILayout.TextField(Filter, GUILayout.Width(200f));
					if(EditorGUI.EndChangeCheck()) {
						ApplyFilter(filter);
					}

					if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
						LoadPage(Mathf.Max(Page-1, 1));
					}
					if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
						LoadPage(Mathf.Min(Page+1, GetPages()));
					}
					EditorGUILayout.EndHorizontal();
				}
				
				int start = GetStart();
				int end = GetEnd();
				for(int i=start; i<end; i++) {
						if(Instances[i].Processed) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else if(Instances[i].Selected) {
							Utility.SetGUIColor(UltiDraw.Gold);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
						Instances[i].Selected = EditorGUILayout.Toggle(Instances[i].Selected, GUILayout.Width(20f));
						EditorGUILayout.LabelField(Utility.GetAssetName(Instances[i].GUID));
						EditorGUILayout.EndHorizontal();
					}
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

	private IEnumerator Process() {
		if(Editor != null) {
			Processing = true;
			foreach(Asset a in Assets) {
				a.Processed = false;
			}
			int count = 0;

			if(Pipeline == PIPELINE.Basketball) {
				string referenceBone = string.Empty;
				MotionData referenceData = null;
				foreach(Asset a in Assets) {
					if(a.Selected) {
						count += 1;
						MotionData data = OfflineProcessing ? Editor.GetAsset(a.GUID) : Editor.LoadData(a.GUID);
						while(!OfflineProcessing && !data.GetScene().isLoaded) {
							Debug.Log("Waiting for scene being loaded...");
							yield return new WaitForSeconds(0f);
						}
						//START OF PROCESSING

						// Reset motion data
						foreach(Frame frame in data.Frames) {
							frame.ResetTransformations();
						}

						// Global
						data.RemoveAllModules();
						data.MirrorAxis = Axis.ZPositive;

						// 1on1 Ball Copying
						{
							string GetID(MotionData asset) {
								return asset.GetName().Substring(0, asset.GetName().LastIndexOf("_P0"));
							}
							referenceBone = "Player 01:Ball";
							if(data.GetName().Contains("_P0")) {
								if(data.GetName().Contains("_P01")) {
									referenceData = data;
								} else if(GetID(data) == GetID(referenceData)) {
									data.SampleTransformations(referenceData, referenceBone);
								} else {
									Debug.LogError("Skipping asset " + data.GetName() + " as information of P01 is not of same capture.");
								}
							} else {
								Debug.LogError("Skipping asset " + data.GetName() + " as it does not contain player information.");
							}
						}

						{
							RootModule root = data.AddModule<RootModule>();
							root.Root = data.Source.FindBone("Player 01:Hips").Index;
							root.RightShoulder = data.Source.FindBone("Player 01:LeftShoulder").Index;
							root.LeftShoulder = data.Source.FindBone("Player 01:RightShoulder").Index;
							root.RightHip = data.Source.FindBone("Player 01:LeftUpLeg").Index;
							root.LeftHip = data.Source.FindBone("Player 01:RightUpLeg").Index;
							root.Neck = data.Source.FindBone("Player 01:Neck").Index;
							root.Hips = data.Source.FindBone("Player 01:Hips").Index;
							root.Smooth = true;
							root.Topology = RootModule.TOPOLOGY.Biped;
							root.ForwardAxis = Axis.XPositive;
						}

						{
							ContactModule contact = data.AddModule<ContactModule>();
							contact.Clear();
							contact.AddSensor("Player 01:LeftFootEnd", "Player 01:LeftFootEnd", Vector3.zero, 0.075f, 1f, -1);
							contact.AddSensor("Player 01:RightFootEnd", "Player 01:RightFootEnd", Vector3.zero, 0.075f, 1f, -1);
							contact.AddSensor("Player 01:LeftHand", "Player 01:LeftHand", new Vector3(-0.1f, 0f, 0f), 0.075f, 0f, -1);
							contact.AddSensor("Player 01:RightHand", "Player 01:RightHand", new Vector3(-0.1f, 0f, 0f), 0.075f, 0f, -1);
							contact.AddSensor("Player 01:Ball", "Player 01:Ball", Vector3.zero, 0.2f, 0f, LayerMask.GetMask("Ground"));
							contact.CaptureContacts(Editor);
						}

						{
							DribbleModule dribble = data.AddModule<DribbleModule>();
							dribble.Area = 2.5f;
							dribble.Radius = 0.125f;
							dribble.Axis = Axis.YPositive;
							Matrix4x4[] motion = dribble.CleanupBallTransformations(false);
							for(int i=0; i<data.Frames.Length; i++) {
								data.Frames[i].Transformations[dribble.Ball] = motion[i];
							}
							data.GetModule<ContactModule>().CaptureContacts(Editor);
						}

						{
							StyleModule style = data.AddModule<StyleModule>();
							RootModule root = data.GetModule<RootModule>();
							DribbleModule dribble = data.GetModule<DribbleModule>();
							style.Clear();
							StyleModule.StyleFunction standing = style.AddStyle("Stand");
							StyleModule.StyleFunction moving = style.AddStyle("Move");
							StyleModule.StyleFunction dribbling = style.AddStyle("Dribble");
							StyleModule.StyleFunction holding = style.AddStyle("Hold");
							StyleModule.StyleFunction shooting = style.AddStyle("Shoot");
							float threshold = 1f;
							for(int f=0; f<data.Frames.Length; f++) {
								float[] timestamps = data.SimulateTimestamps(data.Frames[f], 30);
								float[] weights = new float[timestamps.Length];
								for(int j=0; j<timestamps.Length; j++) {
									weights[j] = Mathf.Clamp(root.GetRootVelocity(timestamps[j], false).magnitude, 0f, threshold);
									weights[j] = weights[j].Normalize(0f, threshold, 0, 1f);
									weights[j] = weights[j].SmoothStep(2f, 0.5f);
								}
								float weight = weights.Gaussian().SmoothStep(2f, 0.5f);
								standing.Values[f] = 1f - weight;
								moving.Values[f] = weight;
								dribbling.Values[f] = dribble.IsDribbling(data.Frames[f].Timestamp, false) ? 1f : 0f;
								holding.Values[f] = dribble.IsHolding(data.Frames[f].Timestamp, false) ? 1f : 0f;
								shooting.Values[f] = dribble.IsShooting(data.Frames[f].Timestamp, false) ? 1f : 0f;
							}
							style.Mode = StyleModule.DRAWING.Frames;
							style.GenerateKeys();
						}

						{
							PhaseModule phase = data.AddModule<PhaseModule>();
							ContactModule contact = data.GetModule<ContactModule>();
							phase.Inspect = true;
							phase.SetFunctions(contact.GetNames());
							phase.ShowNormalized = true;
							phase.ShowHighlighted = true;
							phase.ShowValues = true;
							phase.ShowFitting = true;
							phase.ShowZero = true;
							phase.ShowPhase = true;
							phase.ShowWindow = false;
							phase.DisplayValues = false;
							phase.MaxIterations = 50;
							phase.Individuals = 50;
							phase.Elites = 5;
							phase.Exploration = 0.2f;
							phase.Memetism = 0.1f;
							phase.MaxFrequency = 4f;
							phase.RescalingMethod = PhaseModule.Rescaling.Window;
							phase.ApplyButterworth = true;
							phase.StartFitting();
							while(phase.IsFitting())
							{
								yield return new WaitForSeconds(0f);
							}
						}
						
						//END OF PROCESSING
						data.MarkDirty();
						a.Processed = true;
					}
				}
				foreach(Asset a in Assets) {
					if(a.Selected) {
						MotionData data = Editor.LoadData(a.GUID);
						while(!data.GetScene().isLoaded) {
							Debug.Log("Waiting for scene being loaded...");
							yield return new WaitForSeconds(0f);
						}
						//START OF PROCESSING

						data.GetModule<DribbleModule>().ComputeInteraction();

						//END OF PROCESSING
						if(SaveAfterProcessing) {
							data.MarkDirty(true, !OfflineProcessing);
						}
						a.Processed = true;
						yield return new WaitForSeconds(0f);
					}
				}
				if(SaveAfterProcessing) {
					AssetDatabase.SaveAssets();
					AssetDatabase.Refresh();
				}
			}

			if(Pipeline == PIPELINE.Quadruped) {

				foreach(Asset a in Assets) {
					if(a.Selected) {
						count += 1;
						MotionData data = OfflineProcessing ? Editor.GetAsset(a.GUID) : Editor.LoadData(a.GUID);
						while(!OfflineProcessing && !data.GetScene().isLoaded) {
							Debug.Log("Waiting for scene being loaded...");
							yield return new WaitForSeconds(0f);
						}
						// START OF PROCESSING

						// Reset motion data
						foreach(Frame frame in data.Frames) {
							frame.ResetTransformations();
						}

						// Global
						data.RemoveAllModules();
						data.Scale = 0.01f;
						data.MirrorAxis = Axis.ZPositive;
						data.Source.FindBone("Head").Alignment = new Vector3(90f, 0f, 0f);
						data.Source.FindBone("Tail").Alignment = new Vector3(-45f, 0f, 0f);
						data.Source.FindBone("Tail1").Alignment = new Vector3(-45f, 0f, 0f);
						data.Source.FindBone("Tail1Site").Alignment = new Vector3(-45f, 0f, 0f);

						{
							ContactModule contact = data.AddModule<ContactModule>();
							contact.Clear();
							contact.AddSensor("Hips", "Hips", Vector3.zero, 0.2f, 1f, LayerMask.GetMask("Ground"));
							contact.AddSensor("Neck", "Neck", Vector3.zero, 0.25f, 1f, LayerMask.GetMask("Ground"));
							contact.AddSensor("LeftHandSite", new string[]{"LeftForeArm", "LeftHandSite"}, Vector3.zero, 1f/30f, 1f, LayerMask.GetMask("Ground"));
							contact.AddSensor("RightHandSite", new string[]{"RightForeArm", "RightHandSite"}, Vector3.zero, 1f/30f, 1f, LayerMask.GetMask("Ground"));
							contact.AddSensor("LeftFootSite", "LeftFootSite",  Vector3.zero, 1f/30f, 1f, LayerMask.GetMask("Ground"));
							contact.AddSensor("RightFootSite", "RightFootSite",  Vector3.zero, 1f/30f, 1f, LayerMask.GetMask("Ground"));
							contact.CaptureContacts(Editor);
						}

						{
							StyleModule style = data.AddModule<StyleModule>();
							RootModule root = data.AddModule<RootModule>();
							root.Topology = RootModule.TOPOLOGY.Quadruped;
							ContactModule contact = data.GetModule<ContactModule>();
							style.Clear();
							StyleModule.StyleFunction idling = style.AddStyle("Idle");
							StyleModule.StyleFunction moving = style.AddStyle("Move");
							StyleModule.StyleFunction sitting = style.AddStyle("Sit");
							StyleModule.StyleFunction resting = style.AddStyle("Rest");
							StyleModule.StyleFunction standing = style.AddStyle("Stand");
							StyleModule.StyleFunction jumping = style.AddStyle("Jump");
							StyleModule.StyleFunction speed = style.AddStyle("Speed");
							float[] timeWindow = data.GetTimeWindow(MotionEditor.GetInstance().PastWindow + MotionEditor.GetInstance().FutureWindow, 1f);
							float[] contactHeights = new float[data.Frames.Length];
							List<float[]> sitPatterns = new List<float[]>(){
								new float[]{1f, 0f, 1f, 1f, 1f, 1f},
								new float[]{1f, 0f, 0f, 1f, 1f, 1f},
								new float[]{1f, 0f, 1f, 0f, 1f, 1f}
							};
							List<float[]> restPatterns = new List<float[]>(){
								new float[]{1f, 1f, 1f, 1f, 1f, 1f}
							};
							List<float[]> standPatterns = new List<float[]>(){
								new float[]{1f, 0f, 0f, 0f, 1f, 1f}
							};
							List<float[]> jumpPatterns = new List<float[]>(){
								new float[]{0f, 0f, 0f, 0f, 0f, 0f}
							};
							for(int i=0; i<data.Frames.Length; i++) {
								for(int j=0; j<contact.Sensors.Length; j++) {
									contactHeights[i] += data.Frames[i].GetBoneTransformation(contact.Sensors[j].Bones.Last(), false).GetPosition().y;
								}
								contactHeights[i] /= contact.Sensors.Length;
							}
							for(int f=0; f<data.Frames.Length; f++) {
								float weight = GetMovementWeight(data.Frames[f].Timestamp, 0.5f, 0.5f);
								idling.Values[f] = 1f - weight;
								moving.Values[f] = weight;
								float sit = GetContactsWeight(data.Frames[f].Timestamp, 0.5f, contact, sitPatterns, 0f, 1f);
								float rest = GetContactsWeight(data.Frames[f].Timestamp, 0.5f, contact, restPatterns, 0f, 1f);
								float stand = GetContactsWeight(data.Frames[f].Timestamp, 0.5f, contact, standPatterns, 0f, 1f);
								float jump = GetContactsWeight(data.Frames[f].Timestamp, 0.5f, contact, jumpPatterns, 0.3f, 0.1f);
								float[] actions = new float[]{sit, rest, stand, jump};
								Utility.SoftMax(ref actions);
								sitting.Values[f] = sit;
								resting.Values[f] = rest;
								standing.Values[f] = stand;
								jumping.Values[f] = jump;
								speed.Values[f] = GetRootSpeed(data.Frames[f].Timestamp);
							}

							float GetRootSpeed(float timestamp) {
								return Compute();
								float Compute() {
									Vector3[] positions = new Vector3[timeWindow.Length];
									for(int i=0; i<timeWindow.Length; i++) {
										positions[i] = root.GetRootPosition(timestamp + timeWindow[i], false);
									}
									float length = 0f;
									for(int i=1; i<positions.Length; i++) {
										length += Vector3.Distance(positions[i-1], positions[i]);
									}
									return length / (timeWindow.Last() - timeWindow.First());
								}
							}

							float GetMovementWeight(float timestamp, float window, float threshold) {
								float[] weights = new float[timeWindow.Length];
								for(int j=0; j<timeWindow.Length; j++) {
									weights[j] = Mathf.Clamp(root.GetRootVelocity(timestamp + timeWindow[j], false).magnitude, 0f, threshold) / threshold;
								}
								
								float[] gradients = new float[weights.Length-1];
								for(int i=0; i<gradients.Length; i++) {
									gradients[i] = (weights[i+1] - weights[i]) / (timeWindow[i+1] - timeWindow[i]);
								}
								float gradient = Mathf.Abs(gradients.Gaussian());
								
								return weights.Gaussian(gradient).SmoothStep(2f, 0.5f);
							}

							float GetContactsWeight(float timestamp, float window, ContactModule module, List<float[]> patterns, float heightThreshold, float power) {
								float ContactGaussian(float t) {
									float[] weights = new float[timeWindow.Length];
									for(int j=0; j<timeWindow.Length; j++) {
										bool match = false;
										for(int i=0; i<patterns.Count; i++) {
											float[] contacts = module.GetContacts(t + timeWindow[j], false);
											match = ArrayExtensions.Equal(contacts, patterns[i]).All(true);
											if(match) {
												break;
											}
										}
										if(match && heightThreshold != 0f && contactHeights[data.GetFrame(t).Index-1] < heightThreshold) {
											match = false;
										}
										weights[j] = match ? 1f : 0f;
									}
									return weights.Gaussian();
								}
								float weight = ContactGaussian(timestamp);
								weight = Mathf.Pow(weight, 1f-weight);
								return Mathf.Pow(weight, power);
							}

							style.Mode = StyleModule.DRAWING.Frames;
						}

						{
							PhaseModule phase = data.AddModule<PhaseModule>();
							phase.Inspect = true;
							RootModule root = data.GetModule<RootModule>();
							ContactModule contact = data.GetModule<ContactModule>();
							phase.SetFunctions(contact.GetNames());
							phase.ShowNormalized = true;
							phase.ShowHighlighted = true;
							phase.ShowValues = true;
							phase.ShowFitting = true;
							phase.ShowZero = true;
							phase.ShowPhase = true;
							phase.ShowWindow = false;
							phase.DisplayValues = false;
							phase.MaxIterations = 50;
							phase.Individuals = 100;
							phase.Elites = 10;
							phase.Exploration = 0.2f;
							phase.Memetism = 0.1f;
							phase.MaxFrequency = 4f;
							phase.RescalingMethod = PhaseModule.Rescaling.Window;
							phase.ApplyButterworth = true;

							phase.StartFitting();
							while(phase.IsFitting())
							{
								yield return new WaitForSeconds(0f);
							}
						}

						//END OF PROCESSING
						if(SaveAfterProcessing) {
							data.MarkDirty(true, !OfflineProcessing);
						}
						a.Processed = true;
						yield return new WaitForSeconds(0f);
					}
				}

				for(int i=0; i<Editor.Assets.Length; i++) {
					Editor.GetAsset(i).ResetSequences();
					Editor.GetAsset(i).Export = false;
				}
				Editor.GetAsset(0).Export = true;
				Editor.GetAsset(0).SetSequence(0, 180, 1531);
				Editor.GetAsset(2).Export = true;
				Editor.GetAsset(2).SetSequence(0, 680, 820);
				Editor.GetAsset(6).Export = true;
				Editor.GetAsset(6).SetSequence(0, 90, 593);
				Editor.GetAsset(7).Export = true;
				Editor.GetAsset(7).SetSequence(0, 290, 1072);
				Editor.GetAsset(8).Export = true;
				Editor.GetAsset(8).SetSequence(0, 1, 50);
				Editor.GetAsset(8).SetSequence(1, 400, 911);
				Editor.GetAsset(9).Export = true;
				Editor.GetAsset(10).Export = true;
				Editor.GetAsset(10).SetSequence(0, 230, 548);
				Editor.GetAsset(11).Export = true;
				Editor.GetAsset(11).SetSequence(0, 400, 567);
				Editor.GetAsset(12).Export = true;
				Editor.GetAsset(13).Export = true;
				Editor.GetAsset(14).Export = true;
				Editor.GetAsset(16).Export = true;
				Editor.GetAsset(16).SetSequence(0, 200, 550);
				Editor.GetAsset(17).Export = true;
				Editor.GetAsset(17).SetSequence(0, 470, 720);
				Editor.GetAsset(18).Export = true;
				Editor.GetAsset(18).SetSequence(0, 175, 395);
				Editor.GetAsset(19).Export = true;
				Editor.GetAsset(19).SetSequence(0, 300, 750);
				Editor.GetAsset(19).SetSequence(1, 1040, 1079);
				Editor.GetAsset(20).Export = true;
				Editor.GetAsset(21).Export = true;
				Editor.GetAsset(21).SetSequence(0, 1, 1300);
				Editor.GetAsset(21).SetSequence(1, 2950, 3530);
				Editor.GetAsset(21).SetSequence(2, 3730, 4200);
				Editor.GetAsset(22).Export = true;
				Editor.GetAsset(23).Export = true;
				Editor.GetAsset(23).Export = true;
				Editor.GetAsset(24).Export = true;
				Editor.GetAsset(24).SetSequence(0, 200, 630);
				Editor.GetAsset(25).Export = true;
				Editor.GetAsset(25).SetSequence(0, 1, 2690);
				Editor.GetAsset(25).SetSequence(1, 2760, 4336);
				Editor.GetAsset(26).Export = true;
				Editor.GetAsset(27).Export = true;
				Editor.GetAsset(27).SetSequence(0, 1, 1100);
				Editor.GetAsset(27).SetSequence(1, 2820, 3940);
				Editor.GetAsset(27).SetSequence(2, 4100, 4500);
				Editor.GetAsset(27).SetSequence(3, 5660, 6010);
				Editor.GetAsset(27).SetSequence(4, 6600, 7200);
				Editor.GetAsset(27).SetSequence(5, 12300, 12850);
				Editor.GetAsset(27).SetSequence(6, 13200, 13399);
				Editor.GetAsset(28).Export = true;
				Editor.GetAsset(28).SetSequence(0, 920, 985);
				Editor.GetAsset(28).SetSequence(1, 1700, 1907);
				Editor.GetAsset(29).Export = true;
				Editor.GetAsset(29).SetSequence(0, 250, 790);
				Editor.GetAsset(29).SetSequence(1, 970, 1575);
				Editor.GetAsset(29).SetSequence(2, 1630, 1750);
				Editor.GetAsset(30).Export = true;
				Editor.GetAsset(30).SetSequence(0, 1790, 1920);
				Editor.GetAsset(30).SetSequence(1, 2070, 2470);
				Editor.GetAsset(30).SetSequence(2, 2770, 3025);
				Editor.GetAsset(31).Export = true;
				Editor.GetAsset(31).SetSequence(0, 170, 500);
				Editor.GetAsset(31).SetSequence(1, 1250, 2460);
				Editor.GetAsset(31).SetSequence(2, 3040, 3200);
				Editor.GetAsset(31).SetSequence(3, 4680, 6550);
				Editor.GetAsset(31).SetSequence(4, 7600, 9450);
				Editor.GetAsset(31).SetSequence(5, 11540, 11691);
				Editor.GetAsset(32).Export = true;
				Editor.GetAsset(32).SetSequence(0, 1, 300);
				Editor.GetAsset(32).SetSequence(1, 1360, 1540);
				Editor.GetAsset(32).SetSequence(2, 2380, 3086);
				Editor.GetAsset(33).Export = true;
				Editor.GetAsset(33).SetSequence(0, 1, 1170);
				Editor.GetAsset(33).SetSequence(1, 1980, 2160);
				Editor.GetAsset(33).SetSequence(2, 7830, 8090);
				Editor.GetAsset(34).Export = true;
				Editor.GetAsset(34).SetSequence(0, 1, 270);
				Editor.GetAsset(34).SetSequence(1, 2490, 2856);
				Editor.GetAsset(35).Export = true;
				Editor.GetAsset(37).Export = true;
				Editor.GetAsset(38).Export = true;
				Editor.GetAsset(38).SetSequence(0, 3330, 3900);
				Editor.GetAsset(39).Export = true;
				Editor.GetAsset(39).SetSequence(0, 880, 920);
				Editor.GetAsset(39).SetSequence(1, 1280, 5052);
				Editor.GetAsset(41).Export = true;
				Editor.GetAsset(41).SetSequence(0, 4690, 6190);
				Editor.GetAsset(42).Export = true;
				Editor.GetAsset(42).SetSequence(0, 900, 3594);
				Editor.GetAsset(43).Export = true;
				Editor.GetAsset(43).SetSequence(0, 1, 500);
				Editor.GetAsset(43).SetSequence(1, 4340, 4577);
				Editor.GetAsset(44).Export = true;
				Editor.GetAsset(44).SetSequence(0, 1, 700);
				Editor.GetAsset(44).SetSequence(1, 950, 2000);
				Editor.GetAsset(45).Export = true;
				Editor.GetAsset(45).SetSequence(0, 1, 410);
				Editor.GetAsset(45).SetSequence(1, 680, 778);
				Editor.GetAsset(46).Export = true;
				Editor.GetAsset(46).SetSequence(0, 175, 235);
				Editor.GetAsset(47).Export = true;
				Editor.GetAsset(47).SetSequence(0, 275, 498);
				Editor.GetAsset(48).Export = true;
				Editor.GetAsset(48).SetSequence(0, 1, 220);
				Editor.GetAsset(48).SetSequence(1, 675, 748);
				Editor.GetAsset(49).Export = true;
				Editor.GetAsset(49).SetSequence(0, 1, 700);
				Editor.GetAsset(49).SetSequence(1, 1510, 8300);
				Editor.GetAsset(50).Export = true;
				Editor.GetAsset(50).SetSequence(0, 200, 1000);
				Editor.GetAsset(50).SetSequence(1, 1850, 2100);
				Editor.GetAsset(50).SetSequence(2, 4150, 4700);
				Editor.GetAsset(50).SetSequence(3, 5030, 5356);

				//Mark for saving
				for(int i=0; i<Editor.Assets.Length; i++) {
					Editor.GetAsset(i).MarkDirty(true, false);
				}
				
				if(SaveAfterProcessing) {
					AssetDatabase.SaveAssets();
					AssetDatabase.Refresh();
				}
			}

			Processing = false;
			foreach(Asset a in Assets) {
				a.Processed = false;
			}
			yield return new WaitForSeconds(0f);

			Debug.Log("Finished processing " + count + " assets.");
		}
	}

}
#endif