using UnityEngine;
using System;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public abstract class Module : ScriptableObject {
		private static string[] Types = null;
		public static string[] GetTypes() {
			if(Types == null) {
				Types = Utility.GetAllDerivedTypesNames(typeof(Module));
			}
			return Types;
		}

		public MotionAsset Asset;
		public string Tag = string.Empty;

		//Temporary
		[NonSerialized] public bool Callbacks = true;
		[NonSerialized] public static HashSet<string> Inspect = new HashSet<string>();
		[NonSerialized] public static HashSet<string> Visualize = new HashSet<string>();

		void Awake() {

		}

		void OnEnable() {

		}

		public TimeSeries.Component ExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return DerivedExtractSeries(global, timestamp, mirrored, parameters);
		}
#if UNITY_EDITOR
		public Module Initialize(MotionAsset asset, string tag) {
			Asset = asset;
			Tag = tag == null ? string.Empty : tag;
			DerivedInitialize();
			return this;
		}

		public void Load(MotionEditor editor) {
			DerivedLoad(editor);
		}

		public void Unload(MotionEditor editor) {
			DerivedUnload(editor);
		}

		public virtual void OnTriggerPlay(MotionEditor editor) {

		}

		public void Callback(MotionEditor editor) {
			if(Callbacks) {
				DerivedCallback(editor);
			}
		}

		public void GUI(MotionEditor editor) {
			if(Visualize.Contains(GetID())) {
				TimeSeries.Component series = ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror);
				if(series != null) {
					series.GUI();
				}
				DerivedGUI(editor);
			}
		}

		public void Draw(MotionEditor editor) {
			if(Visualize.Contains(GetID())) {
				TimeSeries.Component series = ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror);
				if(series != null) {
					series.Draw();
				}
				DerivedDraw(editor);
			}
		}

		public void SetCallbacks(MotionEditor editor, bool value) {
			if(Callbacks != value) {
				Callbacks = value;
				editor.LoadFrame(editor.GetTimestamp());
			}
		}

		public void Inspector(MotionEditor editor) {
			Utility.SetGUIColor(UltiDraw.DarkGrey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Mustard);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.BeginHorizontal();
					if(EditorGUILayout.Toggle(Inspect.Contains(GetID()), GUILayout.Width(20f))) {
						Inspect.Add(GetID());
					} else {
						Inspect.Remove(GetID());
					}
					EditorGUILayout.LabelField(GetID().ToString());
					GUILayout.FlexibleSpace();
					if(Utility.GUIButton("Visualize", Visualize.Contains(GetID()) ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 80f, 20f)) {
						if(Visualize.Contains(GetID())) {
							Visualize.Remove(GetID());
						} else {
							Visualize.Add(GetID());
						}
					}
					if(Utility.GUIButton("Callbacks", Callbacks ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 80f, 20f)) {
						Callbacks = !Callbacks;
					}
					if(Utility.GUIButton("X", UltiDraw.DarkRed, UltiDraw.White, 25f, 20f)) {
						Asset.RemoveModule(this);
					}
					EditorGUILayout.EndHorizontal();
				}

				if(Inspect.Contains(GetID())) {
					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						Tag = EditorGUILayout.TextField("Tag", Tag);
						DerivedInspector(editor);
					}
				}
			}
		}
#endif
		public string GetID() {
			return this.GetType().Name + ((Tag == string.Empty) ? string.Empty : (":" + Tag));
		}

		public abstract TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters);

#if UNITY_EDITOR
		protected abstract void DerivedInitialize();
		protected abstract void DerivedLoad(MotionEditor editor);
		protected abstract void DerivedUnload(MotionEditor editor);
		protected abstract void DerivedCallback(MotionEditor editor);
		protected abstract void DerivedGUI(MotionEditor editor);
		protected abstract void DerivedDraw(MotionEditor editor);
		protected abstract void DerivedInspector(MotionEditor editor);
#endif
	}
}