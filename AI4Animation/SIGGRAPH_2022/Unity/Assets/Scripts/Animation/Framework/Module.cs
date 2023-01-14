using UnityEngine;
using System;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public abstract class Module : ScriptableObject {

		#if UNITY_EDITOR
		private static string[] Types = null;
		public static string[] GetTypes() {
			if(Types == null) {
				Types = Utility.GetAllDerivedTypesNames(typeof(Module));
			}
			return Types;
		}

		public MotionAsset Asset;
		public string Tag = string.Empty;
		public bool Precompute = true;
		public bool Callbacks = true;

		//Temporary
		[NonSerialized] public static HashSet<string> Inspect = new HashSet<string>();
		[NonSerialized] public static HashSet<string> Visualize = new HashSet<string>();

		void Awake() {
			// ResetPrecomputation();
		}

		void OnEnable() {
			if(Asset != null) {
				ResetPrecomputation();
			}
		}

		public TimeSeries.Component ExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return DerivedExtractSeries(global, timestamp, mirrored, parameters);
		}

		public Module Initialize(MotionAsset asset, string tag) {
			Asset = asset;
			Tag = tag == null ? string.Empty : tag;
			ResetPrecomputation();
			// Debug.Log("Resetting precomputation in " + Asset.name + " during initialization");
			DerivedInitialize();
			return this;
		}

		public void Load(MotionEditor editor) {
			ResetPrecomputation();
			// Debug.Log("Resetting precomputation in " + Asset.name + " during load");
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
				TimeSeries.Component series = ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror, editor);
				if(series != null) {
					series.GUI();
				}
				DerivedGUI(editor);
			}
		}

		public void Draw(MotionEditor editor) {
			if(Visualize.Contains(GetID())) {
				TimeSeries.Component series = ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror, editor);
				if(series != null) {
					series.Draw();
				}
				DerivedDraw(editor);
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
					if(Utility.GUIButton("Precompute", Precompute ? UltiDraw.Cyan : UltiDraw.LightGrey, UltiDraw.Black, 80f, 20f)) {
						SetPrecomputable(!Precompute);
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
						EditorGUI.BeginChangeCheck();
						DerivedInspector(editor);
						if(EditorGUI.EndChangeCheck()) {
							Asset.ResetPrecomputation();
						}
					}
				}
			}
		}

		public void SetPrecomputable(bool value) {
			if(Precompute != value) {
				Precompute = value;
				ResetPrecomputation();
			}
		}

		public void ResetPrecomputation() {
			DerivedResetPrecomputation();
		}

		public string GetID() {
			return this.GetType().Name + ((Tag == string.Empty) ? string.Empty : (":" + Tag));
		}

		public abstract void DerivedResetPrecomputation();
		public abstract TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters);
		protected abstract void DerivedInitialize();
		protected abstract void DerivedLoad(MotionEditor editor);	
		protected abstract void DerivedUnload(MotionEditor editor);
		protected abstract void DerivedCallback(MotionEditor editor);
		protected abstract void DerivedGUI(MotionEditor editor);
		protected abstract void DerivedDraw(MotionEditor editor);
		protected abstract void DerivedInspector(MotionEditor editor);
		
		//TODO: Precomputables are only created when loading session, but not when retrieving asset alone.
		public class Precomputable<T> {
			private int Padding;
			private int Length;

			public Module Module;
			public Value[] Standard;
			public Value[] Mirrored;

			public class Value {
				public T V;
				public Value(T v) {
					V = v;
				}
			}

			public Precomputable(Module module) {
				Padding = 2*Mathf.RoundToInt(module.Asset.Framerate);
				Length = module.Asset.Frames.Length + 2*Padding;

				Module = module;
				Standard = module.Precompute ? new Value[Length] : null;
				Mirrored = new Value[Length];
			}

			public T Get(float timestamp, bool mirrored, Func<T> function) {
				int index = Mathf.RoundToInt(timestamp * Module.Asset.Framerate) + Padding;
				if(Module.Precompute && index >= 0 && index < Length) {
					if(mirrored && Mirrored[index] == null) {
						Mirrored[index] = new Value(function());
					}
					if(!mirrored && Standard[index] == null) {
						Standard[index] = new Value(function());
					}
					if(mirrored) {
						return Mirrored[index].V;
					}
					if(!mirrored) {
						return Standard[index].V;
					}
				}
				return function();
			}
		}
		#endif

	}
}