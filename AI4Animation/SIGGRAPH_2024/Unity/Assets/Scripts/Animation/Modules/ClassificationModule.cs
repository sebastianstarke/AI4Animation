using UnityEngine;
using System;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
	public class ClassificationModule : Module {
		public float Threshold = 0.1f;
        public int Discretization = 4;

		public Classification[] Classifications = new Classification[0];

		[Serializable]
		public class Classification {
			public float[] Standard;
			public float[] Mirrored;
		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			Series series = new Series(global, Discretization);
			for(int i=0; i<series.Samples.Length; i++) {
				series.Classifications[i] = GetClassification(timestamp + series.Samples[i].Timestamp, mirrored);
			}
			return series;
		}

#if UNITY_EDITOR
		protected override void DerivedInitialize() {
			Classifications = new Classification[Asset.Frames.Length];
			for(int i=0; i<Asset.Frames.Length; i++) {
				Classifications[i] = new Classification();	
				Classifications[i].Standard = new float[Discretization];
				Classifications[i].Mirrored = new float[Discretization];
			}
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
			Series series = ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror) as Series;
			series.Draw();
			series.DrawClassificationPatches(editor.GetSession().GetActor().GetRoot().GetWorldMatrix());
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Threshold = EditorGUILayout.FloatField("Threshold", Threshold);
            Discretization = EditorGUILayout.IntField("Discretization", Discretization);
			if(Utility.GUIButton("Comptue", UltiDraw.DarkGrey, UltiDraw.White)) {
				Compute(editor.GetTimeSeries());
			}

			if(Classifications.Length == Asset.Frames.Length) {
				for(int k=0; k<Discretization; k++) {
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
					if(endTime > Asset.GetTotalTime()) {
						startTime -= endTime-Asset.GetTotalTime();
						endTime = Asset.GetTotalTime();
					}
					startTime = Mathf.Max(0f, startTime);
					endTime = Mathf.Min(Asset.GetTotalTime(), endTime);
					int start = Asset.GetFrame(startTime).Index;
					int end = Asset.GetFrame(endTime).Index;
					int elements = end-start;

					Vector3 bottom = new Vector3(0f, rect.yMax, 0f);
					Vector3 top = new Vector3(0f, rect.yMax - rect.height, 0f);

					start = Mathf.Clamp(start, 1, Asset.Frames.Length);
					end = Mathf.Clamp(end, 1, Asset.Frames.Length);

					UltiDraw.Begin();
					for(int i=start; i<=end; i++) {
						float[] classification = GetClassification(Asset.GetFrame(i).Timestamp, editor.Mirror);
						if(classification.Length == Discretization && classification[k] == 1f) {
							float left = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							float right = left;
							while(i<end && GetClassification(Asset.GetFrame(i).Timestamp, editor.Mirror)[k] == 1f) {
								i++;
								right = rect.xMin + (float)(i-start)/(float)elements * rect.width;
							}
							if(left != right) {
								Vector3 a = new Vector3(left, rect.y, 0f);
								Vector3 b = new Vector3(right, rect.y, 0f);
								Vector3 c = new Vector3(left, rect.y+rect.height, 0f);
								Vector3 d = new Vector3(right, rect.y+rect.height, 0f);
								UltiDraw.DrawTriangle(a, c, b, UltiDraw.GetRainbowColor(k, Discretization));
								UltiDraw.DrawTriangle(b, c, d, UltiDraw.GetRainbowColor(k, Discretization));
							}
						}
					}
					UltiDraw.End();

					editor.DrawPivot(rect);

					EditorGUILayout.EndVertical();
				}
			}
		}
#endif
		public float[] GetClassification(float timestamp, bool mirrored) {
			Frame frame = Asset.GetFrame(timestamp);
			return mirrored ? Classifications[frame.Index-1].Mirrored : Classifications[frame.Index-1].Standard;
		}

		public void Compute(TimeSeries timeSeries) {
			Classifications = new Classification[Asset.Frames.Length];
			RootModule rootModule = Asset.GetModule<RootModule>();
			for(int k=0; k<Asset.Frames.Length; k++) {
				float timestamp = Asset.Frames[k].Timestamp;
				Series instance = new Series(timeSeries, Discretization);

				Classifications[k] = new Classification();

				{
					RootModule.Series series = rootModule.ExtractSeries(timeSeries, timestamp, false) as RootModule.Series;
					Vector3 translation = series.GetIntegratedTranslation(0, timeSeries.SampleCount-1);
					Classifications[k].Standard = GetTranslationClassification(translation.DirectionTo(series.Transformations[series.Pivot]), Discretization, Threshold);
				}

				{
					RootModule.Series series = rootModule.ExtractSeries(timeSeries, timestamp, true) as RootModule.Series;
					Vector3 translation = series.GetIntegratedTranslation(0, timeSeries.SampleCount-1);
					Classifications[k].Mirrored = GetTranslationClassification(translation.DirectionTo(series.Transformations[series.Pivot]), Discretization, Threshold);
				}
			}
		}

		public float[] GetTranslationClassification(Vector3 vector, int discretization, float threshold) {
			float[] result = new float[discretization];
			if(vector.magnitude < threshold) {
				return result;
			} else {
				vector = vector.normalized;
				float step = 360f / discretization;
				float range = step/2f;
				for(int i=0; i<discretization; i++) {
					Vector3 reference = Quaternion.AngleAxis(i*step, Vector3.up) * Vector3.forward;
					if(Vector3.Angle(reference, vector) < range) {
						result[i] = 1f;
					}
				}
			}
			return result;
		}

		public class Series : TimeSeries.Component {

			public Matrix4x4 Reference;
			public int Discretization;
			public float[][] Classifications;
			
			private UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.875f, 0.85f, 0.2f, 0.1f);

			public Series(TimeSeries global, int discretization) : base(global) {
				Discretization = discretization;
                Classifications = new float[global.SampleCount][];
				for(int i=0; i<Classifications.Length; i++) {
					Classifications[i] = new float[discretization];
				}
			}

			public override void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
					for(int j=0; j<Classifications[i].Length; j++) {
						Classifications[i][j] = Classifications[i+1][j];
					}
				}
			}

			public override void GUI(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.Begin();
					UltiDraw.OnGUILabel(Rect.GetCenter() + new Vector2(0f, 0.5f*Rect.H + 0.025f), Rect.GetSize(), 0.0175f, "Classifications", UltiDraw.White);
					UltiDraw.End();
				}
			}

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawGUI) {
					UltiDraw.Begin();
					float[] function = new float[Samples.Length];
					Color[] colors = UltiDraw.GetRainbowColors(Discretization);
					for(int i=0; i<Discretization; i++) {
						for(int j=0; j<function.Length; j++) {
							function[j] = Classifications[j][Discretization-1-i];
						}
						float ratio = i.Ratio(Discretization-1, 0);
						float itemSize = Rect.H / Discretization;
						UltiDraw.PlotBars(new Vector2(Rect.X, ratio.Normalize(0f, 1f, Rect.Y + Rect.H/2f - itemSize/2f, Rect.Y - Rect.H/2f + itemSize/2f)), new Vector2(Rect.W, itemSize), function, yMin: 0f, yMax: 1f, barColor : colors[i]);
					}
					UltiDraw.End();
				}
			}

			public void DrawClassificationPatches(Matrix4x4 reference) {
				UltiDraw.Begin();
				float[] values = Classifications[Pivot];		
				float range = 360f/values.Length;
				for(int i=0; i<values.Length; i++) {
					DrawTriangle(i*range, range, Color.Lerp(UltiDraw.Black.Opacity(0.25f), UltiDraw.White.Opacity(0.5f), values[i]));
				}
				void DrawTriangle(float angle, float range, Color color) {
					UltiDraw.DrawTriangle(
						reference.GetPosition(), 
						reference.GetPosition() + Quaternion.AngleAxis(angle + range/2f, Vector3.up) * new Vector3(0f, 0f, 1f).DirectionFrom(reference), 
						reference.GetPosition() + Quaternion.AngleAxis(angle - range/2f, Vector3.up) * new Vector3(0f, 0f, 1f).DirectionFrom(reference), 
						color
					);
				}
				UltiDraw.End();
			}
		}

	}
}