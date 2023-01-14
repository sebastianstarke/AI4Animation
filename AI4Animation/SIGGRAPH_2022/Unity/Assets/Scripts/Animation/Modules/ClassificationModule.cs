#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;

namespace AI4Animation {
	public class ClassificationModule : Module {

		public float Threshold = 0.25f;
        public int Discretization = 4;

		public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			RootModule root = Asset.GetModule<RootModule>();
			Series instance = new Series(global, Discretization);
			instance.Reference = root.GetRootTransformation(timestamp, mirrored);
			for(int i=0; i<instance.Samples.Length; i++) {
                RootModule.Series series = root.ExtractSeries(global, timestamp + instance.Samples[i].Timestamp, mirrored) as RootModule.Series;
				Vector3 translation = series.GetIntegratedTranslation(0, global.SampleCount-1);
				// Vector3 direction = series.GetIntegratedDirection(0, global.SampleCount-1);
				// Vector3 translation = series.GetIntegratedTranslation(0, global.Pivot);
				// Vector3 direction = series.GetIntegratedDirection(0, global.Pivot);
				instance.Translations[i] = instance.GetTranslationClassification(translation.DirectionTo(series.Transformations[series.Pivot]), Discretization, Threshold);
				// instance.Directions[i] = instance.GetDirectionClassification(direction.DirectionTo(series.Transformations[series.Pivot]), Discretization);
				// instance.Directions[i] = instance.GetDirectionClassification(direction.DirectionTo(series.Transformations[series.Pivot]), Discretization);

			}
			return instance;
		}

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
			ExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror).Draw();
            // UltiDraw.DrawArrow(reference.GetPosition(), reference.GetPosition() + translation, 0.8f, 0.05f, 0.1f, UltiDraw.Blue);
            // UltiDraw.DrawArrow(reference.GetPosition(), reference.GetPosition() + direction, 0.8f, 0.05f, 0.1f, UltiDraw.Orange);
		}

		protected override void DerivedInspector(MotionEditor editor) {
			Threshold = EditorGUILayout.FloatField("Threshold", Threshold);
            Discretization = EditorGUILayout.IntField("Discretization", Discretization);
		}

		public class Series : TimeSeries.Component {

			public Matrix4x4 Reference;
			public float[][] Translations;
			// public float[][] Directions;
			
			public Series(TimeSeries global, int discretization) : base(global) {
                Translations = new float[global.SampleCount][];
				// Directions = new float[global.SampleCount][];
				for(int i=0; i<Translations.Length; i++) {
					Translations[i] = new float[discretization];
					// Directions[i] = new float[discretization];
				}
			}

			public override void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
					for(int j=0; j<Translations[i].Length; j++) {
						Translations[i][j] = Translations[i+1][j];
					}
					// for(int j=0; j<Directions[i].Length; j++) {
					// 	Directions[i][j] = Directions[i+1][j];
					// }
				}
			}

			public override void Interpolate(int start, int end) {

			}

			public override void GUI() {
				if(DrawGUI) {

				}
			}

			public override void Draw() {
				if(DrawGUI) {
					UltiDraw.Begin();
					UltiDraw.PlotBars(new Vector2(0.125f, 0.45f), new Vector2(0.2f, 0.35f), Translations, yMin:0f, yMax:1f);
					// UltiDraw.PlotBars(new Vector2(0.15f, 0.35f), new Vector2(0.2f, 0.2f), Directions, yMin:0f, yMax:1f);
					DrawClassificationPatches(Translations[Pivot], UltiDraw.Blue);
					// DrawClassificationPatches(Directions[Pivot], UltiDraw.Orange);
					UltiDraw.End();
				}
			}

			private void DrawClassificationPatches(float[] values, Color color) {
				float range = 360f/values.Length;
				for(int i=0; i<values.Length; i++) {
					DrawTriangle(i*range, range, Color.Lerp(UltiDraw.Black.Opacity(0.25f), color.Opacity(0.5f), values[i]));
				}
				void DrawTriangle(float angle, float range, Color color) {
					UltiDraw.DrawTriangle(
						Reference.GetPosition(), 
						Reference.GetPosition() + Quaternion.AngleAxis(angle + range/2f, Vector3.up) * new Vector3(0f, 0f, 1f).DirectionFrom(Reference), 
						Reference.GetPosition() + Quaternion.AngleAxis(angle - range/2f, Vector3.up) * new Vector3(0f, 0f, 1f).DirectionFrom(Reference), 
						color
					);
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
			
			// public float[] GetDirectionClassification(Vector3 vector, int discretization) {
			// 	float[] result = new float[discretization];
			// 	if(vector != Vector3.zero) {
			// 		float step = 360f / discretization;
			// 		float range = step/2f;
			// 		for(int i=0; i<discretization; i++) {
			// 			Vector3 reference = Quaternion.AngleAxis(i*step, Vector3.up) * Vector3.forward;
			// 			if(Vector3.Angle(reference, vector) < range) {
			// 				result[i] = Vector3.Angle(reference, vector) / range;
			// 			}
			// 		}
			// 	}
			// 	return result;
			// }
		}

	}
}
#endif
