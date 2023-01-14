#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

namespace AI4Animation {
	public class PerlinModule : Module {

        public int Seed;
        public int Octaves;
        public float Amplitude;
        public float Frequency;

		public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
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
            UltiDraw.Begin();
            float[] perlin = Perlin.Sample(editor.GetTimeSeries().GetTimestamps(), Seed, Octaves, Amplitude, Frequency, editor.GetTimestamp());
            UltiDraw.PlotFunction(new Vector2(0.5f, 0.1f), new Vector2(0.75f, 0.2f), perlin, yMin:-2f, yMax:2f);
            UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            Seed = EditorGUILayout.IntField("Seed", Seed);
            Octaves = EditorGUILayout.IntField("Octaves", Octaves);
            Amplitude = EditorGUILayout.FloatField("Amplitude", Amplitude);
            Frequency = EditorGUILayout.FloatField("Frequency", Frequency);
		}

        public float[] GetPerlinValues(TimeSeries timeSeries, float timestamp, bool mirrored) {
            return Perlin.Sample(timeSeries.GetTimestamps(), Seed, Octaves, Amplitude, Frequency, timestamp);
        }

	}
}
#endif
