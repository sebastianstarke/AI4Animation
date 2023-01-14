#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

namespace AI4Animation {
	public class SineModule : Module {

        public float MinAmplitude;
        public float MaxAmplitude;
        public float MinFrequency;
        public float MaxFrequency;

        public float Amplitude = 1f;
        public float Frequency = 1f;

        public int NoiseSeed = 0;
        public float NoiseAmplitude = 2f;
        public float NoiseFrequency = 2f;

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
            TimeSeries timeSeries = editor.GetTimeSeries();
            float[] values = new float[timeSeries.Samples.Length];
            float[] noiseAmplitude = Perlin.Sample(timeSeries.GetTimestamps(), NoiseSeed, 1, NoiseAmplitude, 1f, editor.GetTimestamp());
            float[] noiseFrequency = Perlin.Sample(timeSeries.GetTimestamps(), NoiseSeed, 1, NoiseFrequency, 1f, editor.GetTimestamp());
            for(int i=0; i<values.Length; i++) {
                noiseAmplitude[i] = noiseAmplitude[i].Normalize(-NoiseAmplitude, NoiseAmplitude, 0f, NoiseAmplitude);
                noiseFrequency[i] = noiseFrequency[i].Normalize(-NoiseFrequency, NoiseFrequency, 0f, NoiseFrequency);
            }
            for(int i=0; i<timeSeries.Samples.Length; i++) {
                float value = SineValue(timeSeries.Samples[i].Timestamp, Amplitude + noiseAmplitude[i], Frequency + noiseFrequency[i], editor.GetTimestamp());
                values[i] = value;
            }
            UltiDraw.PlotFunction(new Vector2(0.5f, 0.5f), new Vector2(0.5f, 0.25f), values, -5f, 5f);
            UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            Amplitude = EditorGUILayout.FloatField("Amplitude", Amplitude);
            Frequency = EditorGUILayout.FloatField("Frequency", Frequency);

            NoiseSeed = EditorGUILayout.IntField("Noise Seed", NoiseSeed);
            NoiseAmplitude = EditorGUILayout.FloatField("Noise Amplitude", NoiseAmplitude);
            NoiseFrequency = EditorGUILayout.FloatField("NoiseFrequency", NoiseFrequency);
		}

        //This function returns a regular sine function, repeating in normalized range for every interval [0,1] by default.
        //Shift is expressed in seconds, where each second covers 1s time for 1Hz frequency.
        private float SineValue(float t, float a, float f, float s) {
            return a * Mathf.Sin(2f*Mathf.PI*(f*t + s));
        }

	}
}
#endif
