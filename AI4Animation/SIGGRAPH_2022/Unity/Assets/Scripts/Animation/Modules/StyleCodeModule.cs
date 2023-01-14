#if UNITY_EDITOR
using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace AI4Animation {
	public class StyleCodeModule : Module {

        // [Serializable]
        // public class CODE {
        //     private float[] Code = null;
        //     [Serializable]
        //     public class CHANNEL {
        //         public float[] Means;
        //         public float[] Sigmas;
        //         public CHANNEL(float[] means, float[] sigmas) {
        //             Means = means;
        //             Sigmas = sigmas;
        //         }
        //     }
        //     public string Name;
        //     public CHANNEL[] Channels;
        //     public float[] MinMean;
        //     public float[] MaxMean;
        //     public float[] MinSigma;
        //     public float[] MaxSigma;
        //     public CODE(string name, CHANNEL[] channels, float[] minMean, float[] maxMean, float[] minSigma, float[] maxSigma) {
        //         Name = name;
        //         Channels = channels;
        //         MinMean = minMean;
        //         MaxMean = maxMean;
        //         MinSigma = minSigma;
        //         MaxSigma = maxSigma;
        //     }
        //     public float[] GetCode() {
        //         // if(Code == null) {
        //             float[][] code = new float[2*Channels.Length][];
        //             for(int i=0; i<Channels.Length; i++) {
        //                 code[i] = Channels[i].Means;
        //                 code[i+Channels.Length] = Channels[i].Sigmas;
        //             }
        //             return code.Flatten();
        //             // Code = code.Flatten();
        //         // }
        //         // return Code;
        //     }
        // }

        // public CODE Code = null;

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
            // if(Code == null) {
            //     return;
            // }
			// UltiDraw.Begin();
            // {
            //     float min = 0.25f;
            //     float max = 0.75f;
            //     float h = (max-min) / Code.Channels.Length;
            //     for(int i=0; i<Code.Channels.Length; i++) {
            //         float[] rMin = new float[Code.Channels[i].Means.Length];
            //         float[] rMax = new float[Code.Channels[i].Means.Length];
            //         rMax.SetAll(1f);
            //         UltiDraw.PlotBars(new Vector2(0.25f, i.Ratio(0, Code.Channels.Length-1).Normalize(0f, 1f, min, max)), new Vector2(0.4f, h), Code.Channels[i].Means.Normalize(Code.MinMean, Code.MaxMean, rMin, rMax), 0f, 1f);
            //     }
            // }
            // {
            //     float min = 0.25f;
            //     float max = 0.75f;
            //     float h = (max-min) / Code.Channels.Length;
            //     for(int i=0; i<Code.Channels.Length; i++) {
            //         float[] rMin = new float[Code.Channels[i].Sigmas.Length];
            //         float[] rMax = new float[Code.Channels[i].Sigmas.Length];
            //         rMax.SetAll(1f);
            //         UltiDraw.PlotBars(new Vector2(0.75f, i.Ratio(0, Code.Channels.Length-1).Normalize(0f, 1f, min, max)), new Vector2(0.4f, h), Code.Channels[i].Sigmas.Normalize(Code.MinSigma, Code.MaxSigma, rMin, rMax), 0f, 1f);
            //     }
            // }
            // UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            // if(Code == null) {
            //     return;
            // }
            // EditorGUILayout.LabelField(Code.Name);
            // foreach(CODE.CHANNEL channel in Code.Channels) {
            //     Utility.SetGUIColor(UltiDraw.LightGrey);
            //     using(new EditorGUILayout.VerticalScope ("Box")) {
            //         Utility.ResetGUIColor();
            //         EditorGUILayout.HelpBox(channel.Means.Format(), MessageType.None);
            //         EditorGUILayout.HelpBox(channel.Sigmas.Format(), MessageType.None);
            //     }
            // }
		}

	}
}
#endif