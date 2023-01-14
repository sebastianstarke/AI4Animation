#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.Collections.Generic;

namespace AI4Animation {
	public class VisualizationModule : Module {

        public float LineWidth = 0.025f;
        public float Past = 1f;
        public float Future = 1f;

        public bool DrawGraph = false;
        public float Width = 0.5f;
        public float Height = 0.25f;
        public float YMin = -1f;
        public float YMax = 1f;
        public float Thickness = 0.05f;
        public Color BackgroundColor = Color.white;

        public int Smoothing = 5;

        public string[] Bones = new string[0];

		public override void DerivedResetPrecomputation() {

		}

		public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
			return null;
		}

		protected override void DerivedInitialize() {
            Thickness = 0.025f;
            Past = 2f;
            Future = 0f;
            ArrayExtensions.Append(ref Bones, "LeftHandSite");
            ArrayExtensions.Append(ref Bones, "RightHandSite");
            ArrayExtensions.Append(ref Bones, "LeftFootSite");
            ArrayExtensions.Append(ref Bones, "RightFootSite");
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
            Frame[] past = Asset.GetFrames(editor.GetTimestamp() - Past, editor.GetTimestamp());
            Frame[] future = Asset.GetFrames(editor.GetTimestamp(), editor.GetTimestamp() + Future);

            UltiDraw.Begin();
            for(int b=0; b<Bones.Length; b++) {
                MotionAsset.Hierarchy.Bone bone = Asset.Source.FindBone(Bones[b]);
                if(bone != null) {
                    Color color = UltiDraw.GetRainbowColor(b, Bones.Length);
                    for(int i=1; i<past.Length; i++) {
                        float ratio = i.Ratio(0, past.Length-1);
                        ratio = Mathf.Pow(ratio, 2f);
                        UltiDraw.DrawLine(past[i-1].GetBoneTransformation(bone.Index, editor.Mirror).GetPosition(), past[i].GetBoneTransformation(bone.Index, editor.Mirror).GetPosition(), LineWidth, color.Opacity(ratio));
                    }
                    for(int i=1; i<future.Length; i++) {
                        float ratio = 1f - i.Ratio(0, future.Length-1);
                        ratio = Mathf.Pow(ratio, 2f);
                        UltiDraw.DrawLine(future[i-1].GetBoneTransformation(bone.Index, editor.Mirror).GetPosition(), future[i].GetBoneTransformation(bone.Index, editor.Mirror).GetPosition(), LineWidth, color.Opacity(ratio));
                    }
                }
            }
            
            if(DrawGraph) {
                List<float[]> functions = new List<float[]>();
                for(int b=0; b<Bones.Length; b++) {
                    MotionAsset.Hierarchy.Bone bone = Asset.Source.FindBone(Bones[b]);
                    if(bone != null) {
                        List<float> values = new List<float>();
                        for(int i=0; i<past.Length; i++) {
                            values.Add(past[i].GetBoneVelocity(bone.Index, editor.Mirror).magnitude);
                        }
                        for(int i=0; i<future.Length; i++) {
                            values.Add(future[i].GetBoneVelocity(bone.Index, editor.Mirror).magnitude);
                        }
                        float[] v = values.ToArray();
                        v.SmoothGaussian(Smoothing);
                        functions.Add(v);
                    }
                }
                UltiDraw.PlotFunctions(
                    new Vector2(0.5f, 0.25f), 
                    new Vector2(Width, Height), 
                    functions.ToArray(),
                    UltiDraw.Dimension.X,
                    YMin, YMax,
                    Thickness,
                    BackgroundColor
                );
            }

            UltiDraw.End();
		}

		protected override void DerivedInspector(MotionEditor editor) {
            LineWidth = EditorGUILayout.FloatField("Line Width", LineWidth);
            Past = EditorGUILayout.FloatField("Past", Past);
            Future = EditorGUILayout.FloatField("Future", Future);

            DrawGraph = EditorGUILayout.Toggle("Draw Graph", DrawGraph);
            Width = EditorGUILayout.FloatField("Width", Width);
            Height = EditorGUILayout.FloatField("Height", Height);
            YMin = EditorGUILayout.FloatField("Y Min", YMin);
            YMax = EditorGUILayout.FloatField("Y Max", YMax);
            Thickness = EditorGUILayout.FloatField("Thickness", Thickness);
            BackgroundColor = EditorGUILayout.ColorField("Background Color", BackgroundColor);
            Smoothing = EditorGUILayout.IntField("Smoothing", Smoothing);

            EditorGUILayout.BeginHorizontal();
            if(Utility.GUIButton("Add Bone", UltiDraw.DarkGrey, UltiDraw.White)) {
                ArrayExtensions.Append(ref Bones, string.Empty);
            }
            if(Utility.GUIButton("Remove Bone", UltiDraw.DarkGrey, UltiDraw.White)) {
                ArrayExtensions.Shrink(ref Bones);
            }
            EditorGUILayout.EndHorizontal();
            for(int i=0; i<Bones.Length; i++) {
                Bones[i] = EditorGUILayout.TextField("Bone " + (i+1), Bones[i]);
            }
		}

	}
}
#endif
