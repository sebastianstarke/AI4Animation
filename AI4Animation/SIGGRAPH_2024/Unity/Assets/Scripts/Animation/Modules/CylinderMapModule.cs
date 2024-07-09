using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class CylinderMapModule : Module {
        public float Size = 1f;
        public int Resolution = 10;
        public int Layers = 10;
        public bool Overlap = true;
        public LayerMask Mask = -1;
        public Color Color = UltiDraw.Cyan;
        public bool DrawReferences = false;
        public bool DrawDistribution = false;

        private int Samples = 0;

        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }
#if UNITY_EDITOR
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
            CylinderMap sensor = GetCylinderMap(editor.GetCurrentFrame(), editor.Mirror);
            sensor.Draw(Color, DrawReferences, DrawDistribution);
        }

        protected override void DerivedInspector(MotionEditor editor) {
            Size = EditorGUILayout.FloatField("Size", Size);
            Resolution = Mathf.Clamp(EditorGUILayout.IntField("Resolution", Resolution), 1, 25);
            Layers = Mathf.Clamp(EditorGUILayout.IntField("Layers", Layers), 1, 25);
            Overlap = EditorGUILayout.Toggle("Overlap", Overlap);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
            Color = EditorGUILayout.ColorField("Color", Color);
            DrawReferences = EditorGUILayout.Toggle("Draw References", DrawReferences);
            DrawDistribution = EditorGUILayout.Toggle("Draw Distribution", DrawDistribution);
            EditorGUILayout.LabelField("Samples: " + Samples);
        }
#endif
        public CylinderMap GetCylinderMap(Frame frame, bool mirrored) {
            CylinderMap sensor = new CylinderMap(Size, Resolution, Layers, Overlap);
            RootModule module = Asset.GetModule<RootModule>();
            if(module != null) {
                Vector3 position = module.GetRootPosition(frame.Timestamp, mirrored);
                Quaternion rotation = module.GetRootRotation(frame.Timestamp, mirrored);
                sensor.Sense(Matrix4x4.TRS(position + new Vector3(0f, 0f, 0f), rotation, Vector3.one), Mask);
            } else {
                sensor.Sense(frame.GetBoneTransformation(0, mirrored), Mask);
            }
            Samples = sensor.Points.Length;
            return sensor;
        }

        public class CylinderMap {

            public Matrix4x4 Pivot = Matrix4x4.identity;
            public Vector3[] Points = new Vector3[0];

            public Vector3[] References = new Vector3[0];
            public float[] Occupancies = new float[0];

            public float Size = 1f;
            public int Resolution = 10;
            public int Layers = 10;
            public bool Overlap = true;

            private float[] Radius = new float[0];

            public CylinderMap(float size, int resolution, int layers, bool overlap) {
                Size = size;
                Resolution = resolution;
                Layers = layers;
                Overlap = overlap;
                Generate();
            }

            public void Setup(float size, int resolution, int layers, bool overlap) {
                if(Size != size || Resolution != resolution || Layers != layers || Overlap != overlap) {
                    Size = size;
                    Resolution = resolution;
                    Layers = layers;
                    Overlap = overlap;
                    Generate();
                }
            }

            private float GetHeight() {
                return (Layers-1) * GetDiameter();
            }

            private float GetDiameter() {
                return Size / (float)(Resolution-1);
            }

            private void Generate() {
                float diameter = GetDiameter();
                List<Vector3> points = new List<Vector3>();
                List<float> radius = new List<float>();
                for(int y=0; y<Layers; y++) {
                    for(int z=0; z<Resolution; z++) {
                        float coverage = 0.5f * diameter;
                        float distance = (float)z * coverage;
                        float arc = 2f * Mathf.PI * distance;
                        int count = Mathf.RoundToInt(arc / coverage);
                        for(int x=0; x<count; x++) {
                            float degrees = (float)x/(float)count*360f;
                            points.Add(
                                new Vector3(
                                distance*Mathf.Cos(Mathf.Deg2Rad*degrees),
                                (float)y*coverage,
                                distance*Mathf.Sin(Mathf.Deg2Rad*degrees)
                                )
                            );
                            if(Overlap) {
                                radius.Add(0.5f*Mathf.Sqrt(2f)*coverage);
                            } else {
                                radius.Add(0.5f*coverage);
                            }
                        }
                    }
                }
                Points = points.ToArray();
                Radius = radius.ToArray();
                References = new Vector3[Points.Length];
                Occupancies = new float[Points.Length];
            }

            public void Sense(Matrix4x4 pivot, LayerMask mask) {
                Pivot = pivot;
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                for(int i=0; i<Points.Length; i++) {
                    References[i] = position + rotation * Points[i];
                    if(Size == 0f) {
                        Occupancies[i] = 0f;
                    } else {
                        Collider c;
                        Vector3 closest = Utility.GetClosestPointOverlapSphere(References[i], Radius[i], mask, out c);
                        Occupancies[i] = c == null ? 0f : 1f - Vector3.Distance(References[i], closest) / Radius[i];
                    }
                }
            }

            public void Draw(Color color, bool references=false, bool distribution=false) {
                if(Size == 0f) {
                    return;
                }

                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();

                float height = GetHeight();

                UltiDraw.Begin();
                UltiDraw.DrawWireCylinder(position + rotation * new Vector3(0f, 0.25f*height, 0f), rotation, new Vector3(Size, 0.5f*height, Size), UltiDraw.Black);
                if(references) {
                    Color reference = UltiDraw.Black.Opacity(0.05f);
                    for(int i=0; i<Points.Length; i++) {
                        UltiDraw.DrawSphere(References[i], Quaternion.identity, 2f*Radius[i], reference);
                    }
                }
                for(int i=0; i<Points.Length; i++) {
                    if(Occupancies[i] > 0f) {
                        UltiDraw.DrawSphere(References[i], rotation, 2f*Radius[i], Color.Lerp(UltiDraw.Transparent, color, Occupancies[i]));
                    }
                }
                if(distribution) {
                    UltiDraw.PlotFunction(new Vector2(0.5f, 0.15f), new Vector2(0.8f, 0.2f), Occupancies, yMin: 0f, yMax: 1f, backgroundColor: UltiDraw.White, lineColor: color);
                }
                UltiDraw.End();
            }

        }

    }
}
