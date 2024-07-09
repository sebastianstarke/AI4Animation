using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class DistanceMapModule : Module {
        public Vector3 Size = Vector3.one;
        public Vector3Int Resolution = new Vector3Int(10, 10, 10);
        public LayerMask Mask = -1;
        public Color Color = UltiDraw.Cyan;
        public bool DrawReferences = false;
        // public bool DrawDistribution = false;
        // public UltiDraw.GUIRect Rect = new UltiDraw.GUIRect(0.5f, 0.1f, 0.9f, 0.1f);
        public string BoneName;
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
            DistanceMap sensor = GetDistanceMap(editor.GetCurrentFrame(), editor.Mirror);
            sensor.Draw(Color);
            if(DrawReferences) {
                sensor.DrawReferences();
            }
            // if(DrawDistribution) {
            // 	sensor.DrawDistribution(Color, Rect);
            // }
        }

        protected override void DerivedInspector(MotionEditor editor) {
            Size = EditorGUILayout.Vector3Field("Size", Size);
            BoneName = EditorGUILayout.TextField("Bone Name", BoneName);
            Resolution = EditorGUILayout.Vector3IntField("Resolution", Resolution);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
            Color = EditorGUILayout.ColorField("Color", Color);
            DrawReferences = EditorGUILayout.Toggle("Draw References", DrawReferences);
            // DrawDistribution = EditorGUILayout.Toggle("Draw Distribution", DrawDistribution);
            // Rect.Inspector();
            EditorGUILayout.LabelField("Samples: " + 0);
        }
#endif
        public DistanceMap GetDistanceMap(Frame frame, bool mirrored) {
            DistanceMap sensor = new DistanceMap(Resolution);
            RootModule module = Asset.HasModule<RootModule>() ? Asset.GetModule<RootModule>() : null;
            if(module != null) {
                Vector3 position = module.GetRootPosition(frame.Timestamp, mirrored);
                Quaternion rotation = module.GetRootRotation(frame.Timestamp, mirrored);
                sensor.Sense(Matrix4x4.TRS(position + new Vector3(0f, 0.5f*Size.y, 0f), rotation, Vector3.one), Mask, Size);
            } else {
                sensor.Sense(frame.GetBoneTransformation(BoneName, mirrored), Mask, Size);
            }
            return sensor;
        }

        public class DistanceMap {

            public Matrix4x4 Pivot = Matrix4x4.identity;
            public Vector3[] Points = new Vector3[0];
            public Vector3[] Closest = new Vector3[0];

            public Vector3[] References = new Vector3[0];
            public float[] Distances = new float[0];

            public Vector3Int Resolution = new Vector3Int(10, 10, 10);
            public Vector3 Size = Vector3.one;

            public DistanceMap(Vector3Int resolution) {
                Size = Vector3.zero;
                Resolution = resolution;
                Generate();
            }

            public void Setup(Vector3Int resolution) {
                if(Resolution != resolution) {
                    Resolution = resolution;
                    Generate();
                }
            }

            private Vector3 GetStep() {
                return new Vector3(Size.x / Resolution.x, Size.y / Resolution.y, Size.z / Resolution.z);
            }

            private int GetDimensionality() {
                return Resolution.x * Resolution.y * Resolution.z;
            }

            public void Generate() {
                Points = new Vector3[GetDimensionality()];
                References = new Vector3[GetDimensionality()];
                Closest = new Vector3[GetDimensionality()];
                Distances = new float[GetDimensionality()];
                for(int y=0; y<Resolution.y; y++) {
                    for(int x=0; x<Resolution.x; x++) {
                        for(int z=0; z<Resolution.z; z++) {
                            Points[y*Resolution.x*Resolution.z + x*Resolution.z + z] = new Vector3(
                                -0.5f + (x+0.5f)/Resolution.x,
                                -0.5f + (y+0.5f)/Resolution.y,
                                -0.5f + (z+0.5f)/Resolution.z
                            );
                        }
                    }
                }
            }

            public void Sense(Matrix4x4 pivot, LayerMask mask, Vector3 size, float smoothing=0f) {
                Pivot = Utility.Interpolate(Pivot, pivot, 1f-smoothing);
                Size = smoothing*Size + (1f-smoothing)*size;

                Vector3 pivotPosition = Pivot.GetPosition();
                Quaternion pivotRotation = Pivot.GetRotation();
                Vector3 sensorPosition = pivot.GetPosition();
                Quaternion sensorRotation = pivot.GetRotation();

                Collider[] colliders = Physics.OverlapBox(pivot.GetPosition(), size/2f, pivot.GetRotation(), mask);
                for(int i=0; i<Points.Length; i++) {
                    if(Size == Vector3.zero) {
                        References[i] = pivotPosition;
                        Distances[i] = 0f;
                    } else {
                        References[i] = pivotPosition + pivotRotation * Vector3.Scale(Points[i], Size);
                        Vector3 sensor = sensorPosition + sensorRotation * Vector3.Scale(Points[i], Size);
                        Vector3 pointMin = sensor;
                        float dMin = 0f;
                        if(colliders.Length > 0) {
                            pointMin = colliders[0].ClosestPoint(sensor);
                            dMin = Vector3.Distance(sensor, pointMin);
                            for(int j=1; j<colliders.Length; j++) {
                                Vector3 point = colliders[j].ClosestPoint(sensor);
                                float d = Vector3.Distance(sensor, point);
                                if(d < dMin) {
                                    pointMin = point;
                                    dMin = d;
                                }
                                if(dMin == 0f) {
                                    break;
                                }
                            }
                        }
                        Closest[i] = pointMin.PositionTo(Pivot);

                        Distances[i] = smoothing*Distances[i] + (1f-smoothing)*dMin;
                    }
                }
            }

            public void Sense(Matrix4x4 pivot, Collider[] colliders, Vector3 size, float smoothing=0f) {
                Pivot = Utility.Interpolate(Pivot, pivot, 1f-smoothing);
                Size = smoothing*Size + (1f-smoothing)*size;

                Vector3 pivotPosition = Pivot.GetPosition();
                Quaternion pivotRotation = Pivot.GetRotation();
                Vector3 sensorPosition = pivot.GetPosition();
                Quaternion sensorRotation = pivot.GetRotation();

                for(int i=0; i<Points.Length; i++) {
                    if(Size == Vector3.zero) {
                        References[i] = pivotPosition;
                        Distances[i] = 0f;
                    } else {
                        References[i] = pivotPosition + pivotRotation * Vector3.Scale(Points[i], Size);
                        Vector3 sensor = sensorPosition + sensorRotation * Vector3.Scale(Points[i], Size);
                        Vector3 pointMin = sensor;
                        float dMin = 0f;
                        if(colliders != null && colliders.Length > 0) {
                            pointMin = colliders[0].ClosestPoint(sensor);
                            dMin = Vector3.Distance(sensor, pointMin);
                            for(int j=1; j<colliders.Length; j++) {
                                Vector3 point = colliders[j].ClosestPoint(sensor);
                                float d = Vector3.Distance(sensor, point);
                                if(d < dMin) {
                                    pointMin = point;
                                    dMin = d;
                                }
                                if(dMin == 0f) {
                                    break;
                                }
                            }
                        }
                        Closest[i] = pointMin.PositionTo(Pivot);

                        Distances[i] = smoothing*Distances[i] + (1f-smoothing)*dMin;
                    }
                }
            }

            public void Retransform(Matrix4x4 pivot) {
                Pivot = pivot;
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                for(int i=0; i<References.Length; i++) {
                    References[i] = position + rotation * Vector3.Scale(Points[i], Size);
                }
            }

            public void Draw(Color color) {
                float max = Distances.Max();
                float[] distances = new float[Distances.Length];
                for(int i=0; i<distances.Length; i++) {
                    distances[i] = Distances[i].Normalize(0f, max, 1f, 0f);
                }

                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                UltiDraw.Begin();
                Vector3 step = GetStep();
                if(Size != Vector3.zero) {
                    UltiDraw.DrawWireCuboid(position, rotation, Size, color);
                    for(int i=0; i<Points.Length; i++) {
                        if(distances[i] > 0f) {
                            UltiDraw.DrawCuboid(References[i], rotation, step, Color.Lerp(UltiDraw.Transparent, color, distances[i]));
                        }
                    }
                }
                UltiDraw.End();
            }

            public void DrawReferences() {
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                UltiDraw.Begin();
                Vector3 step = GetStep();
                Color reference = UltiDraw.Black.Opacity(0.05f);
                UltiDraw.DrawTranslateGizmo(position, rotation, 0.25f);
                for(int i=0; i<Points.Length; i++) {
                    UltiDraw.DrawCuboid(References[i], rotation, step == Vector3.zero ? 0.25f * Vector3.one : step, reference);
                    UltiDraw.DrawSphere(References[i], rotation, 0.0125f, UltiDraw.Black.Opacity(0.25f));
                    UltiDraw.DrawLine(References[i], Closest[i].PositionFrom(Pivot), UltiDraw.White.Opacity(0.5f));
                    UltiDraw.DrawSphere(Closest[i].PositionFrom(Pivot), rotation, 0.0125f, UltiDraw.White);
                }
                UltiDraw.End();
            }

            public void DrawDistribution(Color color, UltiDraw.GUIRect rect) {
                float max = Distances.Max();
                float[] distances = new float[Distances.Length];
                for(int i=0; i<distances.Length; i++) {
                    distances[i] = Distances[i].Normalize(0f, max, 1f, 0f);
                }

                UltiDraw.Begin();
                UltiDraw.PlotFunction(rect.GetCenter(), rect.GetSize(), distances, yMin: 0f, yMax: 1f, backgroundColor: UltiDraw.White, lineColor: color);
                UltiDraw.End();
            }

        }

    }
}
