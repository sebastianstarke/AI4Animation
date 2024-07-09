using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class DepthMapModule : Module {
        public int Sensor = 0;
        public Axis Axis = Axis.ZPositive;
        public int Resolution = 20;
        public float Size = 10f;
        public float Distance = 10f;
        public LayerMask Mask = -1;
        public bool ShowImage = false;
        
        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            return null;
        }

#if UNITY_EDITOR
        protected override void DerivedInitialize() {
            MotionAsset.Hierarchy.Bone bone = Asset.Source.FindBoneContains("Head");
            if(bone == null) {
                Debug.Log("Could not find depth map sensor.");
            } else {
                Sensor = bone.Index;
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
            DepthMap sensor = GetDepthMap(editor.GetCurrentFrame(), editor.Mirror);
            sensor.Draw();
            if(ShowImage) {
                UltiDraw.Begin();
                UltiDraw.GUIRectangle(Vector2.one/2f, Vector2.one, UltiDraw.Mustard);
                Vector2 size = new Vector2(0.5f, 0.5f*Screen.width/Screen.height);
                for(int x=0; x<sensor.GetResolution(); x++) {
                    for(int y=0; y<sensor.GetResolution(); y++) {
                        float distance = Vector3.Distance(sensor.Points[sensor.GridToArray(x,y)], sensor.Pivot.GetPosition());
                        float intensity = 1f - distance / sensor.GetDistance();
                        UltiDraw.GUIRectangle(
                            Vector2.one/2f - size/2f + new Vector2((float)x*size.x, 
                            (float)y*size.y) / (sensor.GetResolution()-1), 
                            size / (sensor.GetResolution()-1), 
                            Color.Lerp(Color.black, Color.white, intensity)
                        );
                    }
                }
                UltiDraw.End();
            }
        }

        protected override void DerivedInspector(MotionEditor editor) {
            Sensor = EditorGUILayout.Popup("Sensor", Sensor, Asset.Source.GetBoneNames());
            Axis = (Axis)EditorGUILayout.EnumPopup("Axis", Axis);
            Resolution = EditorGUILayout.IntField("Resolution", Resolution);
            Size = EditorGUILayout.FloatField("Size", Size);
            Distance = EditorGUILayout.FloatField("Distance", Distance);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
            ShowImage = EditorGUILayout.Toggle("Show Image", ShowImage);
        }
#endif
        public DepthMap GetDepthMap(Frame frame, bool mirrored) {
            DepthMap sensor = new DepthMap(Resolution, Size, Distance);
            Matrix4x4 pivot = frame.GetBoneTransformation(Sensor, mirrored);
            pivot *= Matrix4x4.TRS(Vector3.zero, Quaternion.FromToRotation(Vector3.forward, Axis.GetAxis()), Vector3.one);
            sensor.Sense(pivot, Mask);
            return sensor;
        }

        public class DepthMap {

            public Matrix4x4 Pivot = Matrix4x4.identity;
            public Vector3[] Points = new Vector3[0];

            private int Resolution;
            private float Size;
            private float Distance;

            public DepthMap(int resolution, float size, float distance) {
                Resolution = resolution;
                Size = size;
                Distance = distance;
            }

            public int GetResolution() {
                return Resolution;
            }

            public float GetSize() {
                return Size;
            }

            public float GetDistance() {
                return Distance;
            }

            public void Sense(Matrix4x4 pivot, LayerMask mask) {
                Pivot = pivot;
                Points = new Vector3[Resolution*Resolution];
                RaycastHit hit;
                for(int x=0; x<Resolution; x++) {
                    for(int y=0; y<Resolution; y++) {
                        Vector3 direction = new Vector3(
                            -Size/2f + (float)x/(float)(Resolution-1) * Size, 
                            -Size/2f + (float)y/(float)(Resolution-1) * Size,
                            Distance).PositionFrom(Pivot) 
                            - 
                            Pivot.GetPosition();
                        direction.Normalize();
                        if(Physics.Raycast(Pivot.GetPosition(), direction, out hit, Distance, mask)) {
                            Points[GridToArray(x,y)] = hit.point;
                        } else {
                            Points[GridToArray(x,y)] = Pivot.GetPosition() + Distance*direction;
                        }
                    }
                }
            }

            public int GridToArray(int x, int y) {
                return x + y*Resolution;
            }

            public float[] GetDistances() {
                float[] distances = new float[Points.Length];
                for(int i=0; i<Points.Length; i++) {
                    distances[i] = Vector3.Distance(Pivot.GetPosition(), Points[i]);
                }
                return distances;
            }

            public void Draw() {
                UltiDraw.Begin();
                UltiDraw.DrawTranslateGizmo(Pivot.GetPosition(), Pivot.GetRotation(), 0.1f);
                float size = 0.5f * Size/Resolution;
                Quaternion rotation = Pivot.GetRotation();
                for(int i=0; i<Points.Length; i++) {
                    UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Opacity(0.05f));
                    UltiDraw.DrawQuad(Points[i], rotation, size, size, UltiDraw.Orange.Opacity(0.5f));
                }
                UltiDraw.End();
            }

        }

    }
}
