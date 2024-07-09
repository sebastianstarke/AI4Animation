using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class SphereMapModule : Module {

        public float Radius = 1f;
        public LayerMask Mask = -1;

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
            SphereMap sensor = GetSphereMap(editor.GetCurrentFrame(), editor.Mirror);
            sensor.Draw();
        }

        protected override void DerivedInspector(MotionEditor editor) {
            Radius = EditorGUILayout.FloatField("Radius", Radius);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
        }

        public SphereMap GetSphereMap(Frame frame, bool mirrored) {
            SphereMap sensor = new SphereMap(Radius, Mask);
            RootModule module = Asset.GetModule<RootModule>();
            sensor.Sense(module == null ? Matrix4x4.identity : module.GetRootTransformation(frame.Timestamp, mirrored));
            return sensor;
        }
#endif
        public class SphereMap {
            
            public Matrix4x4 Pivot = Matrix4x4.identity;
            public Vector3[] Points = new Vector3[0];

            private float Radius;
            private LayerMask Mask;

            public SphereMap(float radius, LayerMask mask) {
                Radius = radius;
                Mask = mask;
            }

            public float GetRadius() {
                return Radius;
            }

            public void Sense(Matrix4x4 pivot) {
                Pivot = pivot;
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Quaternion.AngleAxis(Pivot.GetRotation().eulerAngles.y, Vector3.up);
                Points = GetSpherePoints();
                RaycastHit hit;
                for(int i=0; i<Points.Length; i++) {
                    Vector3 target = position + rotation * Points[i];
                    if(Physics.Raycast(position, target - position, out hit, Radius, Mask)) {
                        Points[i] = hit.point;
                    } else {
                        Points[i] = target;
                    }
                }
            }

            private Vector3[] GetSpherePoints() {
                int n = 5;
                float radius = 1f;

                int nn = n * 4;
                int vertexNum = (nn * nn / 16) * 24;
                Vector3[] vertices = new Vector3[vertexNum];

                Quaternion[] init_vectors = new Quaternion[24];
                // 0
                init_vectors[0] = new Quaternion(0, 1, 0, 0);   //the triangle vertical to (1,1,1)
                init_vectors[1] = new Quaternion(0, 0, 1, 0);
                init_vectors[2] = new Quaternion(1, 0, 0, 0);
                // 1
                init_vectors[3] = new Quaternion(0, -1, 0, 0);  //to (1,-1,1)
                init_vectors[4] = new Quaternion(1, 0, 0, 0);
                init_vectors[5] = new Quaternion(0, 0, 1, 0);
                // 2
                init_vectors[6] = new Quaternion(0, 1, 0, 0);   //to (-1,1,1)
                init_vectors[7] = new Quaternion(-1, 0, 0, 0);
                init_vectors[8] = new Quaternion(0, 0, 1, 0);
                // 3
                init_vectors[9] = new Quaternion(0, -1, 0, 0);  //to (-1,-1,1)
                init_vectors[10] = new Quaternion(0, 0, 1, 0);
                init_vectors[11] = new Quaternion(-1, 0, 0, 0);
                // 4
                init_vectors[12] = new Quaternion(0, 1, 0, 0);  //to (1,1,-1)
                init_vectors[13] = new Quaternion(1, 0, 0, 0);
                init_vectors[14] = new Quaternion(0, 0, -1, 0);
                // 5
                init_vectors[15] = new Quaternion(0, 1, 0, 0); //to (-1,1,-1)
                init_vectors[16] = new Quaternion(0, 0, -1, 0);
                init_vectors[17] = new Quaternion(-1, 0, 0, 0);
                // 6
                init_vectors[18] = new Quaternion(0, -1, 0, 0); //to (-1,-1,-1)
                init_vectors[19] = new Quaternion(-1, 0, 0, 0);
                init_vectors[20] = new Quaternion(0, 0, -1, 0);
                // 7
                init_vectors[21] = new Quaternion(0, -1, 0, 0);  //to (1,-1,-1)
                init_vectors[22] = new Quaternion(0, 0, -1, 0);
                init_vectors[23] = new Quaternion(1, 0, 0, 0);
                
                int j = 0;
                for (int i = 0; i < 24; i += 3) {
                    for (int p = 0; p < n; p++) {   
                        Quaternion edge_p1 = Quaternion.Lerp(init_vectors[i], init_vectors[i + 2], (float)p / n);
                        Quaternion edge_p2 = Quaternion.Lerp(init_vectors[i + 1], init_vectors[i + 2], (float)p / n);
                        Quaternion edge_p3 = Quaternion.Lerp(init_vectors[i], init_vectors[i + 2], (float)(p + 1) / n);
                        Quaternion edge_p4 = Quaternion.Lerp(init_vectors[i + 1], init_vectors[i + 2], (float)(p + 1) / n);
                        for (int q = 0; q < (n - p); q++) {   
                            Quaternion a = Quaternion.Lerp(edge_p1, edge_p2, (float)q / (n - p));
                            Quaternion b = Quaternion.Lerp(edge_p1, edge_p2, (float)(q + 1) / (n - p));
                            Quaternion c, d;
                            if(edge_p3 == edge_p4) {
                                c = edge_p3;
                                d = edge_p3;
                            } else {
                                c = Quaternion.Lerp(edge_p3, edge_p4, (float)q / (n - p - 1));
                                d = Quaternion.Lerp(edge_p3, edge_p4, (float)(q + 1) / (n - p - 1));
                            }
                            vertices[j++] = new Vector3(a.x, a.y, a.z);
                            vertices[j++] = new Vector3(b.x, b.y, b.z);
                            vertices[j++] = new Vector3(c.x, c.y, c.z);
                            if (q < n - p - 1) {
                                vertices[j++] = new Vector3(c.x, c.y, c.z);
                                vertices[j++] = new Vector3(b.x, b.y, b.z);
                                vertices[j++] = new Vector3(d.x, d.y, d.z);
                            }
                        }
                    }
                }
                for (int i = 0; i < vertexNum; i++) {
                    vertices[i] *= radius;
                }
                return vertices;
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
                for(int i=0; i<Points.Length; i++) {
                    UltiDraw.DrawLine(Pivot.GetPosition(), Points[i], UltiDraw.DarkGreen.Opacity(0.1f));
                    UltiDraw.DrawCircle(Points[i], 0.025f, UltiDraw.Orange.Opacity(0.5f));
                }
                UltiDraw.End();
            }

        }

    }
}
