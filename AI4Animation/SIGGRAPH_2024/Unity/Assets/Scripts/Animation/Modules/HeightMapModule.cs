using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
namespace AI4Animation {
    public class HeightMapModule : Module {
        public float Size = 1f;
        public int Resolution = 25;
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
            HeightMap sensor = GetHeightMap(editor.GetSession().GetActor());
            sensor.Draw();
            sensor.Render(new Vector2(0.1f, 0.25f), new Vector2(0.3f*Screen.height/Screen.width, 0.3f), Resolution, Resolution, 1f);
        }

        protected override void DerivedInspector(MotionEditor editor) {
            Size = EditorGUILayout.FloatField("Size", Size);
            Resolution = EditorGUILayout.IntField("Resolution", Resolution);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
        }
#endif

        public HeightMap GetHeightMap(Actor actor) {
            HeightMap sensor = new HeightMap(Size, Resolution, Mask);
            sensor.Sense(actor.GetRoot().GetGlobalMatrix());
            return sensor;
        }

        public class HeightMap {

            public Matrix4x4 Pivot = Matrix4x4.identity;

            public Vector3[] Map = new Vector3[0];
            public Vector3[] Points = new Vector3[0];

            public float Size = 1f;
            public int Resolution = 25;
            public LayerMask Mask = -1;

            public HeightMap(float size, int resolution, LayerMask mask) {
                Size = size;
                Resolution = resolution;
                Mask = mask;
                Generate();
            }

            private void Generate() {
                Map = new Vector3[Resolution*Resolution];
                Points = new Vector3[Resolution*Resolution];
                for(int x=0; x<Resolution; x++) {
                    for(int y=0; y<Resolution; y++) {
                        Map[y*Resolution + x] = new Vector3(-Size/2f + (float)x/(float)(Resolution-1)*Size, 0f, -Size/2f + (float)y/(float)(Resolution-1)*Size);
                    }
                }
            }

            public void SetSize(float value) {
                if(Size != value) {
                    Size = value;
                    Generate();
                }
            }

            public void SetResolution(int value) {
                if(Resolution != value) {
                    Resolution = value;
                    Generate();
                }
            }

            public void Sense(Matrix4x4 pivot) {
                Pivot = pivot;
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Quaternion.AngleAxis(Pivot.GetRotation().eulerAngles.y, Vector3.up);
                for(int i=0; i<Map.Length; i++) {
                    Points[i] = Project(position + rotation * Map[i]);
                }
            }

            public float[] GetHeights() {
                float[] heights = new float[Points.Length];
                for(int i=0; i<heights.Length; i++) {
                    heights[i] = Points[i].y;
                }
                return heights;
            }

            public float[] GetHeights(float maxHeight) {
                float[] heights = new float[Points.Length];
                for(int i=0; i<heights.Length; i++) {
                    heights[i] = Mathf.Clamp(Points[i].y, 0f, maxHeight);
                }
                return heights;
            }

            private Vector3 Project(Vector3 position) {
                RaycastHit hit;
                Physics.Raycast(new Vector3(position.x, 100f, position.z), Vector3.down, out hit, float.PositiveInfinity, Mask);
                position = hit.point;
                return position;
            }

            public void Draw(float[] mean=null, float[] std=null) {
                //return;
                UltiDraw.Begin();

                //Quaternion rotation = Pivot.GetRotation() * Quaternion.Euler(90f, 0f, 0f);
                Color color = UltiDraw.IndianRed.Opacity(0.5f);
                //float area = (float)Size/(float)(Resolution-1);
                for(int i=0; i<Points.Length; i++) {
                    UltiDraw.DrawCircle(Points[i], 0.025f, color);
                    //UltiDraw.DrawQuad(Points[i], rotation, area, area, color);
                }

                UltiDraw.End();
            }
            
            public void Render(Vector2 center, Vector2 size, int width, int height, float maxHeight) {
                UltiDraw.Begin();
                UltiDraw.PlotGreyscaleImage(center, size, width, height, GetHeights(maxHeight));
                UltiDraw.End();
            }
        }
    }
}
