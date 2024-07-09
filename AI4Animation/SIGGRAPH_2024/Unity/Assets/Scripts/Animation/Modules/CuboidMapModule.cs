using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
using UnityEditorInternal;
#endif
using UnityEngine.SceneManagement;

namespace AI4Animation {
    public class CuboidMapModule : Module {
        public string BoneName;
        public Vector3 Offset = Vector3.zero;
        public Vector3 Size = Vector3.one;
        public Vector3Int Resolution = new Vector3Int(10, 10, 10);
        public LayerMask Mask = ~0;
        public Color Color = UltiDraw.Cyan;
        public bool RandomNoiseSampling = true;
        public bool DrawSensor = true;
        public bool DrawReferences = false;
        public bool DrawClosestPoints = false;
        public bool DrawEndEffectors = false;
        public bool DrawDistribution = false;

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

        //Drawing
        protected override void DerivedDraw(MotionEditor editor) {
            Random.InitState(editor.GetCurrentFrame().Index);
            UltiDraw.SetDepthRendering(true);
            //HOM
            // CuboidMap sensor = GetCuboidMapFromScene(editor.GetSession().Asset.GetScene(), Mask, Resolution, Size, Offset);

            Matrix4x4 pivot;
            if(BoneName == "body_world") {
                pivot = Matrix4x4.TRS(
                    Asset.GetModule<RootModule>("Hips").GetRootTransformation(editor.GetTimestamp(), editor.Mirror).GetPosition(),
                    Asset.GetModule<RootModule>("Hips").GetRootTransformation(editor.GetTimestamp(), editor.Mirror).GetRotation(),
                    Vector3.one
                );
            }
            else {
                pivot = Matrix4x4.TRS(
                    Asset.GetFrame(editor.GetTimestamp()).GetBoneTransformation(Asset.Source.FindBone(BoneName).Index, editor.Mirror).GetPosition(),
                    Asset.GetFrame(editor.GetTimestamp()).GetBoneTransformation(Asset.Source.FindBone(BoneName).Index, editor.Mirror).GetRotation(),
                    Vector3.one
                );
            }

            CuboidMap sensor = GetCuboidMap(pivot, Mask, Resolution, Size, Offset);
            if(RandomNoiseSampling) {
                sensor.SetRandomOccupacion();
            }

            if(DrawEndEffectors) {
                Matrix4x4[] endEffectors = Asset.GetFrame(editor.GetTimestamp()).GetBoneTransformations(editor.GetSession().GetActor().GetBoneNames(), editor.Mirror);
                Vector3[] ee = new Vector3[endEffectors.Length];
                for (int i = 0; i < ee.Length; i++)
                {
                    ee[i] = endEffectors[i].GetPosition();
                }
                Vector3[] neighbors = sensor.GetClosestOccupancyPoints(ee);
                UltiDraw.Begin();
                for (int i = 0; i < neighbors.Length; i++)
                {
                    Color color = UltiDraw.GetRainbowColor(i, neighbors.Length);
                    
                    Vector3 point = Vector3.Lerp(ee[i], neighbors[i], Random.value);
                    UltiDraw.DrawLine(ee[i], neighbors[i], 0.01f, color);
                    UltiDraw.DrawSphere(point, endEffectors[i].GetRotation(), 0.1f, color);
                }
                UltiDraw.End();
            }

            if(DrawSensor) {
                sensor.Draw(Color);
            }
            if(DrawReferences) {
                sensor.DrawReferences();
            }
            if(DrawClosestPoints) {
                sensor.DrawClosestPoints();
            }

            if(DrawDistribution) {
                sensor.DrawDistribution(UltiDraw.Black, new UltiDraw.GUIRect(0f, 0f, 500f, 500f));
            }
            UltiDraw.SetDepthRendering(false);
        }

        protected override void DerivedInspector(MotionEditor editor) {
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Bone", GUILayout.Width(40f));
            MotionAsset.Hierarchy.Bone bone = Asset.Source.FindBone(BoneName);
            int index = bone == null ? 0 : bone.Index;
            BoneName = Asset.Source.Bones[EditorGUILayout.Popup(index, Asset.Source.GetBoneNames(), GUILayout.Width(150f))].GetName();
            EditorGUILayout.EndHorizontal();
            Offset = EditorGUILayout.Vector3Field("Offset", Offset);
            Size = EditorGUILayout.Vector3Field("Size", Size);
            Resolution = EditorGUILayout.Vector3IntField("Resolution", Resolution);
            Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField("Mask", InternalEditorUtility.LayerMaskToConcatenatedLayersMask(Mask), InternalEditorUtility.layers));
            Color = EditorGUILayout.ColorField("Color", Color);

            RandomNoiseSampling = EditorGUILayout.Toggle("Sample Random Noise", RandomNoiseSampling);
            DrawSensor = EditorGUILayout.Toggle("Draw Sensor", DrawSensor);
            DrawReferences = EditorGUILayout.Toggle("Draw References", DrawReferences);
            DrawClosestPoints = EditorGUILayout.Toggle("Draw Closest Points", DrawClosestPoints);
            DrawEndEffectors = EditorGUILayout.Toggle("Draw Noised Closest Points", DrawEndEffectors);
            DrawDistribution = EditorGUILayout.Toggle("Draw Distribution", DrawDistribution);
        }
#endif
        public CuboidMap GetCuboidMapFromScene(Scene scene, LayerMask mask, Vector3Int resolution, Vector3 size, Vector3 offset) {
            //Scene scene = editor.GetSession().Asset.GetScene();

            GameObject[] sceneObjects = scene.GetRootGameObjects();
            GameObject root = null;
            if(sceneObjects.Length > 0){
                root = sceneObjects[0];
            }

            if(root != null) {
                Transform obj = root.transform.GetChild(0);
                if(obj == null){
                    return null;
                } 
                return GetCuboidMap(obj.GetWorldMatrix(), mask, resolution, size, offset);
            }
            return null;
        }

        public CuboidMap GetCuboidMap(Matrix4x4 pivot, LayerMask mask, Vector3Int resolution, Vector3 size, Vector3 offset) {
            // if(Sensor == null || Sensor.Resolution != resolution) {
            // 	Sensor = new CuboidMap(resolution);
            // }
            CuboidMap sensor = new CuboidMap(resolution);
            Matrix4x4 location = Matrix4x4.TRS(pivot.GetPosition() + pivot.GetRotation() * offset, pivot.GetRotation(), Vector3.one);
            sensor.Sense(location, mask, size);
            return sensor;
        }
        
        public class CuboidMap {

            public Matrix4x4 Pivot = Matrix4x4.identity;
            public Vector3[] LocalPoints = new Vector3[0];

            public Vector3[] WorldPoints = new Vector3[0];

            public float[] Occupancies = new float[0];
            public Vector3[] ClosestPoints = new Vector3[0];

            public Vector3Int Resolution = new Vector3Int(10, 10, 10);
            public Vector3 Size = Vector3.one;

            public CuboidMap(Vector3Int resolution) {
                Size = Vector3.zero;
                Resolution = resolution;
                Generate();
            }

            private Vector3 GetStep() {
                return new Vector3(Size.x / Resolution.x, Size.y / Resolution.y, Size.z / Resolution.z);
            }

            private int GetDimensionality() {
                return Resolution.x * Resolution.y * Resolution.z;
            }

            public void Generate() {
                LocalPoints = new Vector3[GetDimensionality()];
                WorldPoints = new Vector3[GetDimensionality()];
                Occupancies = new float[GetDimensionality()];
                ClosestPoints = new Vector3[GetDimensionality()];
                for(int y=0; y<Resolution.y; y++) {
                    for(int x=0; x<Resolution.x; x++) {
                        for(int z=0; z<Resolution.z; z++) {
                            LocalPoints[y*Resolution.x*Resolution.z + x*Resolution.z + z] = new Vector3(
                                -0.5f + (x+0.5f)/Resolution.x,
                                -0.5f + (y+0.5f)/Resolution.y,
                                -0.5f + (z+0.5f)/Resolution.z
                            );
                        }
                    }
                }
            }

            public void Sense(Matrix4x4 pivot, LayerMask mask, Vector3 size) {
                Pivot = pivot;
                Size = size;

                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                Vector3 step = GetStep();
                float range = Mathf.Max(step.x, step.y, step.z);
                for(int i=0; i<LocalPoints.Length; i++) {
                    WorldPoints[i] = position + rotation * Vector3.Scale(LocalPoints[i], Size);
                    Collider c;
                    ClosestPoints[i] = Utility.GetClosestPointOverlapBox(WorldPoints[i], step/2f, rotation, mask, out c);
                    Occupancies[i] = c == null ? 0f : (1f - Vector3.Distance(WorldPoints[i], ClosestPoints[i]) / range);
                }
            }

            public void SetRandomOccupacion() {
                for(int i=0; i<Occupancies.Length; i++) {
                    if(Occupancies[i] == 0f) {
                        //if(Random.value > (1f-probability)) {
                        Occupancies[i] = Random.value;
                        //}
                    } else { 
                        Occupancies[i] = 0f;
                    }
                }
            }

            public Vector3[] GetClosestOccupancyPoints(Vector3[] pivots, float probability = 1f){
                if(Random.value > probability) {
                    return pivots;
                }

                Vector3[] points = new Vector3[pivots.Length];
                float[] distances = new float[pivots.Length];
                for (int i = 0; i < distances.Length; i++)
                {
                    distances[i] = float.MaxValue;
                }

                for(int i=0; i<Occupancies.Length; i++) {
                    //only consider occupied points
                    if(Occupancies[i] == 0f){ continue; }

                    for (int j = 0; j < pivots.Length; j++)
                    {
                        float dstNew = Vector3.Distance(pivots[j], WorldPoints[i]);
                        float dstOld = distances[j];
                        if(dstNew < dstOld){
                            //points[j] = Vector3.Lerp(pivots[j], Worldpoints[i], Random.value);
                            points[j] = WorldPoints[i];
                            distances[j] = dstNew;
                        }
                    }	
                }	
                return points;
            }

            // public void Retransform(Matrix4x4 pivot) {
            // 	Pivot = pivot;
            // 	Vector3 position = Pivot.GetPosition();
            // 	Quaternion rotation = Pivot.GetRotation();
            // 	for(int i=0; i<Worldpoints.Length; i++) {
            // 		Worldpoints[i] = position + rotation * Vector3.Scale(Localpoints[i], Size);
            // 	}
            // }

            public void Draw(Color color) {
                Vector3 position = Pivot.GetPosition();
                Quaternion rotation = Pivot.GetRotation();
                UltiDraw.Begin();
                Vector3 step = GetStep();
                UltiDraw.DrawWireCuboid(position, rotation, Size, color);
                for(int i=0; i<Occupancies.Length; i++) {
                    if(Occupancies[i] > 0f) {
                        UltiDraw.DrawCuboid(WorldPoints[i], rotation, step, Color.Lerp(UltiDraw.Transparent, color, Occupancies[i]));
                    }
                }
                UltiDraw.End();
            }

            public void DrawReferences() {
                UltiDraw.Begin();
                Vector3 step = GetStep();
                for(int i=0; i<WorldPoints.Length; i++) {
                    UltiDraw.DrawSphere(WorldPoints[i], Quaternion.identity, step.magnitude/10f, UltiDraw.Black);
                }
                UltiDraw.End();
            }

            public void DrawClosestPoints() {
                UltiDraw.Begin();
                Vector3 step = GetStep();
                for(int i=0; i<ClosestPoints.Length; i++) {
                    UltiDraw.DrawSphere(ClosestPoints[i], Quaternion.identity, step.magnitude/10f, UltiDraw.Red);
                }
                UltiDraw.End();
            }

            public void DrawDistribution(Color color, UltiDraw.GUIRect rect) {
                UltiDraw.Begin();
                UltiDraw.PlotFunction(rect.GetCenter(), rect.GetSize(), Occupancies, yMin: 0f, yMax: 1f, backgroundColor: UltiDraw.White, lineColor: color);
                UltiDraw.End();
            }

        }

    }
}
