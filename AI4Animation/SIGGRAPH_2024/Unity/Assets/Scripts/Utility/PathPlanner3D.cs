#if UNITY_EDITOR
using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
using UnityEditor;
using UnityEditorInternal;

[ExecuteInEditMode]
public class PathPlanner3D : MonoBehaviour {

    public Vector3 Center = new Vector3(0f,0f,0f);
    public Vector3 Size = new Vector3(10f,10f,10f);
    public Vector3Int Resolution = new Vector3Int(10,10,10);
    public LayerMask Mask = ~0;

    public Transform Start;
    public Transform Goal;
    public int MaxDepth = 5;
    public bool ProjectZero = false;

    public bool DrawGeometry = false;
    public bool DrawPath = false;
    public int SplineResolution = 100;
    [Range(0f,1f)] public float SplinePivot = 0f;

    [HideInInspector] public Voxel[] Voxels;

    [Serializable]
    public class Voxel {
        public Vector3Int Coordinates;
        public Vector3 Position;
        public Vector3 Volume;
        public bool Walkable;
        
        [NonSerialized] public Voxel Parent;
        [NonSerialized] public int Depth;
        [NonSerialized] public float Cost;

        public Voxel(Vector3Int coordinates, Vector3 position, Vector3 volume, bool walkable) {
            Coordinates = coordinates;
            Position = position;
            Volume = volume;
            Walkable = walkable;
        }

        public float CuboidToSphereRadius() {
            // Vc = x*y*z
            // Vs = 4/3*pi*r^3
            // Vc=Vs
            // x*y*z = 4/3*pi*r^3
            // x * y * z * 3 / 4 / pi = r^3
            // (x * y * z * 3 / 4 / pi) ^ (1/3) = r
            return Mathf.Pow(Volume.x*Volume.y*Volume.z*(3f/4f)/Mathf.PI, 1f/3f);
        }
    }

    public Path Search(Vector3 start, Vector3 end, int maxSearchDepth) {
        return Path.GetShortestPath(
            this, 
            GetClosestVoxel(start), 
            GetClosestVoxel(end),
            maxSearchDepth,
            (x,y) => Vector3.Distance(x.Position, y.Position),
            (x,y) => x==y
        );
    }

    public Path Search(Vector3 start, Vector3 end, int maxSearchDepth, float distanceToTarget) {
        return Path.GetShortestPath(
            this, 
            GetClosestVoxel(start), 
            GetClosestVoxel(end),
            maxSearchDepth,
            (x,y) => Mathf.Abs(Vector3.Distance(x.Position, y.Position) - distanceToTarget),
            (x,y) => (Vector3.Distance(x.Position, y.Position) >= distanceToTarget - y.CuboidToSphereRadius()) && (Vector3.Distance(x.Position, y.Position) <= distanceToTarget + y.CuboidToSphereRadius())
        );
    }

    private void Generate() {
        Voxels = new Voxel[Resolution.x * Resolution.y * Resolution.z];
        for(int z=0; z<Resolution.z; z++) {
            for(int y=0; y<Resolution.y; y++) {
                for(int x=0; x<Resolution.x; x++) {
                    Vector3Int coordinates = new Vector3Int(x,y,z);
                    Vector3 position = GetPosition(coordinates);
                    bool walkable = !Physics.CheckBox(position, GetVolume()/2f, Quaternion.identity, Mask);
                    Voxels[GetIndex(coordinates)] = new Voxel(coordinates, position, GetVolume(), walkable);
                }
            }
        }
    }

    private int GetIndex(Vector3Int coordinates) {
        return coordinates.z*Resolution.y*Resolution.x + coordinates.y*Resolution.x + coordinates.x;
    }

    private Voxel GetVoxel(Vector3Int coordinates) {
        coordinates.x = Mathf.Clamp(coordinates.x, 0, Resolution.x-1);
        coordinates.y = Mathf.Clamp(coordinates.y, 0, Resolution.y-1);
        coordinates.z = Mathf.Clamp(coordinates.z, 0, Resolution.z-1);
        return Voxels[GetIndex(coordinates)];
    }

    private Voxel GetClosestVoxel(Vector3 position) {
        position.x = Mathf.Clamp(position.x, Center.x - Size.x/2f, Center.x + Size.x/2f);
        position.y = Mathf.Clamp(position.y, Center.y - Size.y/2f, Center.y + Size.y/2f);
        position.z = Mathf.Clamp(position.z, Center.z - Size.z/2f, Center.z + Size.z/2f);
        position.x = position.x.Normalize(Center.x - Size.x/2f, Center.x + Size.x/2f, 0f, 1f);
        position.y = position.y.Normalize(Center.y - Size.y/2f, Center.y + Size.y/2f, 0f, 1f);
        position.z = position.z.Normalize(Center.z - Size.z/2f, Center.z + Size.z/2f, 0f, 1f);
        Vector3Int coordinates = new Vector3Int(
            Mathf.RoundToInt(position.x * (Resolution.x-1)),
            Mathf.RoundToInt(position.y * (Resolution.y-1)),
            Mathf.RoundToInt(position.z * (Resolution.z-1))
        );
        return GetVoxel(coordinates);
    }

    private Voxel[] Neighbors = new Voxel[26];
    private Voxel[] GetNeighbors(Voxel voxel) {
        int index = 0;
        for(int z=-1; z<=1; z++) {
            for(int y=-1; y<=1; y++) {
                for(int x=-1; x<=1; x++) {
                    if(x==0 && y==0 && z==0) {
                        continue;
                    }
                    Vector3Int coordinates = voxel.Coordinates + new Vector3Int(x,y,z);
                    if(coordinates.x < 0 || coordinates.x >= Resolution.x) {Neighbors[index] = null;}
                    if(coordinates.y < 0 || coordinates.y >= Resolution.y) {Neighbors[index] = null;}
                    if(coordinates.z < 0 || coordinates.z >= Resolution.z) {Neighbors[index] = null;}
                    Voxel v = GetVoxel(coordinates);
                    Neighbors[index] = v.Walkable ? v : null;
                    index += 1;
                }
            }
        }
        return Neighbors;
    }

    private Vector3 GetPosition(Vector3Int coordinates) {
        Vector3 volume = GetVolume();
        return new Vector3(
            coordinates.x.Ratio(0, Resolution.x-1).Normalize(0f, 1f, Center.x - Size.x/2f + volume.x/2f, Center.x + Size.x/2f - volume.x/2f),
            coordinates.y.Ratio(0, Resolution.y-1).Normalize(0f, 1f, Center.y - Size.y/2f + volume.y/2f, Center.y + Size.y/2f - volume.y/2f),
            coordinates.z.Ratio(0, Resolution.z-1).Normalize(0f, 1f, Center.z - Size.z/2f + volume.z/2f, Center.z + Size.z/2f - volume.z/2f)
        );
    }

    private Vector3 GetVolume() {
        return new Vector3(
            Size.x/Resolution.x,
            Size.y/Resolution.y,
            Size.z/Resolution.z
        );
    }

    void OnRenderObject() {
        if(DrawGeometry) {
            UltiDraw.Begin();
            UltiDraw.SetDepthRendering(true);
            foreach(Voxel voxel in Voxels) {
                if(!voxel.Walkable) {
                    UltiDraw.DrawCuboid(voxel.Position, Quaternion.identity, voxel.Volume, UltiDraw.White.Opacity(0.5f));
                }
            }
            UltiDraw.DrawWireCuboid(Center, Quaternion.identity, Size, UltiDraw.Cyan);
            UltiDraw.SetDepthRendering(false);
            UltiDraw.End();
        }

        if(DrawPath) {
            Path path = Search(Start.position, Goal.position, MaxDepth);
            path.Draw();
            UltiDraw.Begin();
            for(int i=0; i<SplineResolution; i++) {
                UltiDraw.DrawSphere(path.GetPathPoint(i.Ratio(0, SplineResolution-1), SplineResolution), 0.05f, Color.magenta);
            }
            UltiDraw.DrawSphere(path.GetPathPoint(SplinePivot, SplineResolution), 0.1f, Color.cyan);
            UltiDraw.End();
        }
    }

    #if UNITY_EDITOR
    [CustomEditor(typeof(PathPlanner3D))]
    public class Inspector : Editor {
        public override void OnInspectorGUI() {
            PathPlanner3D instance = (PathPlanner3D)target;

            Utility.SetGUIColor(UltiDraw.LightGrey);
            using(new EditorGUILayout.VerticalScope ("Box")) {
                Utility.ResetGUIColor();
                instance.Center = EditorGUILayout.Vector3Field("Center", instance.Center);
                instance.Size = EditorGUILayout.Vector3Field("Size", instance.Size);
                instance.Resolution = EditorGUILayout.Vector3IntField("Resolution", instance.Resolution);
                instance.Mask = InternalEditorUtility.ConcatenatedLayersMaskToLayerMask(EditorGUILayout.MaskField(
                    "Collision Mask", 
                    InternalEditorUtility.LayerMaskToConcatenatedLayersMask(instance.Mask), 
                    InternalEditorUtility.layers)
                );

                instance.Start = EditorGUILayout.ObjectField("Start", instance.Start, typeof(Transform), true) as Transform;
                instance.Goal = EditorGUILayout.ObjectField("Goal", instance.Goal, typeof(Transform), true) as Transform;
                instance.MaxDepth = EditorGUILayout.IntField("Max Depth", instance.MaxDepth);
                instance.ProjectZero = EditorGUILayout.Toggle("Project Zero", instance.ProjectZero);

                if(Utility.GUIButton("Generate", UltiDraw.DarkGrey, UltiDraw.White)) {
                    instance.Generate();
                }

                instance.DrawGeometry = EditorGUILayout.Toggle("Draw Geometry", instance.DrawGeometry);
                instance.DrawPath = EditorGUILayout.Toggle("Draw Path", instance.DrawPath);
                instance.SplineResolution = EditorGUILayout.IntField("Spline Resolution", instance.SplineResolution);
                instance.SplinePivot = EditorGUILayout.Slider("Spline Pivot", instance.SplinePivot, 0f, 1f);
            }
            if(GUI.changed) {
                SceneView.RepaintAll();
            }
        }
    }

    public class Path {
        public Vector3[] Points;
        
        public static HashSet<Voxel> History = new HashSet<Voxel>();

        public Path(params Vector3[] points) {
            Points = points;
        }

        public void Draw(bool history=false) {
            UltiDraw.Begin();
            UltiDraw.DrawSphere(Points.First(), Quaternion.identity, 0.25f, UltiDraw.Magenta);
            UltiDraw.DrawSphere(Points.Last(), Quaternion.identity, 0.25f, UltiDraw.Magenta);
            for(int i=1; i<Points.Length; i++) {
                UltiDraw.DrawArrow(Points[i-1], Points[i], 0.75f, 0.025f, 0.25f, UltiDraw.Green);
            }
            if(history) {
                foreach(Voxel v in History) {
                    UltiDraw.DrawCuboid(v.Position, Quaternion.identity, v.Volume, UltiDraw.Black.Opacity(0.5f));
                }
            }
            UltiDraw.End();
        }

        public Matrix4x4 GetPathPoint(float percentage, int resolution) {
            float step = 1f/(resolution-1);
            Vector3 pos = UltiMath.UltiMath.GetPointOnSpline(Points, percentage);
            Quaternion rot = Quaternion.LookRotation((UltiMath.UltiMath.GetPointOnSpline(Points, percentage + step) - UltiMath.UltiMath.GetPointOnSpline(Points, percentage)).normalized, Vector3.up);
            return Matrix4x4.TRS(pos, rot, Vector3.one);
        }

        public static float ManhattanDistance(Voxel a, Voxel b) {
            return Mathf.Abs(b.Coordinates.x - a.Coordinates.x) + Mathf.Abs(b.Coordinates.y - a.Coordinates.y) + Mathf.Abs(b.Coordinates.z - a.Coordinates.z);
        }

        //Searches a path from source to target that minimizes a heuristic cost and has a maximum search depth
        public static Path GetShortestPath(PathPlanner3D planner, Voxel source, Voxel target, int maxSearchDepth, Func<Voxel, Voxel, float> cost, Func<Voxel, Voxel, bool> termination) {
            History = new HashSet<Voxel>();

            Voxel best = source;
            best.Parent = null;
            best.Depth = 0;
            best.Cost = source.Walkable ? cost(source, target) : float.MaxValue;

            Collections.PriorityQueue<Voxel, float> candidates = new Collections.PriorityQueue<Voxel, float>();
            candidates.Enqueue(best, best.Cost);
            while(candidates.Count > 0) {
                Voxel current = candidates.Dequeue();
                History.Add(current);

                //Evaluate current voxel
                if(current.Cost < best.Cost) {
                    best = current;
                }

                //Check termination
                if(termination(best, target)) {
                    break;
                }

                //Ignore successors that exceed depth limit
                if(current.Depth == maxSearchDepth) {
                    continue;
                }

                //Reduce search time for blocked targets or targest that are out of range
                if(!target.Walkable) {
                    //Ignore successors that are unlikely to improve further search
                    if(current.Depth < best.Depth && current.Cost > best.Cost) {
                        continue;
                    }
                    //Ignore successors that are not greedy and would not be able to reach the target anymore
                    if(current != best && current.Depth + ManhattanDistance(current, target) > maxSearchDepth) {
                        continue;
                    }
                }
                
                //Expand path search
                foreach(Voxel n in planner.GetNeighbors(current)) {
                    if(n != null && !History.Contains(n) && !candidates.UnorderedItems.Any(x => x.Element == n)) {
                        n.Parent = current;
                        n.Depth = n.Parent.Depth + 1;
                        n.Cost = cost(n, target);
                        candidates.Enqueue(n, n.Cost);
                    }
                }
            }

            List<Vector3> path = new List<Vector3>();
            Voxel pivot = best;
            while(pivot != null) {
                path.Add(pivot.Position);
                pivot = pivot.Parent;
            }
            path.Add(planner.Start.position);
            path.Reverse();
            path.Add(planner.Goal.position);
            if(planner.ProjectZero) {
                for(int i=0; i<path.Count; i++) {
                    path[i] = path[i].ZeroY();
                }
            }
            return new Path(path.ToArray());
        }
    }
    #endif
}
#endif