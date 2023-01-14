#if UNITY_EDITOR
using UnityEngine;

namespace AI4Animation {
    public class VelocityPlot : MonoBehaviour {

        public float ArrowSize = 0.1f;
        public float SphereSize = 0.1f;
        public Vector3 Scale = Vector3.one;
        public bool Temporal = false;
        [Range(0f,1f)] public float Opacity = 1f;

        public Group[] Groups;

        void OnRenderObject() {
            UltiDraw.Begin();

            for(int i=0; i<Groups.Length; i++) {
                if(Groups[i].Visualise) {
                    if(Groups[i].Asset == null) {
                        continue;
                    }
                    if(Groups[i].Colors.Length != Groups[i].Bones.Length) {
                        continue;
                    }
                    Groups[i].Interval.Start = Mathf.Max(Groups[i].Interval.Start, 1);
                    Groups[i].Interval.End = Mathf.Min(Groups[i].Interval.End, Groups[i].Asset.GetTotalFrames());
                    RootModule module = Groups[i].Asset.GetModule<RootModule>();
                    Frame[] frames = Groups[i].Asset.GetFrames(Groups[i].Interval.Start, Groups[i].Interval.End);
                    int[] indices = Groups[i].Asset.Source.GetBoneIndices(Groups[i].Bones);
                    Vector3[] previous = new Vector3[indices.Length];
                    for(int f=0; f<frames.Length; f++) {
                        Matrix4x4 root = module.GetRootTransformation(frames[f].Timestamp, false);
                        for(int j=0; j<indices.Length; j++) {
                            if(Temporal) {
                                float weight = (float)(f+1)/(float)(frames.Length);
                                Vector3 velocity = Vector3.Scale(Scale, frames[f].GetBoneVelocity(indices[j], false).DirectionTo(root));
                                if(f != 0) {
                                    UltiDraw.DrawArrow(previous[j], velocity, 0.75f, 0.1f*ArrowSize, ArrowSize, Groups[i].Colors[j].Opacity(Opacity).Darken(weight));
                                    previous[j] = velocity;
                                }
                                UltiDraw.DrawSphere(velocity, Quaternion.identity, SphereSize, Groups[i].Colors[j].Opacity(Opacity).Darken(weight));
                            } else {
                                Vector3 velocity = Vector3.Scale(Scale, frames[f].GetBoneVelocity(indices[j], false).DirectionTo(root));
                                if(f != 0) {
                                    UltiDraw.DrawArrow(previous[j], velocity, 0.75f, 0.1f*ArrowSize, ArrowSize, Groups[i].Colors[j].Opacity(Opacity));
                                    previous[j] = velocity;
                                }
                                UltiDraw.DrawSphere(velocity, Quaternion.identity, SphereSize, Groups[i].Colors[j].Opacity(Opacity));
                            }
                        }
                    }
                }
            }

            UltiDraw.End();
        }

        [System.Serializable]
        public class Group {
            public string ID;
            public bool Visualise = true;
            public MotionAsset Asset;
            public Interval Interval;
            public Color[] Colors;
            public string[] Bones;
        }

    }
}
#endif