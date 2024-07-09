using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace AI4Animation {
    public class MotionSearchDatabase : SKDTree.KDTree<Value> {
        public string[] Bones;
        public string[] Labels;
    }

    [Serializable]
    public class Value {

        [SerializeReference] public MotionAsset Asset;
        public float Timestamp;
        public bool Mirrored;

        [SerializeReference] public Value Previous;
        [SerializeReference] public Value Next;

        public Value(Value previous, MotionAsset asset, float timestamp, bool mirrored) {
            if(previous != null) {
                Previous = previous;
                Previous.Next = this;
            }

            Asset = asset;
            Timestamp = timestamp;
            Mirrored = mirrored;
        }

        public static float[] GenerateKey(MotionSearchModule.State[] states) {
            List<float> features = new List<float>();

            Matrix4x4 root = states.First().Root;
            foreach(MotionSearchModule.State state in states) {
                for(int i=0; i<state.Transformations.Length; i++) {
                    Matrix4x4 m = state.Transformations[i].TransformationTo(root);
                    features.AddRange(m.GetPosition().ToArray());
                    // features.AddRange(m.GetRotation().ToArray());
                }
                // for(int i=0; i<state.Velocities.Length; i++) {
                //     Vector3 v = state.Velocities[i].DirectionTo(root);
                //     features.AddRange(v.ToArray());
                // }
            }

            return features.ToArray();
        }

        public static string[] GenerateLabels(int states, int bones) {
            List<string> labels = new List<string>();

            for(int s=0; s<states; s++) {
                // string state = "State"+(s+1).ToString();
                for(int i=0; i<bones; i++) {
                    string bone = "Bone"+(i+1).ToString();
                    labels.Add(bone + "Position" + "X");
                    labels.Add(bone + "Position" + "Y");
                    labels.Add(bone + "Position" + "Z");

                    // labels.Add(state + bone + "Position" + "X");
                    // labels.Add(state + bone + "Position" + "Y");
                    // labels.Add(state + bone + "Position" + "Z");

                    // labels.Add(state + bone + "Rotation" + "X");
                    // labels.Add(state + bone + "Rotation" + "Y");
                    // labels.Add(state + bone + "Rotation" + "Z");
                    // labels.Add(state + bone + "Rotation" + "W");
                }
                // for(int i=0; i<bones; i++) {
                //     string bone = "Bone"+(i+1).ToString();
                //     labels.Add(state + bone + "Velocity" + "X");
                //     labels.Add(state + bone + "Velocity" + "Y");
                //     labels.Add(state + bone + "Velocity" + "Z");
                // }
            }

            return labels.ToArray();
        }

        public Matrix4x4 GetRoot() {
            RootModule module = Asset.GetModule<RootModule>("Hips");
            return module.GetRootTransformation(Timestamp, Mirrored);
        }
        
        public Matrix4x4 GetDelta(float deltaTime) {
            RootModule module = Asset.GetModule<RootModule>("Hips");
            return module.GetRootTransformation(Timestamp, Mirrored).TransformationTo(module.GetRootTransformation(Timestamp-deltaTime, Mirrored));
        }
        
        public Matrix4x4[] GetTransformations(Actor actor) {
            return Asset.GetFrame(Timestamp).GetBoneTransformations(actor.GetBoneNames(), Mirrored);
        }

        public Vector3[] GetVelocities(Actor actor) {
            return Asset.GetFrame(Timestamp).GetBoneVelocities(actor.GetBoneNames(), Mirrored);
        }

        public Matrix4x4[] GetTransformations(int[] bones) {
            return Asset.GetFrame(Timestamp).GetBoneTransformations(bones, Mirrored);
        }

        public Vector3[] GetVelocities(int[] bones) {
            return Asset.GetFrame(Timestamp).GetBoneVelocities(bones, Mirrored);
        }

        public float GetRootLock() {
            RootModule module = Asset.GetModule<RootModule>("Hips");
            return module.GetRootLock(Timestamp, Mirrored);
        }

        public float[] GetContacts() {
            ContactModule module = Asset.GetModule<ContactModule>();
            return module.GetContacts(Timestamp, Mirrored);
        }
    }
}
