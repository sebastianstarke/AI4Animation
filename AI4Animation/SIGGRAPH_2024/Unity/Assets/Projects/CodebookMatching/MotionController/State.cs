using UnityEngine;
using AI4Animation;
using System.Collections.Generic;

namespace SIGGRAPH_2024 {
    public class State {
        public State Previous = null;
        public State Next = null;
        public Matrix4x4 Reference; //The reference matrix that this state was queried from.
        public Matrix4x4 Delta; //The matrix that transforms this state from the previous frame.
        public string[] Names; //The bone names of this state.
        public Vector3[] Positions; //The bone positions in local coordinate frame.
        public Quaternion[] Rotations; //The bone rotations in local coordinate frame.
        public Vector3[] Velocities; //The bone velocities in local coordinate frame.
        public float RootLock; //The amount of damping that should be applied to the root delta.
        public float[] Contacts; //The contacts labels for this frame.
        public float Distance; //The distance from the state that this state was queried from.
        public State(State previous, Matrix4x4 reference, Matrix4x4 delta, string[] names, Vector3[] positions, Quaternion[] rotations, Vector3[] velocities, float rootLock, float[] contacts) {
            Previous = previous;
            if(Previous != null) {
                Previous.Next = this;
            }
            Reference = reference;
            Delta = delta;
            Names = names;
            Positions = positions;
            Rotations = rotations;
            Velocities = velocities;
            RootLock = rootLock;
            Contacts = contacts;
        }
        public void ComputeDistance(Actor actor, string[] bones) {
            Matrix4x4 root = actor.GetRoot().GetWorldMatrix();
            Distance = 0f;
            for(int i=0; i<bones.Length; i++) {
                // Distance += Vector3.Distance(actor.GetBonePosition(bones[i]).PositionTo(root), Positions[Names.FindIndex(bones[i])]);
                Distance += Mathf.Pow(Vector3.Distance(actor.GetBonePosition(bones[i]).PositionTo(root), Positions[Names.FindIndex(bones[i])]), 2f);
                // Distance += Quaternion.Angle(actor.GetBoneRotation(bones[i]).RotationTo(root), Rotations[Names.FindIndex(bones[i])]) / 180f;
                // Distance += Vector3.Angle(actor.GetBoneVelocity(bones[i]).DirectionTo(root), Velocities[Names.FindIndex(bones[i])]) / 180f;
            }
        }
        public void DrawSequence(Actor actor, string[] bones, Color color, float opacity) {
            Matrix4x4 root = actor.GetRoot().GetWorldMatrix();
            State current = this;
            while(current.Next != null) {
                current = current.Next;
                root *= Utility.Interpolate(current.Delta, Matrix4x4.identity, RootLock);
                UltiDraw.Begin();
                UltiDraw.DrawTranslateGizmo(root, 0.125f);
                UltiDraw.DrawCircle(root.GetPosition(), 0.1f, UltiDraw.Red.Opacity(RootLock));
                UltiDraw.End();
                actor.Draw(current.GetTransformations().TransformationsFrom(root, false), bones, color.Opacity(opacity), UltiDraw.Black.Opacity(opacity), Actor.DRAW.Skeleton);
            }
        }
        public static void DrawSequence(State[] sequence, Actor actor, string[] bones, Color color, float opacity) {
            Matrix4x4 root = sequence.First().Reference;
            for(int i=0; i<sequence.Length; i++) {
                if(i>0) {
                    root *= Utility.Interpolate(sequence[i].Delta, Matrix4x4.identity, sequence[i].RootLock);
                }
                UltiDraw.Begin();
                UltiDraw.DrawTranslateGizmo(root, 0.125f);
                UltiDraw.DrawCircle(root.GetPosition(), 0.1f, UltiDraw.Red.Opacity(sequence[i].RootLock));
                UltiDraw.End();
                actor.Draw(sequence[i].GetTransformations().TransformationsFrom(root, false), bones, color.Opacity(opacity), UltiDraw.Black.Opacity(opacity), Actor.DRAW.Skeleton);
            }
        }

        public Vector3 GetRolloutPosition(Actor actor, string bone) {
            Matrix4x4 root = actor.GetRoot().GetWorldMatrix();
            State current = this;
            while(current.Next != null) {
                current = current.Next;
                root *= current.Delta;
            }
            return current.Positions[Names.FindIndex(bone)].PositionFrom(root);
        }
        public Vector3 GetRolloutPosition(Actor actor, string bone, int positionLock) {
            Matrix4x4 root = actor.GetRoot().GetWorldMatrix();
            Vector3 p = actor.GetBonePosition(bone);
            int index = Names.FindIndex(bone);
            State current = this;
            while(current.Next != null) {
                current = current.Next;
                root *= Utility.Interpolate(current.Delta, Matrix4x4.identity, current.RootLock);
                p = Vector3.Lerp(current.Positions[index].PositionFrom(root), p, current.Contacts[positionLock]);
            }
            return p;
        }
        public float[] GetContactSequence(int channel) {
            State[] sequence = GetSequence();
            float[] contacts = new float[sequence.Length];
            for(int i=0; i<contacts.Length; i++) {
                contacts[i] = sequence[i].Contacts[channel];
            }
            return contacts;
        }
        public State[] GetSequence() {
            List<State> sequence = new List<State>();
            State pivot = this;
            while(pivot != null) {
                sequence.Add(pivot);
                pivot = pivot.Next;
            }
            return sequence.ToArray();
        }

        private Matrix4x4[] GetTransformations() {
            Matrix4x4[] m = new Matrix4x4[Positions.Length];
            for(int i=0; i<m.Length; i++) {
                m[i] = Matrix4x4.TRS(Positions[i], Rotations[i], Vector3.one);
            }
            return m;
        }
    }
}