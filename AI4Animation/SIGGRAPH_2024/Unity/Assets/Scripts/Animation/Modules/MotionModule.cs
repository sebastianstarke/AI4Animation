using System;
using System.Collections.Generic;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AI4Animation {
    public class MotionModule : Module {
        public override TimeSeries.Component DerivedExtractSeries(TimeSeries global, float timestamp, bool mirrored, params object[] parameters) {
            Series instance = new Series(global);
            int[] bones = new int[parameters.Length];
            if(parameters.Length > 0) {
                bones = Asset.Source.GetBoneIndices((string[])parameters);
            }
            foreach(int bone in bones) {
                instance.AddTrajectory(Asset.Source.Bones[bone].GetName());
            }
            for(int i=0; i<instance.Samples.Length; i++) {
                for(int j=0; j<instance.Trajectories.Length; j++) {
                    instance.Trajectories[j].Transformations[i] = GetBoneTransformation(timestamp + instance.Samples[i].Timestamp, mirrored, bones[j]);
                    instance.Trajectories[j].Velocities[i] = GetBoneVelocity(timestamp + instance.Samples[i].Timestamp, mirrored, bones[j]);
                }
            }
            return instance;
        }

		public Matrix4x4 GetBoneTransformation(float timestamp, bool mirrored, int bone, TimeSeries smoothing=null) {
            if(smoothing == null) {
			    return Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
            }
            Matrix4x4 pivot = Asset.GetFrame(timestamp).GetBoneTransformation(bone, mirrored);
            Matrix4x4[] values = new Matrix4x4[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
                values[i] = Asset.GetFrame(timestamp + smoothing.GetKey(i).Timestamp).GetBoneTransformation(bone, mirrored).TransformationTo(pivot);
            }
            Matrix4x4 value = values.Gaussian().TransformationFrom(pivot);
            return value;
		}

		public Vector3 GetBoneVelocity(float timestamp, bool mirrored, int bone, TimeSeries smoothing=null) {
            if(smoothing == null) {
			    return Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
            }
            Vector3 pivot = Asset.GetFrame(timestamp).GetBoneVelocity(bone, mirrored);
            Vector3[] values = new Vector3[smoothing.KeyCount];
            for(int i=0; i<values.Length; i++) {
                values[i] = Asset.GetFrame(timestamp + smoothing.GetKey(i).Timestamp).GetBoneVelocity(bone, mirrored) - pivot;
            }
            Vector3 value = pivot + values.Gaussian();
            return value;
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
            // DerivedExtractSeries(editor.GetTimeSeries(), editor.GetTimestamp(), editor.Mirror, new string[]{"b_root", "b_neck0"}).Draw();
        }

        protected override void DerivedInspector(MotionEditor editor) {
            // EditorGUILayout.LabelField(Skeleton == null ? "No Skeleton!" : "Skeleton: " + Skeleton.ToString());

			// for(int i=0; i<DefaultBoneNames.Length; i++) {
            //     EditorGUILayout.BeginHorizontal();
            //     EditorGUILayout.LabelField("Bone", GUILayout.Width(40f));
            //     DefaultBoneNames[i] = Asset.Source.Bones[(EditorGUILayout.Popup(Asset.Source.FindBone(DefaultBoneNames[i]).Index, Asset.Source.GetBoneNames(), GUILayout.Width(150f)))].GetName();
                
				// if(Utility.GUIButton("-", UltiDraw.DarkRed, UltiDraw.White, 28f, 18f)) {
                //     if(!ArrayExtensions.Remove(ref DefaultBoneNames, DefaultBoneNames[i])) {
                //         Debug.Log("Bone name could not be found in " + Asset.name + ".");
                //     }
				// }
                // EditorGUILayout.EndHorizontal();
			// }
			// if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
			// 	AddDefaultBone(Asset.Source.Bones[0].GetName());
			// }
            // for (int i = 0; i < Bones.Length; i++)
            // {
            //     EditorGUILayout.LabelField("Bone " + Bones[i] + " : " + Asset.Source.Bones[Bones[i]].GetName());
            // }
        }

        // private void VerifyBones(string[] names) {

        //     List<int> bones = new List<int>();
        //     void TryAdd(string name) {
        //         if(Asset.Source.HasBone(name)) {
        //             MotionAsset.Hierarchy.Bone b = Asset.Source.FindBone(name);
        //             if(!b.Name.Equals(name)) {
        //                 Debug.Log("Searched for name " + name + ", but found name is " + b.Name);
        //             }
        //             bones.Add(b.Index);
        //         }
        //     }
        //     for (int i = 0; i < names.Length; i++)
        //     {
        //         TryAdd(names[i]);
        //     }
        //     Bones = bones.ToArray();

        // }
#endif

        public class Trajectory {
            public MotionModule.Series Series;
            public string Name;
            public Matrix4x4[] Transformations;
            public Vector3[] Velocities;

            public bool Controlled;

            private Color Color = Color.black;
            private float Thickness = 0f;
            private int Start = 0;
            private int End = 0;

            public Trajectory(MotionModule.Series series, string name, Color color, float thickness, int start, int end) {
                Series = series;
                Name = name;
                Transformations = new Matrix4x4[series.SampleCount];
                Velocities = new Vector3[series.SampleCount];
                for(int i=0; i<series.SampleCount; i++) {
                    Transformations[i] = Matrix4x4.identity;
                    Velocities[i] = Vector3.zero;
                }

                Controlled = true;

                Color = color;
                Thickness = thickness;
                Start = start;
                End = end;
            }

            public void SetColor(Color color) {
                Color = color;
            }

            public Color GetColor() {
                return Color;
            }

            public void SetTransformation(int index, Matrix4x4 value) {
                Transformations[index] = value;
            }

            public void SetTransformation(int index, Matrix4x4 value, float weight) {
                Transformations[index] = Utility.Interpolate(Transformations[index], value, weight);
            }

            public void SetPosition(int index, Vector3 value) {
                Matrix4x4Extensions.SetPosition(ref Transformations[index], value);
            }

            public Vector3 GetPosition(int index) {
                return Transformations[index].GetPosition();
            }

            public void SetRotation(int index, Quaternion value) {
                Matrix4x4Extensions.SetRotation(ref Transformations[index], value);
            }

            public Quaternion GetRotation(int index) {
                return Transformations[index].GetRotation();
            }

            public void SetVelocity(int index, Vector3 value) {
                Velocities[index] = value;
            }

            public void SetVelocity(int index, Vector3 value, float weight) {
                Velocities[index] = Vector3.Lerp(Velocities[index], value, weight);
            }

            public Vector3 GetVelocity(int index) {
                return Velocities[index];
            }

            public float ControlWeight(float x, float weight) {
                // return 1f - Mathf.Pow(1f-x, weight);
                return x.SmoothStep(2f, 1f-weight);
                // return x.ActivateCurve(weight, 0f, 1f);
            }

            public void SetTargetTransformation(Matrix4x4 value) {
                Transformations[Series.Pivot] = value;
            }

            public void SetTargetTransformation(Matrix4x4 value, float weight) {
                Transformations[Series.Pivot] = Utility.Interpolate(Transformations[Series.Pivot], value, weight);
            }

            public Matrix4x4 GetTargetTransformation() {
                return Transformations[Series.Pivot];
            }

            public void SetTargetVelocity(Vector3 value) {
                Velocities[Series.Pivot] = value;
            }

            public void SetTargetVelocity(Vector3 value, float weight) {
                Velocities[Series.Pivot] = Vector3.Lerp(Velocities[Series.Pivot], value, weight);
            }

            public Vector3 GetTargetVelocity() {
                return Velocities[Series.Pivot];
            }

            private Vector3[] CopyPositions;
            private Quaternion[] CopyRotations;
            private Vector3[] CopyVelocities;
            public void Control(Vector3 move, Quaternion rotation, float weight, float positionBias=1f, float directionBias=1f, float velocityBias=1f) {
                Increment(0, Series.Samples.Length-1);
                CopyPositions = new Vector3[Series.Samples.Length];
                CopyRotations = new Quaternion[Series.Samples.Length];
                CopyVelocities = new Vector3[Series.Samples.Length];
                for(int i=0; i<Series.Samples.Length; i++) {
                    CopyPositions[i] = GetPosition(i);
                    CopyRotations[i] = GetRotation(i);
                    CopyVelocities[i] = GetVelocity(i);
                }
                for(int i=Series.Pivot; i<Series.Samples.Length; i++) {
                    float ratio = i.Ratio(Series.Pivot-1, Series.Samples.Length-1);
                    //Root Positions
                    CopyPositions[i] = CopyPositions[i-1] +
                        Vector3.Lerp(
                            GetPosition(i) - GetPosition(i-1),
                            1f/Series.FutureSamples * move,
                            weight * ControlWeight(ratio, positionBias)
                        );

                    //Root Rotations
                    CopyRotations[i] = CopyRotations[i-1] *
                        Quaternion.Slerp(
                            GetRotation(i).RotationTo(GetRotation(i-1)),
                            rotation.RotationTo(CopyRotations[i-1]),
                            weight * ControlWeight(ratio, directionBias)
                        );

                    //Root Velocities
                    CopyVelocities[i] = CopyVelocities[i-1] +
                        Vector3.Lerp(
                            GetVelocity(i) - GetVelocity(i-1),
                            move-CopyVelocities[i-1],
                            weight * ControlWeight(ratio, velocityBias)
                        );
                }
                for(int i=0; i<Series.Samples.Length; i++) {
                    SetPosition(i, CopyPositions[i]);
                    SetRotation(i, CopyRotations[i]);
                    SetVelocity(i, CopyVelocities[i]);
                }
            }

            public void Increment(int start, int end) {
                for(int i=start; i<end; i++) {
                    Transformations[i] = Transformations[i+1];
                    Velocities[i] = Velocities[i+1];
                }
            }

            public void Interpolate(int start, int end) {
                for(int i=start; i<end; i++) {
                    float weight = (float)(i % Series.Resolution) / (float)Series.Resolution;
                    int prevIndex = Series.GetPreviousKey(i).Index;
                    int nextIndex = Series.GetNextKey(i).Index;
                    if(prevIndex != nextIndex) {
                        SetPosition(i, Vector3.Lerp(GetPosition(prevIndex), GetPosition(nextIndex), weight));
                        SetRotation(i, Quaternion.Slerp(GetRotation(prevIndex), GetRotation(nextIndex), weight));
                        SetVelocity(i, Vector3.Lerp(GetVelocity(prevIndex), GetVelocity(nextIndex), weight));
                    }
                }
            }

            public void Draw(
                Color color,
                int start, 
                int end, 
                float thickness=1f,
                float fade=0f,
                bool drawConnections=true,
                bool drawPositions=true,
                bool drawRotations=false,
                bool drawVelocities=true,
                bool keys=true
            ) {
                //Connections
                if(drawConnections) {
                    for(int i=start; i<end-1; i++) {
                        float opacity = i.Ratio(start, end-2).Normalize(0f, 1f, 1f-fade, 1f);
                        int from = keys ? Series.GetKey(i).Index : i;
                        int to = keys ? Series.GetKey(i+1).Index : (i+1);
                        UltiDraw.DrawLine(Transformations[from].GetPosition(), Transformations[to].GetPosition(), thickness*0.01f, color.Opacity(opacity));
                    }
                }

                //Positions
                if(drawPositions) {
                    for(int i=start; i<end; i++) {
                        float opacity = i.Ratio(start, end-1).Normalize(0f, 1f, 1f-fade, 1f);
                        int index = keys ? Series.GetKey(i).Index : i;
                        UltiDraw.DrawCircle(Transformations[index].GetPosition(), thickness*0.025f, color.Opacity(opacity));
                    }
                }

                //Rotations
                if(drawRotations) {
                    for(int i=start; i<end; i++) {
                        UltiDraw.DrawTranslateGizmo(Transformations[Series.GetKey(i).Index], thickness*0.1f);
                    }
                }

                //Velocities
                if(drawVelocities) {
                    for(int i=start; i<end; i++) {
                        float opacity = i.Ratio(start, end-1).Normalize(0f, 1f, 1f-fade, 1f);
                        int index = keys ? Series.GetKey(i).Index : i;
                        UltiDraw.DrawLine(Transformations[index].GetPosition(), Transformations[index].GetPosition() + Series.GetTemporalScale(Velocities[index]), thickness*0.01f, color.Lighten(0.5f).Opacity(opacity));
                    }
                }
            }

            public void Sketch() {
                UltiDraw.Begin();

                //Connections
                for(int i=Start; i<End-1; i++) {
                    UltiDraw.DrawLine(Transformations[Series.GetKey(i).Index].GetPosition(), Transformations[Series.GetKey(i+1).Index].GetPosition(), Thickness*0.01f, Color.Opacity(0.5f));
                }

                //Positions
                for(int i=Start; i<End; i++) {
                    UltiDraw.DrawCircle(Transformations[Series.GetKey(i).Index].GetPosition(), Thickness*0.025f, Color);
                }

                UltiDraw.End();
            }
        }

        public class Series : TimeSeries.Component {

            public Trajectory[] Trajectories = new Trajectory[0];

			public Series(TimeSeries global, params string[] names) : base(global) {
                foreach(string name in names) {
                    AddTrajectory(name);
                }
			}

			public Series(TimeSeries global, Actor actor, params string[] names) : base(global) {
                foreach(string name in names) {
                    Trajectory trajectory = AddTrajectory(name);
                    trajectory.Transformations.SetAll(actor.GetBoneTransformation(name));
                    trajectory.Velocities.SetAll(Vector3.zero);
                }
			}

            public Trajectory AddTrajectory(string name) {
                Trajectory trajectory = new Trajectory(this, name, Color.white, 1f, 0, KeyCount);
                ArrayExtensions.Append(ref Trajectories, trajectory);
                return trajectory;
            }

            public Trajectory AddTrajectory(Trajectory trajectory) {
                ArrayExtensions.Append(ref Trajectories, trajectory);
                return trajectory;
            }
            public Trajectory AddTrajectory(Trajectory trajectory, Color color) {
                trajectory.SetColor(color);
                ArrayExtensions.Append(ref Trajectories, trajectory);
                return trajectory;
            }
            public Trajectory AddTrajectory(string name, Color color) {
                Trajectory trajectory = new Trajectory(this, name, color, 1f, 0, KeyCount);
                ArrayExtensions.Append(ref Trajectories, trajectory);
                return trajectory;
            }

            public Trajectory AddTrajectory(string name, Color color, float thickness, int start, int end) {
                Trajectory trajectory = new Trajectory(this, name, color, thickness, start, end);
                ArrayExtensions.Append(ref Trajectories, trajectory);
                return trajectory;
            }

            public Trajectory[] AddTrajectories(string[] names) {
                Trajectory[] trajectories = new Trajectory[names.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    trajectories[i] = AddTrajectory(names[i]);
                }
                return trajectories;
            }

            public Trajectory[] AddTrajectories(string[] names, Color color) {
                Trajectory[] trajectories = new Trajectory[names.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    trajectories[i] = AddTrajectory(names[i], color);
                }
                return trajectories;
            }

            public Trajectory[] AddTrajectories(Trajectory[] trajectories) {
                Trajectories = ArrayExtensions.Concat(Trajectories, trajectories);
                return Trajectories;
            }

            public Trajectory[] SetTrajectories(Trajectory[] trajectories) {
                Trajectories = trajectories;
                return Trajectories;
            }

            public Trajectory GetTrajectory(string name) {
                return Array.Find(Trajectories, x => x.Name == name);
            }

            public Trajectory[] GetTrajectories(params string[] names) {
                Trajectory[] trajectories = new Trajectory[names.Length];
                for (int i = 0; i < names.Length; i++)
                {
                    trajectories[i] = Array.Find(Trajectories, x => x.Name == names[i]);
                }
                return trajectories;
            }

            public Matrix4x4[] GetTransformations(int index) {
                Matrix4x4[] values = new Matrix4x4[Trajectories.Length];
                for(int i=0; i<values.Length; i++) {
                    values[i] = Trajectories[i].Transformations[index];
                }
                return values;
            }

            public string[] GetNames() {
                string[] names = new string[Trajectories.Length];
                for(int i=0; i<Trajectories.Length; i++) {
                    names[i] = Trajectories[i].Name;
                }
                return names;
            }

            public override void Increment(int start, int end) {
                foreach(Trajectory trajectory in Trajectories) {
                    trajectory.Increment(start, end);
                }
            }

            public override void GUI(UltiDraw.GUIRect rect=null) {
                if(DrawGUI) {

                }
            }

			public override void Draw(UltiDraw.GUIRect rect=null) {
				if(DrawScene) {
                    Draw(Color.white, 0, KeyCount);
				}
			}

			public void Draw(Color color) {
				if(DrawScene) {
                    Draw(color, 0, KeyCount);
				}
			}

			public void DrawComplete(Color color) {
				if(DrawScene) {
                    Draw(color, 0, SampleCount, keys:false);
				}
			}

            public void Draw(
                Color color,
                int start, 
                int end, 
                float thickness=1f,
                float fade=0f,
                bool drawConnections=true,
                bool drawPositions=true,
                bool drawRotations=false,
                bool drawVelocities=true,
                bool keys=true
            ) {
                UltiDraw.Begin();
                foreach(Trajectory trajectory in Trajectories) {
                    trajectory.Draw(
                        color,
                        start,
                        end,
                        thickness,
                        fade,
                        drawConnections,
                        drawPositions,
                        drawRotations,
                        drawVelocities,
                        keys
                    );
                }
                UltiDraw.End();
            }
        }
    }
}
