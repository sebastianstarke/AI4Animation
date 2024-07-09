using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEditor;

namespace MagicIK {
    public class GenericIK : IK {

        public string Tag = string.Empty;
        public bool Draw = true;
        public bool AutoUpdate = true;
        public Transform Root = null;
        public List<Objective> Objectives = new List<Objective>();
        [NonSerialized] public bool ShowActivations = false;

        public void Refresh() {
            Solver = new Solver(Solver, Root, GetTransforms());
        }

        void Update() {
            if(AutoUpdate) {
                Solve();
            }
        }

        public void Solve() {
            for(int i=0; i<Solver.Objectives.Length; i++) {
                if(Objectives[i].Target != null) {
                    Solver.Objectives[i].Position = Objectives[i].Target.position;
                    Solver.Objectives[i].Rotation = Objectives[i].Target.rotation;
                }
            }
            Solver.Solve();
        }
        
        void OnDrawGizmos() {
            if(!Application.isPlaying) {
                OnRenderObject();
            }
        }

        void OnRenderObject() {
            if(Draw){
                Solver.Draw();
            }
        }

        private Transform[] GetTransforms() {
            Transform[] transforms = new Transform[Objectives.Count];
            for(int i=0; i<Objectives.Count; i++) {
                transforms[i] = Objectives[i].Transform;
            }
            return transforms;
        }

        [Serializable]
        public class Objective {
            public Transform Transform;
            public Transform Target;
        }

        #if UNITY_EDITOR
        [CustomEditor(typeof(GenericIK), true)]
        public class GenericIKEditor : Editor {
            public GenericIK Instance;

            void Awake() {
                Instance = (GenericIK)target;
            }

            public override void OnInspectorGUI() {
                Undo.RecordObject(Instance, Instance.name);

                Instance.Tag = EditorGUILayout.TextField("Tag", Instance.Tag);
                Instance.Draw = EditorGUILayout.Toggle("Draw", Instance.Draw);
                Instance.AutoUpdate = EditorGUILayout.Toggle("Auto Update", Instance.AutoUpdate);
                Instance.Root = EditorGUILayout.ObjectField("Root", Instance.Root, typeof(Transform), true) as Transform;

                Instance.Solver.MaxIterations = EditorGUILayout.IntField("Max Iterations", Instance.Solver.MaxIterations);
                Instance.Solver.MaxError = EditorGUILayout.FloatField("Max Error", Instance.Solver.MaxError);
                Instance.Solver.RootPull = EditorGUILayout.Slider("Root Pull", Instance.Solver.RootPull, 0f, 1f);
                Instance.Solver.SeedZeros = EditorGUILayout.Toggle("Seed Zeros", Instance.Solver.SeedZeros);

                Utility.SetGUIColor(UltiDraw.Black);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.LabelField("Objectives");
                    foreach(Objective o in Instance.Objectives) {
                        EditorGUILayout.BeginHorizontal();
                        Transform tmp = o.Transform;
                        o.Transform = EditorGUILayout.ObjectField(o.Transform, typeof(Transform), true) as Transform;
                        if(o.Transform != tmp) {
                            goto Exit;
                        }
                        if(Utility.GUIButton("-", UltiDraw.DarkGrey, UltiDraw.White, 20f, 20f)) {
                            Instance.Objectives.Remove(o);
                            goto Exit;
                        }
                        EditorGUILayout.EndHorizontal();
                    }
                    if(Utility.GUIButton("+", UltiDraw.DarkGrey, UltiDraw.White)) {
                        Instance.Objectives.Add(new Objective());
                    }
                }

                Utility.SetGUIColor(UltiDraw.Black);
                using(new EditorGUILayout.VerticalScope ("Box")) {
                    Utility.ResetGUIColor();
                    EditorGUILayout.BeginHorizontal();
                    EditorGUILayout.LabelField("Hierarchy");
                    if(Utility.GUIButton("Show Activations", Instance.ShowActivations ? UltiDraw.Green : UltiDraw.DarkGrey, UltiDraw.White)) {
                        Instance.ShowActivations = !Instance.ShowActivations;
                    }
                    EditorGUILayout.EndHorizontal();
                    void Recursion(Solver.Node node) {
                        Utility.SetGUIColor(UltiDraw.Black);
                        using(new EditorGUILayout.VerticalScope ("Box")) {
                            Utility.ResetGUIColor();

                            Utility.SetGUIColor(UltiDraw.Black);
                            using(new EditorGUILayout.VerticalScope ("Box")) {
                                Utility.ResetGUIColor();
                                EditorGUILayout.BeginHorizontal();
                                node.Active = EditorGUILayout.Toggle(node.Active, GUILayout.Width(20f));
                                string indent = string.Empty;
                                for(int i=0; i<node.Depth; i++) {
                                    indent += " ";
                                }
                                EditorGUILayout.LabelField(indent + node.Transform.name);
                                Utility.SetGUIColor(node.Limit != null ? UltiDraw.Green : UltiDraw.Black);
                                EditorGUILayout.ObjectField(node.Limit, typeof(Limit), true, GUILayout.Width(150f));
                                Utility.ResetGUIColor();
                                if(Utility.GUIButton("Seed Zero", node.SeedZero ? UltiDraw.Green : UltiDraw.Black, UltiDraw.White)) {
                                    node.SeedZero = !node.SeedZero;
                                }
                                EditorGUILayout.EndHorizontal();
                            }

                            if(Instance.ShowActivations) {
                                EditorGUI.BeginDisabledGroup(true);
                                // EditorGUILayout.LabelField("Parent", node.Parent == null ? "None" : node.Parent.Transform.name);
                                // EditorGUILayout.LabelField("Childs", node.Childs.Length.ToString());
                                // EditorGUILayout.IntField("Depth", node.Depth);
                                // EditorGUILayout.FloatField("Length", node.Length);
                                EditorGUILayout.BeginHorizontal();
                                for(int i=0; i<node.Activations.Length; i++) {
                                    EditorGUILayout.Slider(node.Activations[i], 0f, 1f);
                                }
                                EditorGUILayout.EndHorizontal();
                                EditorGUI.EndDisabledGroup();
                            }

                            Solver.Objective objective = Instance.Solver.FindObjective(node.Transform);
                            if(objective != null) {
                                Objective o = Instance.Objectives.Find(x => x.Transform == objective.Node.Transform);
                                o.Target = EditorGUILayout.ObjectField("Target", o.Target, typeof(Transform), true) as Transform;
                                objective.Weight = EditorGUILayout.Slider("Weight", objective.Weight, 0f, 1f);
                                objective.ApplyRotation = EditorGUILayout.Toggle("Apply Rotation", objective.ApplyRotation);
                            }
                        }
                        foreach(Solver.Node child in node.Childs) {
                            Recursion(child);
                        }
                    }
                    if(Instance.Solver.Nodes.Length > 0) {
                        Recursion(Instance.Solver.Nodes[0]);
                    }
                }

                if(Instance.Solver.Info != null && Instance.Solver.Info.Count > 0) {
                    EditorGUILayout.HelpBox(Instance.Solver.Info.ToArray().FormatAsLines(), MessageType.Warning);
                }

                Exit:
                if(GUI.changed) {
                    Instance.Refresh();
                    EditorUtility.SetDirty(Instance);
                }
            }
        }
        #endif

    }
}