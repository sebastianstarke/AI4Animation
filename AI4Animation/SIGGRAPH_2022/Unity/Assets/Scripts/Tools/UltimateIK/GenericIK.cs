using System.Collections.Generic;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace UltimateIK {

  public class GenericIK : MonoBehaviour {
      public bool AutoUpdate = true;

      public Transform Root = null;
      public Transform[] Objectives = new Transform[0];
      public Transform[] Targets = new Transform[0];

      public bool Draw = true;

      public IK Model;

      void Reset() {
          Root = transform;
          Model = new IK();
      }

      void Awake() {
        for(int i=0; i<Objectives.Length; i++) {
            Targets[i].transform.position = Objectives[i].transform.position;
            Targets[i].transform.rotation = Objectives[i].transform.rotation;
        }
      }

      void LateUpdate() {
          if(AutoUpdate) {
              List<Matrix4x4> targets = new List<Matrix4x4>();
              for(int i=0; i<Objectives.Length; i++) {
                  if(Objectives[i] != null) {
                      targets.Add(Targets[i] != null ? Targets[i].GetWorldMatrix() : Objectives[i].GetWorldMatrix());
                  }
              }
              Model.SetTargets(targets.ToArray());
              Model.Solve();
          }
      }

      public void Rebuild() {
          Model = IK.Create(Model, Root, Objectives);
      }

      #if UNITY_EDITOR
      [CustomEditor(typeof(GenericIK), true)]
      public class GenericIK_Editor : Editor {

          public static GenericIK_Editor Instance = null;

          public GenericIK Target;

          private int SelectedJoint = -1;
          private Vector3 Preview = Vector3.zero;

          private Color Background = new Color(0.25f, 0.25f, 0.25f, 1f); //Dark Grey
          private Color Header = new Color(212f/255f, 175f/255f, 55f/255f, 1f); //Gold
          private Color Content = Color.white;
          private Color Section = Color.white;
          private Color Panel = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
          private Color RegularField = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
          private Color ValidField = new Color(92/255f, 205/255f, 92/255f, 1f); //Light Green
          private Color InvalidField = new Color(205/255f, 92/255f, 92/255f, 1f); //Light Red
          private Color RegularButton = new Color(0.25f, 0.25f, 0.25f, 1f); //Dark Grey
          private Color PassiveButton = new Color(0.75f, 0.75f, 0.75f, 1f); //Light Grey
          private Color ActiveButton = new Color(0.4f, 0.5f, 0.6f, 1f); //Metal Blue
          private Color RegularFont = Color.white;
          private Color PassiveFont = new Color(1f/3f, 1f/3f, 1f/3f, 1f); //Grey
          private Color ActiveFont = Color.white;

          void Awake() {
              Instance = this;
              Target = (GenericIK)target;
          }
          
          void SetRoot(Transform t) {
              if(t == null || Target.Root == t || !IK.IsInsideHierarchy(Target.transform, t)) {
                  return;
              }
              Target.Root = t;
              Target.Rebuild();
          }

          void SetObjective(int index, Transform t) {
              if(Target.Objectives[index] == t) {
                  return;
              }
              if(t != null) {
                  if(Target.Objectives.Contains(t) || !IK.IsInsideHierarchy(Target.transform, t)) {
                      return;
                  }
              }
              Target.Objectives[index] = t;
              Target.Rebuild();
          }

          public override void OnInspectorGUI() {
              Undo.RecordObject(Target, Target.name);

              Utility.SetGUIColor(Background);
              using(new EditorGUILayout.VerticalScope ("Box")) {
                  Utility.ResetGUIColor();
                  if(Utility.GUIButton(Target.Draw ? "Drawing On" : "Drawing Off", Target.Draw ? ActiveButton : PassiveButton, Target.Draw ? ActiveFont : PassiveFont)) {
                      Target.Draw = !Target.Draw;
                  }
              }

              Utility.SetGUIColor(Background);
              using(new EditorGUILayout.VerticalScope ("Box")) {
                  Utility.ResetGUIColor();

                  Utility.SetGUIColor(Header);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      EditorGUILayout.LabelField("Setup");
                  }
                  Utility.SetGUIColor(Content);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      EditorGUILayout.HelpBox("Solve Time: " + (1000f*Target.Model.GetSolveTime()).ToString("F3") + "ms", MessageType.None);
                      Target.AutoUpdate = EditorGUILayout.Toggle("Auto Update", Target.AutoUpdate);
                      Target.Model.SetIterations(EditorGUILayout.IntField("Iterations", Target.Model.Iterations));
                      Target.Model.SetThreshold(EditorGUILayout.FloatField("Threshold", Target.Model.Threshold));
                      Target.Model.Activation = (UltimateIK.ACTIVATION)EditorGUILayout.EnumPopup("Activation", Target.Model.Activation);
                      Target.Model.AvoidJointLimits = EditorGUILayout.Toggle("Avoid Bone Limits", Target.Model.AvoidJointLimits);
                      Target.Model.AllowRootUpdateX = EditorGUILayout.Toggle("Allow Root Update X", Target.Model.AllowRootUpdateX);
                      Target.Model.AllowRootUpdateY = EditorGUILayout.Toggle("Allow Root Update Y", Target.Model.AllowRootUpdateY);
                      Target.Model.AllowRootUpdateZ = EditorGUILayout.Toggle("Allow Root Update Z", Target.Model.AllowRootUpdateZ);
                      Target.Model.RootWeight = EditorGUILayout.FloatField("Root Weight", Target.Model.RootWeight);
                      EditorGUILayout.BeginHorizontal();
                      Target.Model.SeedZeroPose = EditorGUILayout.Toggle("Seed Zero Pose", Target.Model.SeedZeroPose);
                      if(Utility.GUIButton("Save", RegularButton, RegularFont, 60f, 18f)) {
                          Target.Model.SaveZeroPose();
                      }
                      EditorGUILayout.EndHorizontal();
                      SetRoot((Transform)EditorGUILayout.ObjectField("Root", Target.Root, typeof(Transform), true));
                  }
              }

              Utility.SetGUIColor(Background);
              using(new EditorGUILayout.VerticalScope ("Box")) {
                  Utility.ResetGUIColor();

                  Utility.SetGUIColor(Header);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      EditorGUILayout.LabelField("Objectives");
                  }
                  Utility.SetGUIColor(Content);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      for(int i=0; i<Target.Objectives.Length; i++) {
                          InspectObjective(i);
                      }
                      EditorGUILayout.BeginHorizontal();
                      if(Utility.GUIButton("Add Objective", RegularButton, RegularFont)) {
                          ArrayExtensions.Expand(ref Target.Objectives);
                          ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                      }
                      EditorGUILayout.EndHorizontal();
                  }
              }

              Utility.SetGUIColor(Background);
              using(new EditorGUILayout.VerticalScope ("Box")) {
                  Utility.ResetGUIColor();

                  Utility.SetGUIColor(Header);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      EditorGUILayout.LabelField("Skeleton");
                  }

                  foreach(Joint joint in Target.Model.Joints) {
                      InspectJoint(joint);
                  }

              }

              if(GUI.changed) {
                  EditorUtility.SetDirty(Target);
              }
          }

          void InspectJoint(UltimateIK.Joint joint) {
              Utility.SetGUIColor(Content);
              using(new EditorGUILayout.VerticalScope ("Box")) {
                  Utility.ResetGUIColor();
                  EditorGUILayout.BeginHorizontal();
                  joint.Active = EditorGUILayout.Toggle(joint.Active, GUILayout.Width(20f));
                  EditorGUILayout.BeginVertical();
                  EditorGUI.BeginDisabledGroup(!joint.Active);
                  
                  if(joint.Index == SelectedJoint) {
                      if(!joint.Active || Utility.GUIButton(joint.Transform.name, ActiveButton, ActiveFont)) {
                          if(Preview != Vector3.zero) {
                              joint.Transform.localRotation = joint.ZeroRotation;
                              Preview = Vector3.zero;
                          }
                          SelectedJoint = -1;
                      }

                      Utility.SetGUIColor(Panel);
                      using(new EditorGUILayout.VerticalScope ("Box")) {
                          Utility.ResetGUIColor();

                          joint.SetJointType((UltimateIK.TYPE)EditorGUILayout.EnumPopup("Joint", joint.GetJointType()));
                          switch(joint.GetJointType()) {
                              case UltimateIK.TYPE.Free:
                              break;

                              case UltimateIK.TYPE.HingeX:
                              HingeInspector(joint);
                              break;

                              case UltimateIK.TYPE.HingeY:
                              HingeInspector(joint);
                              break;

                              case UltimateIK.TYPE.HingeZ:
                              HingeInspector(joint);
                              break;

                              case UltimateIK.TYPE.Ball:
                              BallInspector(joint);
                              break;
                          }
                      }

                      Utility.SetGUIColor(Panel);
                      using(new EditorGUILayout.VerticalScope ("Box")) {
                          Utility.ResetGUIColor();

                          EditorGUILayout.LabelField("Preview");
                          EditorGUI.BeginChangeCheck();
                          EditorGUILayout.BeginHorizontal();
                          EditorGUILayout.LabelField("X", GUILayout.Width(50f));
                          Preview.x = EditorGUILayout.Slider(Preview.x, -180f, 180f);
                          EditorGUILayout.EndHorizontal();
                          EditorGUILayout.BeginHorizontal();
                          EditorGUILayout.LabelField("Y", GUILayout.Width(50f));
                          Preview.y = EditorGUILayout.Slider(Preview.y, -180f, 180f);
                          EditorGUILayout.EndHorizontal();
                          EditorGUILayout.BeginHorizontal();
                          EditorGUILayout.LabelField("Z", GUILayout.Width(50f));
                          Preview.z = EditorGUILayout.Slider(Preview.z, -180f, 180f);
                          EditorGUILayout.EndHorizontal();
                          if(EditorGUI.EndChangeCheck()) {
                              joint.Transform.localRotation = joint.ZeroRotation * Quaternion.Euler(Preview);
                          }
                      }

                      Utility.SetGUIColor(Panel);
                      using(new EditorGUILayout.VerticalScope ("Box")) {
                          Utility.ResetGUIColor();
                          EditorGUI.BeginDisabledGroup(true);
                          EditorGUILayout.Vector3Field("Zero Position", joint.ZeroPosition);
                          EditorGUILayout.Vector3Field("Zero Rotation", joint.ZeroRotation.eulerAngles);
                          EditorGUI.EndDisabledGroup();
                      }
                  } else {
                      if(Utility.GUIButton(joint.Transform.name, RegularButton, RegularFont)) {
                          if(Preview != Vector3.zero) {
                              Target.Model.Joints[SelectedJoint].Transform.localRotation = Target.Model.Joints[SelectedJoint].ZeroRotation;
                              Preview = Vector3.zero;
                          }
                          SelectedJoint = joint.Index;
                      }
                  }
                  EditorGUI.EndDisabledGroup();
                  EditorGUILayout.EndVertical();
                  EditorGUILayout.EndHorizontal();
              }
          }

          void HingeInspector(UltimateIK.Joint joint) {
              joint.SetLowerLimit(EditorGUILayout.Slider("Lower", joint.GetLowerLimit(), -180f, 0f));
              joint.SetUpperLimit(EditorGUILayout.Slider("Upper", joint.GetUpperLimit(), 0f, 180f));
          }

          void BallInspector(UltimateIK.Joint joint) {
              float value = EditorGUILayout.Slider("Twist", 0.5f * (joint.GetUpperLimit() - joint.GetLowerLimit()), 0f, 180f);
              joint.SetLowerLimit(-value);
              joint.SetUpperLimit(value);
          }

          void InspectObjective(int index) {
              if(Target.Objectives[index] != null && Target.Model.FindObjective(Target.Objectives[index]) != null) {
                  UltimateIK.Objective o = Target.Model.FindObjective(Target.Objectives[index]);
                  Utility.SetGUIColor(Panel);
                  using(new EditorGUILayout.VerticalScope ("Box")) {
                      Utility.ResetGUIColor();
                      EditorGUILayout.BeginHorizontal();
                      o.Active = EditorGUILayout.Toggle(o.Active, GUILayout.Width(20f));
                      EditorGUI.BeginDisabledGroup(true);
                      Utility.SetGUIColor(ValidField);
                      EditorGUILayout.ObjectField(Target.Objectives[index], typeof(Transform), true);
                      Utility.ResetGUIColor();
                      EditorGUI.EndDisabledGroup();
                      if(Utility.GUIButton("X", InvalidField, RegularFont, 36f, 18f)) {
                          ArrayExtensions.RemoveAt(ref Target.Objectives, index);
                          ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                          Target.Rebuild();
                      } else {
                          EditorGUILayout.EndHorizontal();
                          Target.Targets[index] = (Transform)EditorGUILayout.ObjectField("Target", Target.Targets[index], typeof(Transform), true);
                          EditorGUILayout.BeginHorizontal();
                          o.Weight = EditorGUILayout.Slider("Weight", o.Weight, 0f, 1f);
                          if(Utility.GUIButton("Solve Position", o.SolvePosition ? ActiveButton : PassiveButton, o.SolvePosition ? ActiveFont : PassiveFont)) {
                              o.SolvePosition = !o.SolvePosition;
                          }
                          if(Utility.GUIButton("Solve Rotation", o.SolveRotation ? ActiveButton : PassiveButton, o.SolveRotation ? ActiveFont : PassiveFont)) {
                              o.SolveRotation = !o.SolveRotation;
                          }
                          EditorGUILayout.EndHorizontal();
                      }
                  }
              } else {
                  EditorGUILayout.BeginHorizontal();
                  Utility.SetGUIColor(InvalidField);
                  SetObjective(index, (Transform)EditorGUILayout.ObjectField("Bone", Target.Objectives[index], typeof(Transform), true));
                  Utility.ResetGUIColor();
                  if(Utility.GUIButton("X", InvalidField, RegularFont, 36f, 18f)) {
                      ArrayExtensions.RemoveAt(ref Target.Objectives, index);
                      ArrayExtensions.Resize(ref Target.Targets, Target.Objectives.Length);
                  }
                  EditorGUILayout.EndHorizontal();
              }
          }

          [DrawGizmo(GizmoType.Active | GizmoType.NotInSelectionHierarchy)]
          static void OnScene(Transform t, GizmoType gizmoType) {
              if(Instance != null) {
                  if(Instance.Target.Draw && Instance.Target.Model.IsSetup()) {
                      if(!Application.isPlaying) {
                          List<Matrix4x4> targets = new List<Matrix4x4>();
                          for(int i=0; i<Instance.Target.Objectives.Length; i++) {
                              if(Instance.Target.Objectives[i] != null) {
                                  targets.Add(Instance.Target.Targets[i] != null ? Instance.Target.Targets[i].GetWorldMatrix() : Instance.Target.Objectives[i].GetWorldMatrix());
                              }
                          }
                          for(int i=0; i<Instance.Target.Model.Objectives.Length; i++) {
                              Instance.Target.Model.Objectives[i].TargetPosition = targets[i].GetPosition();
                              Instance.Target.Model.Objectives[i].TargetRotation = targets[i].GetRotation();
                          }
                      }

                      if(Instance.Target.Model.Joints.Length > 0) {
                          Instance.DrawSkeleton(null, Instance.Target.Model.Joints.First());
                      }
                      foreach(UltimateIK.Objective objective in Instance.Target.Model.Objectives) {
                          Instance.DrawObjective(objective);
                      }
                      foreach(UltimateIK.Joint joint in Instance.Target.Model.Joints) {
                          Instance.DrawBone(joint);
                      }
                      Instance.DrawSelection();
                  }
              }
          }

          void DrawCoordinateSystem(Vector3 position, Quaternion rotation, float size) {
              Handles.color = Color.red;
              Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.right), size, EventType.Repaint);
              Handles.color = Color.green;
              Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.up), size, EventType.Repaint);
              Handles.color = Color.blue;
              Handles.ArrowHandleCap(0, position, rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.forward), size, EventType.Repaint);
          }

          void DrawSelection() {
              if(SelectedJoint != -1) {
                  UltimateIK.Joint joint = Target.Model.Joints[SelectedJoint];
                  Handles.color = Color.black.Opacity(0.5f);
                  Handles.SphereHandleCap(0, joint.Transform.position, Quaternion.identity, 0.3f, EventType.Repaint);

                  Handles.color = Color.magenta.Opacity(0.75f);
                  Handles.CubeHandleCap(0, joint.Transform.position, joint.Transform.rotation, 0.05f, EventType.Repaint);

                  Quaternion seed = joint.Transform.parent != null ? joint.Transform.parent.rotation : Quaternion.identity;
                  switch(joint.GetJointType()) {
                      case UltimateIK.TYPE.Free:
                      DrawLimit(joint, Axis.XPositive, false);
                      DrawLimit(joint, Axis.YPositive, false);
                      DrawLimit(joint, Axis.ZPositive, false);
                      break;
                      
                      case UltimateIK.TYPE.HingeX:
                      DrawLimit(joint, Axis.XPositive, true);
                      DrawLimit(joint, Axis.YPositive, false);
                      DrawLimit(joint, Axis.ZPositive, false);
                      break;

                      case UltimateIK.TYPE.HingeY:
                      DrawLimit(joint, Axis.XPositive, false);
                      DrawLimit(joint, Axis.YPositive, true);
                      DrawLimit(joint, Axis.ZPositive, false);
                      break;

                      case UltimateIK.TYPE.HingeZ:
                      DrawLimit(joint, Axis.XPositive, false);
                      DrawLimit(joint, Axis.YPositive, false);
                      DrawLimit(joint, Axis.ZPositive, true);
                      break;

                      case UltimateIK.TYPE.Ball:
                      DrawLimit(joint, Axis.XPositive, true);
                      DrawLimit(joint, Axis.YPositive, true);
                      DrawLimit(joint, Axis.ZPositive, true);
                      break;
                  }

              }
          }

          void DrawLimit(UltimateIK.Joint joint, Axis axis, bool active) {
              if(active) {
                  Quaternion seed = joint.Transform.parent != null ? joint.Transform.parent.rotation : Quaternion.identity;
                  switch(axis) {
                      case Axis.XPositive:
                      Handles.color = Color.red.Opacity(0.25f);
                      Handles.DrawSolidArc(joint.Transform.position, seed * joint.ZeroRotation.GetRight(), Quaternion.AngleAxis(joint.GetLowerLimit(), seed * joint.ZeroRotation.GetRight()) * seed * joint.ZeroRotation.GetUp(), joint.GetUpperLimit() - joint.GetLowerLimit(), 0.15f);
                      Handles.color = Color.red;
                      Handles.ArrowHandleCap(0, joint.Transform.position, joint.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.right), 0.15f, EventType.Repaint);
                      break;

                      case Axis.YPositive:
                      Handles.color = Color.green.Opacity(0.25f);
                      Handles.DrawSolidArc(joint.Transform.position, seed * joint.ZeroRotation.GetUp(), Quaternion.AngleAxis(joint.GetLowerLimit(), seed * joint.ZeroRotation.GetUp()) * seed * joint.ZeroRotation.GetForward(), joint.GetUpperLimit() - joint.GetLowerLimit(), 0.15f);
                      Handles.color = Color.green;
                      Handles.ArrowHandleCap(0, joint.Transform.position, joint.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.up), 0.15f, EventType.Repaint);
                      break;

                      case Axis.ZPositive:
                      Handles.color = Color.blue.Opacity(0.25f);
                      Handles.DrawSolidArc(joint.Transform.position, seed * joint.ZeroRotation.GetForward(), Quaternion.AngleAxis(joint.GetLowerLimit(), seed * joint.ZeroRotation.GetForward()) * seed * joint.ZeroRotation.GetRight(), joint.GetUpperLimit() - joint.GetLowerLimit(), 0.15f);
                      Handles.color = Color.blue;
                      Handles.ArrowHandleCap(0, joint.Transform.position, joint.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, Vector3.forward), 0.15f, EventType.Repaint);
                      break;
                  }
              } else {
                  Handles.color = Color.grey;
                  Handles.ArrowHandleCap(0, joint.Transform.position, joint.Transform.rotation * Quaternion.FromToRotation(Vector3.forward, axis.GetAxis()), 0.15f, EventType.Repaint);
              }
          }

          void DrawBone(UltimateIK.Joint joint) {
              if(joint.Active && SelectedJoint != joint.Index) {
                  Handles.color = Color.magenta.Opacity(0.75f);
                  Handles.SphereHandleCap(0, joint.Transform.position, Quaternion.identity, 0.025f, EventType.Repaint);
              }
          }

          void DrawObjective(UltimateIK.Objective o) {
              if(o.Active) {
                  Handles.color = Color.green.Opacity(0.5f);
                  Handles.SphereHandleCap(0, Target.Model.Joints[o.Joint].Transform.position, Quaternion.identity, 0.1f, EventType.Repaint);
                  DrawCoordinateSystem(Target.Model.Joints[o.Joint].Transform.position, Target.Model.Joints[o.Joint].Transform.rotation, 0.05f);
                  Handles.color = Color.red.Opacity(0.75f);
                  Handles.DrawDottedLine(Target.Model.Joints[o.Joint].Transform.position, o.TargetPosition, 10f);
                  Handles.SphereHandleCap(0, o.TargetPosition, Quaternion.identity, 0.025f, EventType.Repaint);
                  DrawCoordinateSystem(o.TargetPosition, o.TargetRotation, 0.025f);
              }
          }

          void DrawSkeleton(UltimateIK.Joint parent, UltimateIK.Joint bone) {
              if(bone.Active) {
                  parent = bone;
              }
              foreach(int child in bone.Childs) {
                  if(parent != null && Target.Model.Joints[child].Active) {
                      Handles.color = Color.cyan.Opacity(0.5f);
                      float distance = Vector3.Distance(Target.Model.Joints[child].Transform.position, Camera.current.transform.position);
                      Handles.DrawAAPolyLine(5f / distance, new Vector3[2]{parent.Transform.position, Target.Model.Joints[child].Transform.position});
                      
                  }
              }
              foreach(int child in bone.Childs) {
                  DrawSkeleton(parent, Target.Model.Joints[child]);
              }
          }

      }
      #endif
  }

}