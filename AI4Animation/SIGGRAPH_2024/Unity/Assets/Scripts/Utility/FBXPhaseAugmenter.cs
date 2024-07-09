// #if UNITY_EDITOR
// using UnityEngine;
// using UnityEditor;
// using System;
// using System.IO;
// using System.Collections;
// using System.Collections.Generic;
// using UnityEditor.Formats.Fbx.Exporter;

// namespace AI4Animation {
//     public class FBXPhaseAugmenter : BatchProcessor {

//         public string FBXFolder;
//         public string SaveFolder;
//         public string Root;

//         private MotionEditor Editor = null;

//         [MenuItem ("AI4Animation/Tools/FBX Phase Augmenter")]
//         static void Init() {
//             Window = EditorWindow.GetWindow(typeof(FBXPhaseAugmenter));
//             Scroll = Vector3.zero;
//         }

// 		public MotionEditor GetEditor() {
// 			if(Editor == null) {
// 				Editor = GameObjectExtensions.Find<MotionEditor>(true);
// 			}
// 			return Editor;
// 		}

//         public override string GetID(Item item) {
//             int separator = item.ID.IndexOf(";");
//             string id = separator >= 0 ? item.ID.Substring(0, separator) : item.ID;
//             return Utility.GetAssetName(id);
//         }

//         public override void DerivedRefresh() {
            
//         }
        
//         public override void DerivedInspector() {
// 			if(GetEditor() == null) {
// 				EditorGUILayout.LabelField("No editor available in scene.");
// 				return;
// 			}

//             FBXFolder = EditorGUILayout.TextField("FBX Folder", FBXFolder);
//             SaveFolder = EditorGUILayout.TextField("Save Folder", SaveFolder);
//             Root = EditorGUILayout.TextField("Root Bone", Root);
            
//             if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
//                 LoadItems(GetEditor().Assets.ToArray());
//                 List<string> fbx = new List<string>();
//                 string directory = Application.dataPath + "/" + FBXFolder;
//                 if(Directory.Exists(directory)) {
//                     Iterate(directory);
//                     void Iterate(string folder) {
//                         DirectoryInfo info = new DirectoryInfo(folder);
//                         foreach(FileInfo i in info.GetFiles()) {
//                             string path = i.FullName.Substring(i.FullName.IndexOf("Assets"));
//                             if((AnimationClip)AssetDatabase.LoadAssetAtPath(path, typeof(AnimationClip))) {
//                                 fbx.Add(Utility.GetAssetGUID(AssetDatabase.LoadMainAssetAtPath(path)));
//                             }
//                         }
//                         Resources.UnloadUnusedAssets();
//                         foreach(DirectoryInfo i in info.GetDirectories()) {
//                             Iterate(i.FullName);
//                         }
//                     }
//                 }
//                 foreach(Item item in GetItems()) {
//                     bool valid = false;
//                     string reference = GetID(item);
//                     foreach(string guid in fbx) {
//                         if(Utility.GetAssetName(guid) == reference) {
//                             item.ID += ";" + guid;
//                             valid = true;
//                             break;
//                         }
//                     }
//                     if(!valid) {
//                         Debug.LogWarning("No matching FBX for asset " + GetID(item) + " could be found.");
//                     }
//                 }
//             }
//         }
        
//         public override void DerivedInspector(Item item) {
        
//         }

//         public override bool CanProcess() {
//             return true;
//         }

//         public override void DerivedStart() {
//             string destination = "Assets/" + SaveFolder;
//             Directory.CreateDirectory(destination + "/Tmp");
//         }

//         public override IEnumerator DerivedProcess(Item item) {
//             string destination = "Assets/" + SaveFolder;

//             GameObject FBX = GetFBX(item);
//             MotionAsset Asset = GetAsset(item);

//             AnimationClip source = (AnimationClip)AssetDatabase.LoadAssetAtPath(AssetDatabase.GetAssetPath(FBX), typeof(AnimationClip));

//             AnimationClip target = Instantiate(source);
//             target.name = source.name;
//             AssetDatabase.CreateAsset(target, destination + "/Tmp/" + target.name + ".anim");

//             GameObject instance = GameObject.Instantiate(FBX);
//             instance.name = FBX.name+"_phase_curves";
//             GameObject phases = new GameObject("Phases");
//             GameObject parent = FindChild(instance, Root);
//             if(parent == null) {
//                 Debug.Log("Root bone " + Root + " could not be found.");
//                 phases.transform.SetParent(instance.transform);
//             } else {
//                 phases.transform.SetParent(parent.transform);
//             }

//             Animator animator = instance.AddComponent<Animator>();
//             animator.runtimeAnimatorController = UnityEditor.Animations.AnimatorController.CreateAnimatorControllerAtPathWithClip(destination + "/Tmp/" + target.name+".controller", target);

//             AlignmentModule module = Asset.GetModule<AlignmentModule>();
//             for(int i=0; i<module.Functions.Length; i++) {
//                 AlignmentModule.Function f = module.Functions[i];
//                 List<Keyframe> framesX = new List<Keyframe>();
//                 List<Keyframe> framesY = new List<Keyframe>();
//                 float t = 0f;
//                 while(t < target.length) {
//                     Vector2 phase = module.GetAmplitude(i, t, false) * Utility.PhaseVector(module.GetPhase(i, t, false));
//                     framesX.Add(new Keyframe(t, phase.x));
//                     framesY.Add(new Keyframe(t, phase.y));
//                     t += 1f/target.frameRate;
//                 }
//                 string nameX = f.GetName()+"Phase";
//                 string nameY = f.GetName()+"Phase";
//                 GameObject goX = new GameObject(f.GetName()+"PhaseX");
//                 goX.transform.SetParent(phases.transform);
//                 GameObject goY = new GameObject(f.GetName()+"PhaseY");
//                 goY.transform.SetParent(phases.transform);
//                 AddCurve(target, Root+"/"+"Phases"+"/"+goX.name, "m_LocalPosition.x", typeof(Transform), new AnimationCurve(framesX.ToArray()));
//                 AddCurve(target, Root+"/"+"Phases"+"/"+goY.name, "m_LocalPosition.y", typeof(Transform), new AnimationCurve(framesY.ToArray()));
//             }

//             ModelExporter.ExportObject(destination + "/Tmp/" + instance.name, instance);
        
//             DirectoryInfo info = new DirectoryInfo(Application.dataPath + "/" + SaveFolder + "/Tmp");
//             foreach(FileInfo f in info.GetFiles()) {
//                 if(!f.FullName.EndsWith(".meta")) {
//                     string upper = f.Directory.Parent.FullName.Substring(f.Directory.Parent.FullName.IndexOf("Assets"));
//                     string folder = f.Directory.FullName.Substring(f.Directory.FullName.IndexOf("Assets"));
//                     string path = f.FullName.Substring(f.FullName.IndexOf("Assets"));
//                     AssetDatabase.RenameAsset(path, instance.name);
//                     AssetDatabase.MoveAsset(folder + "/" + instance.name + ".fbx", upper + "/" + instance.name + ".fbx");
//                 }
//             }

//             AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(target));
//             AssetDatabase.DeleteAsset(AssetDatabase.GetAssetPath(animator.runtimeAnimatorController));
//             Utility.Destroy(instance);

//             yield return new WaitForSeconds(1f);   
//         }

//         public override void BatchCallback() {
            
//         }

//         public override void DerivedFinish() {
//             string destination = "Assets/" + SaveFolder;
//             AssetDatabase.DeleteAsset(destination + "/Tmp");
//             AssetDatabase.SaveAssets();
//             AssetDatabase.Refresh();
//         }

//         private MotionAsset GetAsset(Item item) {
//             int separator = item.ID.IndexOf(";");
//             return MotionAsset.Retrieve(separator >= 0 ? item.ID.Substring(0, separator) : item.ID);
//         }

//         private GameObject GetFBX(Item item) {
//             int separator = item.ID.IndexOf(";");
//             return separator >= 0 ? (GameObject)AssetDatabase.LoadMainAssetAtPath(Utility.GetAssetPath(item.ID.Substring(separator+1))) : null;
//         }

//         private void AddCurve(AnimationClip clip, string path, string property, Type type, AnimationCurve curve) {
//             EditorCurveBinding binding = new EditorCurveBinding();
//             binding.path = path;
//             binding.propertyName = property;
//             binding.type = type;
//             AnimationUtility.SetEditorCurve(clip, binding, curve);
//         }

//         private GameObject FindChild(GameObject go, string name) {
//             Transform element = null;
//             Action<Transform> recursion = null;
//             recursion = new Action<Transform>((t) => {
//                 if(t.name == name) {
//                     element = t;
//                     return;
//                 }
//                 for(int i=0; i<t.childCount; i++) {
//                     recursion(t.GetChild(i));
//                 }
//             });
//             recursion(go.transform);
//             return element == null ? null : element.gameObject;
//         }

//     }
// }
// #endif