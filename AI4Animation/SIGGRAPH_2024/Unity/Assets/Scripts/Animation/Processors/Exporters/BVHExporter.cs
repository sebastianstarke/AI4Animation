#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections.Generic;
using System.Collections;
using System.IO;

namespace AI4Animation {
    public class BVHExporter : BatchProcessor {
        public bool OfflineProcessing = false;
        public string Destination = string.Empty;
        public string Root = string.Empty;

        private List<string> Hierarchy = new List<string>();
        private List<string> BoneMapping = new List<string>();
        private List<Quaternion> TPoseRotation = new List<Quaternion>();

        [MenuItem ("AI4Animation/Exporter/BVH Exporter")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(BVHExporter));
            Scroll = Vector3.zero;
        }

        public override string GetID(Item item) {
            return Utility.GetAssetName(item.ID);
        }

        public override void DerivedRefresh() {
            
        }

        public override bool CanProcess()
        {
            return true;
        }

        public override void DerivedInspector() {
            MotionEditor editor = MotionEditor.GetInstance();
            if(editor == null) {
                EditorGUILayout.LabelField("No editor available in scene.");
                return;
            }

            Utility.SetGUIColor(AssetDatabase.IsValidFolder(Destination) ? UltiDraw.DarkGreen : UltiDraw.DarkRed);
            Destination = EditorGUILayout.TextField("Path to save: ", Destination);
            Utility.ResetGUIColor();

            OfflineProcessing = EditorGUILayout.Toggle("Offline Processing", OfflineProcessing);

            Root = EditorGUILayout.TextField("Root", Root);

            if(Utility.GUIButton("Refresh", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadItems(editor.Assets.ToArray());
            }
        }

        public override void DerivedInspector(Item item) {

        }

        private void BuildHierarchy(Transform node, int layer, Transform parent) {
            string tab = new String('\t', layer);
            string jointType = "JOINT";
            if(layer == 0) {
                jointType = "ROOT";
            } else if(node.childCount == 0) {
                jointType = "End";
            }
            BoneMapping.Add(node.name);
            Hierarchy.Add(tab + jointType + " " + node.name);
            Hierarchy.Add(tab + "{");
            Hierarchy.Add(tab+ "\t" + "OFFSET " + String.Format("{0:0.00000}", node.localPosition.x) + " " + String.Format("{0:0.00000}", node.localPosition.y) + " " + String.Format("{0:0.00000}", node.localPosition.z));
            if(jointType != "End") {
                if(layer==0) {
                    Hierarchy.Add(tab+ "\t" + "CHANNELS 6 Xposition Yposition Zposition Yrotation Xrotation Zrotation");
                    TPoseRotation.Add(node.rotation);
                } else {
                    Hierarchy.Add(tab+ "\t" + "CHANNELS 3 Yrotation Xrotation Zrotation");
                    TPoseRotation.Add(node.localRotation);
                }
                foreach(Transform child in node) {
                    BuildHierarchy(child, layer+1, node);
                }
            }
            Hierarchy.Add(tab + "}");
        }

        public override void DerivedStart() {

        }

        public override IEnumerator DerivedProcess(Item item) {
            List<int> parentMappingCache = new List<int>();
            MotionEditor editor = MotionEditor.GetInstance();
            if(editor != null) {
                //Setup Asset
                if(!OfflineProcessing) {
                    editor.LoadSession(item.ID);
                }
                MotionAsset asset = MotionAsset.Retrieve(item.ID);
    
                StreamWriter writer = new StreamWriter(Destination + "/" + asset.name + ".bvh", true);

                Hierarchy = new List<string>();
                BoneMapping = new List<string>();
                Hierarchy.Add("HIERARCHY");
                GameObject root = GameObject.Find(Root);
                BuildHierarchy(root.transform, 0, root.transform);
                foreach(string s in Hierarchy) {
                    writer.WriteLine(s);
                }

                writer.WriteLine("MOTION");
                writer.WriteLine("Frames: " + asset.Frames.Length);
                writer.WriteLine("Frame Time: " + string.Format("{0:0.0000000}", 1f/asset.Framerate));

                foreach(Frame f in asset.Frames) {
                    string newLine = string.Empty;
                    for(int i=0; i < BoneMapping.Count; i++) {
                        MotionAsset.Hierarchy.Bone bone = asset.Source.FindBone(BoneMapping[i]);
                        MotionAsset.Hierarchy.Bone parent = bone == null ? null : asset.Source.FindBone(bone.GetParent(asset.Source).GetName());
                        Vector3 q = new Vector3();
                        if(bone == null) {

                        }
                        if(bone != null && parent == null) {
                            Matrix4x4 t = f.Transformations[bone.Index];
                            Vector3 position = t.GetPosition();
                            newLine = newLine + String.Format("{0:0.00000}", position.x) + " " 
                                + String.Format("{0:0.00000}", position.y) + " "
                                + String.Format("{0:0.00000}", position.z) + " ";
                            // q = (TPoseRotation[i].GetInverse()*Quaternion.LookRotation(t.GetColumn(2), t.GetColumn(1))).eulerAngles;
                            q = f.Transformations[bone.Index].GetRotation().eulerAngles;
                        }
                        if(bone != null && parent != null) {
                            Matrix4x4 t = f.Transformations[bone.Index];
                            Matrix4x4 tp = f.Transformations[parent.Index];
                            // q = (TPoseRotation[i].GetInverse()*Quaternion.LookRotation(t_p.GetColumn(2), t_p.GetColumn(1)).GetInverse()*Quaternion.LookRotation(t.GetColumn(2), t.GetColumn(1))).eulerAngles;
                            q = (t.TransformationTo(tp).GetRotation()).eulerAngles;
                        }
                        newLine = newLine
                                + String.Format("{0:0.00000}", q.y) + " " 
                                + String.Format("{0:0.00000}", q.x) + " "
                                + String.Format("{0:0.00000}", q.z) + " ";
                    }
                    if(newLine != string.Empty) {
                        writer.WriteLine(newLine);
                    }
                } 

                writer.Close();

                yield return new WaitForSeconds(0f);
            }
        }

        public override void BatchCallback() {
            
        }

        public override void DerivedFinish() {
            
        }

    }
}
#endif
