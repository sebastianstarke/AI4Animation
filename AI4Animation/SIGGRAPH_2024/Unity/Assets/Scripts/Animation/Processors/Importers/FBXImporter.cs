#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class FBXImporter : BatchProcessor {

        public string Source = string.Empty;
        public string Destination = string.Empty;
        public Actor Skeleton = null;
        public string[] Mappings = new string[0];
        private bool ShowMappings = false;

        private List<string> Imported;
        private List<string> Skipped;

        [MenuItem ("AI4Animation/Importer/FBX Importer")]
        static void Init() {
            Window = EditorWindow.GetWindow(typeof(FBXImporter));
            Scroll = Vector3.zero;
        }

        public override string GetID(Item item) {
            return item.ID;
        }

        public override void DerivedRefresh() {
            
        }

        public override void DerivedInspector() {
            EditorGUILayout.LabelField("Source");
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
            Source = EditorGUILayout.TextField(Source);
            GUI.skin.button.alignment = TextAnchor.MiddleCenter;
            if(GUILayout.Button("O", GUILayout.Width(20))) {
                SetSource(EditorUtility.OpenFolderPanel("FBX Importer", Source == string.Empty ? Application.dataPath : Source, ""));
                GUIUtility.ExitGUI();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField("Destination");
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
            Destination = EditorGUILayout.TextField(Destination);
            EditorGUILayout.EndHorizontal();

            SetSkeleton(EditorGUILayout.ObjectField("Skeleton", Skeleton, typeof(Actor), true) as Actor);
            ShowMappings = EditorGUILayout.Toggle("Show Mappings", ShowMappings);
            if(ShowMappings) {
                for(int i=0; i<Mappings.Length; i++) {
                    Mappings[i] = EditorGUILayout.TextField(Skeleton.Bones[i].GetName(), Mappings[i]);
                }
            }

            if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadDirectory(Source);
            }
        }

        public override void DerivedInspector(Item item) {
        
        }
        
        private void SetSkeleton(Actor actor) {
            if(Skeleton != actor) {
                Skeleton = actor;
                Mappings = actor == null ? new string[0] : Skeleton.GetBoneNames();
            }
        }

        private void SetSource(string source) {
            if(Source != source) {
                Source = source;
                LoadDirectory(null);
            }
        }

        private void LoadDirectory(string directory) {
            if(directory == null) {
                LoadItems(new string[0]);
            } else {
                if(Directory.Exists(directory)) {
                    List<string> paths = new List<string>();
                    Iterate(directory);
                    LoadItems(paths.ToArray());
                    void Iterate(string folder) {
                        DirectoryInfo info = new DirectoryInfo(folder);
                        foreach(FileInfo i in info.GetFiles()) {
                            if(i.Name.EndsWith(".fbx")) {
                                paths.Add(i.FullName);
                            }
                        }
                        //Resources.UnloadUnusedAssets();
                        foreach(DirectoryInfo i in info.GetDirectories()) {
                            Iterate(i.FullName);
                        }
                    }
                } else {
                    LoadItems(new string[0]);
                }
            }
        }

        public override bool CanProcess() {
            return true;
        }

        public override void DerivedStart() {
            Imported = new List<string>();
            Skipped = new List<string>();
        }

        public override IEnumerator DerivedProcess(Item item) {
            string source = Source;
            string destination = "Assets/" + Destination;

            string target = (destination + item.ID.Remove(0, source.Length)).Replace(".fbx", "");
            
            if(!Directory.Exists(target)) {
                //Create Directory
                Directory.CreateDirectory(target);
                DirectoryInfo targetInfo = new DirectoryInfo(target);

                //Import source FBX to folder
                string tmpFileName = targetInfo.Name + ".fbx";
                string absolutePath = targetInfo.FullName + "/" + tmpFileName;
                // string relativePath = destination + "/" + targetInfo.Name + "/" + tmpFileName; //Buggy if fbx is in subfolder
                string relativePath = target + "/" + tmpFileName;
                
                File.Copy(item.ID, absolutePath);
                AssetDatabase.ImportAsset(relativePath);

                GameObject go = (GameObject)AssetDatabase.LoadAssetAtPath(relativePath, typeof(GameObject));
                AnimationClip clip = (AnimationClip)AssetDatabase.LoadAssetAtPath(relativePath, typeof(AnimationClip));

                //Create Asset
                MotionAsset asset = ScriptableObject.CreateInstance<MotionAsset>();
                
                asset.name = targetInfo.Name;
                AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

                //Set Frames
                ArrayExtensions.Resize(ref asset.Frames, Mathf.RoundToInt(clip.frameRate * clip.length));

                //Set Framerate
                asset.Framerate = clip.frameRate;

                //Create Model
                GameObject instance = Instantiate(go) as GameObject;
                List<Transform> transforms = new List<Transform>(instance.GetComponentsInChildren<Transform>());

                // Might be needed!
                //transforms.RemoveAt(0);

                if(Skeleton == null) {
                    //Create Source Data
                    asset.Source = new MotionAsset.Hierarchy(transforms.Count);
                    Debug.Log(transforms.Count);
                    for(int i=0; i<asset.Source.Bones.Length; i++) {
                        Debug.Log("Name: " + transforms[i].name + " , Parent: " + transforms[i].parent);
                        asset.Source.SetBone(i, transforms[i].name, i==0 ? "None" : transforms[i].parent.name);
                    }

                    //Compute Frames
                    Matrix4x4[] transformations = new Matrix4x4[asset.Source.Bones.Length];
                    for(int i=0; i<asset.GetTotalFrames(); i++) {
                        clip.SampleAnimation(instance, (float)i / asset.Framerate);
                        for(int j=0; j<transformations.Length; j++) {
                            transformations[j] = transforms[j].GetWorldMatrix();
                        }
                        asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, transformations);
                    }
                } else {
                    //Create Source Data
                    asset.Source = new MotionAsset.Hierarchy(Skeleton.Bones.Length);
                    for(int i=0; i<Skeleton.Bones.Length; i++) {
                        asset.Source.SetBone(i, Skeleton.Bones[i].GetName(), i==0 ? "None" : Skeleton.Bones[i].GetParent().GetName());
                    }

                    //Compute Frames
                    int[] mapping = new int[Skeleton.Bones.Length];
                    for(int i=0; i<Skeleton.Bones.Length; i++) {
                        // mapping[i] = transforms.FindIndex(x => x.name == Skeleton.Bones[i].GetName());
                        // if(mapping[i] == -1) {
                        //     Debug.LogWarning("Could not find mapping for skeleton bone: " + Skeleton.Bones[i].GetName());
                        // }
                        mapping[i] = transforms.FindIndex(x => x.name == Mappings[i]);
                    }
                    Matrix4x4[] transformations = new Matrix4x4[Skeleton.Bones.Length];
                    for(int i=0; i<asset.GetTotalFrames(); i++) {
                        clip.SampleAnimation(instance, (float)i / asset.Framerate);
                        for(int j=0; j<transformations.Length; j++) {
                            transformations[j] = mapping[j] == -1 ? Matrix4x4.identity : transforms[mapping[j]].GetWorldMatrix();
                        }
                        asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, transformations);
                    }
                }

                //Remove Instance
                Utility.Destroy(instance);
                
                //Detect Symmetry
                asset.DetectSymmetry();

                //Add Sequence
                asset.AddSequence();

                //Add Scene
                asset.CreateScene();

                //Remove source FBX
                AssetDatabase.DeleteAsset(relativePath);

                //Save
                EditorUtility.SetDirty(asset);

                Imported.Add(target);
            } else {
                Skipped.Add(target);
            }
            
            yield return new WaitForSeconds(0f);
        }

        public override void BatchCallback() {
            AssetDatabase.SaveAssets();
            Resources.UnloadUnusedAssets();
        }

        public override void DerivedFinish() {
            if(Imported.Count > 0) {
                AssetDatabase.Refresh();
            }

            Debug.Log("Imported " + Imported.Count + " assets.");
            Imported.ToArray().Print();

            Debug.Log("Skipped " + Skipped.Count + " assets.");
            Skipped.ToArray().Print();
        }

    }
}
#endif
