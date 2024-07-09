#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
    public class AGROLImporter : BatchProcessor {

        public string Source = string.Empty;
        public string Destination = string.Empty;
        public float Framerate = 30f;
        // public Actor Skeleton = null;

        private List<string> Imported;
        private List<string> Skipped;

        [MenuItem ("AI4Animation/Importer/AGROL Importer")]
        static void Init() {
            Window = GetWindow(typeof(AGROLImporter));
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
                SetSource(EditorUtility.OpenFolderPanel("AGROL Importer", Source == string.Empty ? Application.dataPath : Source, ""));
                GUIUtility.ExitGUI();
            }
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.LabelField("Destination");
            EditorGUILayout.BeginHorizontal();
            EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
            Destination = EditorGUILayout.TextField(Destination);
            EditorGUILayout.EndHorizontal();
            Framerate = EditorGUILayout.FloatField("Framerate", Framerate);
            // Skeleton = EditorGUILayout.ObjectField("Skeleton", Skeleton, typeof(Actor), true) as Actor;

            if(Utility.GUIButton("Load Source Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
                LoadDirectory(Source);
            }
        }

        public override void DerivedInspector(Item item) {
        
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
                            if(i.Name.EndsWith(".txt")) {
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

            string target = (destination + item.ID.Remove(0, source.Length)).Replace(".txt", "");
            
            string[] bones = new string[] {
                "m_avg_Pelvis", //0
                "m_avg_L_Hip", //1
                "m_avg_R_Hip", //2
                "m_avg_Spine1", //3
                "m_avg_L_Knee", //4
                "m_avg_R_Knee", //5
                "m_avg_Spine2", //6
                "m_avg_L_Ankle", //7
                "m_avg_R_Ankle", //8
                "m_avg_Spine3", //9
                "m_avg_L_Foot", //10
                "m_avg_R_Foot", //11
                "m_avg_Neck", //12
                "m_avg_L_Collar", //13
                "m_avg_R_Collar", //14
                "m_avg_Head", //15
                "m_avg_L_Shoulder", //16
                "m_avg_R_Shoulder", //17
                "m_avg_L_Elbow", //18
                "m_avg_R_Elbow", //19
                "m_avg_L_Wrist", //20
                "m_avg_R_Wrist" //21
            };

            string[] parents = new string[] {
                "None",
                "m_avg_Pelvis",
                "m_avg_Pelvis",
                "m_avg_Pelvis",
                "m_avg_L_Hip",
                "m_avg_R_Hip",
                "m_avg_Spine1",
                "m_avg_L_Knee",
                "m_avg_R_Knee",
                "m_avg_Spine2",
                "m_avg_L_Ankle",
                "m_avg_R_Ankle",
                "m_avg_Spine3",
                "m_avg_Spine3",
                "m_avg_Spine3",
                "m_avg_Neck",
                "m_avg_L_Collar",
                "m_avg_R_Collar",
                "m_avg_L_Shoulder",
                "m_avg_R_Shoulder",
                "m_avg_L_Elbow",
                "m_avg_R_Elbow"
            };

            if(!Directory.Exists(target)) {
                //Create Directory
                Directory.CreateDirectory(target);
                DirectoryInfo targetInfo = new DirectoryInfo(target);

                //Import source file to folder
                float[][] data = FileUtility.ReadMatrix(item.ID);

                //Create Asset
                MotionAsset asset = CreateInstance<MotionAsset>();
                
                asset.name = targetInfo.Name;
                AssetDatabase.CreateAsset(asset, target+"/"+asset.name+".asset");

                //Set Frames
                ArrayExtensions.Resize(ref asset.Frames, data.Length-1);

                //Set Framerate
                asset.Framerate = Framerate;

                //Create Source Data
                asset.Source = new MotionAsset.Hierarchy(bones, parents);
                // asset.Source = new MotionAsset.Hierarchy(data.Last().Length/2);
                // for(int i=0; i<asset.Source.Bones.Length; i++) {
                //     // if(Skeleton == null) {
                //         // asset.Source.SetBone(i, data.iobt_keypoint_names[i], "None");
                //     // } else {
                //     //     Actor.Bone bone = Skeleton.FindBone(data.iobt_keypoint_names[i]);
                //     //     if(bone == null) {
                //     //         Debug.Log("Bone was null: " + data.iobt_keypoint_names[i]);
                //     //         asset.Source.SetBone(i, data.iobt_keypoint_names[i], "None");
                //     //     } else {
                //     //         asset.Source.SetBone(i, data.iobt_keypoint_names[i], bone.GetParent() == null ? "None" : bone.GetParent().GetName());
                //     //     }
                //     // }
                // }

                //Compute Frames
                Matrix4x4[] transformations = new Matrix4x4[asset.Source.Bones.Length];
                for(int i=0; i<asset.GetTotalFrames(); i++) {
                    float[] values = data[i];
                    for(int j=0; j<transformations.Length; j++) {
                        Vector3 position = new Vector3(
                            values[9*j + 0], 
                            values[9*j + 2],
                            values[9*j + 1]
                        );
                        Vector3 forward = new Vector3(
                            values[9*j + 3], 
                            values[9*j + 5],
                            values[9*j + 4]
                        );
                        Vector3 up = new Vector3(
                            values[9*j + 6], 
                            values[9*j + 8],
                            values[9*j + 7]
                        );
                        Quaternion rotation = Quaternion.LookRotation(forward, up);
                        transformations[j] = Matrix4x4.TRS(position, rotation, Vector3.one);
                    }
                    asset.Frames[i] = new Frame(asset, i+1, (float)i / asset.Framerate, transformations);
                }
                
                //Detect Symmetry
                asset.DetectSymmetry();

                //Add Sequence
                asset.AddSequence();

                //Add Scene
                asset.CreateScene();

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
