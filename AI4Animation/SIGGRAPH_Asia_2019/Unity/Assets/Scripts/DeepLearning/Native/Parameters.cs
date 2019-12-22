using UnityEngine;
using System.IO;
using System.Collections.Generic;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepLearning {

    public class Parameters : ScriptableObject {
        public Buffer[] Buffers = new Buffer[0];

        #if UNITY_EDITOR
        public static void Import(string folder) {
            if(!Directory.Exists(folder)) {
                Debug.Log("Folder " + folder + " does not exist.");
                return;
            }
            string[] files = Directory.GetFiles(folder);
            string directory = new FileInfo(files[0]).Directory.Name;
            Parameters asset = ScriptableObject.CreateInstance<Parameters>();
            string path = AssetDatabase.GenerateUniqueAssetPath("Assets/" + directory + ".asset");
            foreach(string file in files) {
                string id = Path.GetFileNameWithoutExtension(file);
                asset.Import(file, id);
            }
            AssetDatabase.CreateAsset(asset, path);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
		    Selection.activeObject = asset;
        }
        #endif

        public void Import(string fn, string id) {
            for(int i=0; i<Buffers.Length; i++) {
                if(Buffers[i] != null) {
                    if(Buffers[i].ID == id) {
                        Debug.Log("Buffer with ID " + id + " already contained.");
                        return;
                    }
                }
            }
            ArrayExtensions.Add(ref Buffers, ReadBinary(fn, id));
        }

        public Buffer Load(string id) {
            Buffer buffer = System.Array.Find(Buffers, x => x.ID == id);
            if(buffer == null) {
                Debug.Log("Buffer with ID " + id + " not found.");
            }
            return buffer;
        }

        public void Clear() {
            ArrayExtensions.Resize(ref Buffers, 0);
        }

        private Buffer ReadBinary(string fn, string id) {
            if(File.Exists(fn)) {
                List<float> values = new List<float>();
                BinaryReader reader = new BinaryReader(File.Open(fn, FileMode.Open));
                while(true) {
                    try {
                        values.Add(reader.ReadSingle());
                    } catch {
                        break;
                    }
                }
                reader.Close();
                return new Buffer(id, values.ToArray());
            } else {
                Debug.Log("File at path " + fn + " does not exist.");
                return null;
            }
        }

        [System.Serializable]
        public class Buffer {
            public string ID;
            public float[] Values;
            public Buffer(string id, float[] values) {
                ID = id;
                Values = values;
            }
        }
    }
    
    #if UNITY_EDITOR
    public class MyWindow : EditorWindow {

        private string Folder = string.Empty;

        [MenuItem("AI4Animation/Parameter Importer")]
        static void Init() {
            ((MyWindow)EditorWindow.GetWindow(typeof(MyWindow))).Show();
        }

        void OnGUI() {
            GUILayout.Label("Folder", EditorStyles.boldLabel);
            Folder = EditorGUILayout.TextField(Folder);
            if(Utility.GUIButton("Import", Color.grey, Color.black)) {
                Parameters.Import(Folder);
            }
        }

    }
    #endif

}