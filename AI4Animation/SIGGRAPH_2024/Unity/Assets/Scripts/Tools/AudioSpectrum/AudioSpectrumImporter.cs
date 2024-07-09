#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public class AudioSpectrumImporter : BatchProcessor {

		public string Source = string.Empty;
		public string Destination = string.Empty;
        public int Framerate = 60;

		[MenuItem ("AI4Animation/Importer/Audio Spectrum Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(AudioSpectrumImporter));
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
				Source = EditorUtility.OpenFolderPanel("Audio Spectrum Importer", Source == string.Empty ? Application.dataPath : Source, "");
				GUIUtility.ExitGUI();
			}
			EditorGUILayout.EndHorizontal();

			EditorGUILayout.LabelField("Destination");
			EditorGUILayout.BeginHorizontal();
			EditorGUILayout.LabelField("Assets/", GUILayout.Width(50));
			Destination = EditorGUILayout.TextField(Destination);
			EditorGUILayout.EndHorizontal();

            Framerate = EditorGUILayout.IntField("Framerate", Framerate);

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
						foreach(FileInfo i in info.GetFiles("*.wav")) {
							paths.Add(i.FullName);
						}
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

		}

		public override IEnumerator DerivedProcess(Item item) {
			string destination = "Assets/" + Destination;

			string fullFileName = Path.GetFileName(item.ID);
			string rawFileName = Path.GetFileNameWithoutExtension(item.ID);

			string wavDestination = destination + "/wav";
			if(!Directory.Exists(wavDestination)) {
				Directory.CreateDirectory(wavDestination);
            }
			AudioClip clip = AssetDatabase.LoadAssetAtPath<AudioClip>(wavDestination + "/" + fullFileName);
			if(clip == null) {
				FileUtil.CopyFileOrDirectory(item.ID, wavDestination + "/" + fullFileName);
				AssetDatabase.ImportAsset(wavDestination + "/" + fullFileName);
				clip = AssetDatabase.LoadAssetAtPath<AudioClip>(wavDestination + "/" + fullFileName);
			}

			string featuresDestination = destination + "/features";
			if(!Directory.Exists(featuresDestination)) {
				Directory.CreateDirectory(featuresDestination);
            }
            AudioSpectrum asset = AssetDatabase.LoadAssetAtPath<AudioSpectrum>(featuresDestination + "/" + rawFileName + ".asset");
            if(asset == null) {
                asset = ScriptableObjectExtensions.Create<AudioSpectrum>(featuresDestination, rawFileName, true);
            }

			string path = Path.GetDirectoryName(item.ID) + "/features/" + rawFileName + "/";
            float[][] spectogram = Utility.LoadTxt(path+"spectogram.txt").ToFloat();
            float[][] beats = Utility.LoadTxt(path+"beats.txt").ToFloat();
            float[][] flux = Utility.LoadTxt(path+"flux.txt").ToFloat();
			float[][] mfcc = Utility.LoadTxt(path+"mfcc.txt").ToFloat();
			float[][] chroma = Utility.LoadTxt(path+"chroma.txt").ToFloat();
			float[][] zeroCrossing = Utility.LoadTxt(path+"zerocrossing.txt").ToFloat();

            if(!ArrayExtensions.Same(spectogram.Length, beats.Length, flux.Length)) {
                Debug.Log("Length of arrays is not the same: " + "(" + spectogram.Length + ", " + beats.Length + ", " + flux.Length + ")");
            } else {
				asset.Clip = clip;
                asset.Framerate = Framerate;
                asset.Samples = new AudioSpectrum.Sample[spectogram.Length];
                for(int i=0; i<asset.Samples.Length; i++) {
                    asset.Samples[i] = new AudioSpectrum.Sample(spectogram[i], beats[i], flux[i], mfcc[i], chroma[i], zeroCrossing[i]);
                }
            }

            EditorUtility.SetDirty(asset);

			yield return new WaitForSeconds(0f);
		}
		
		public override void BatchCallback() {
			AssetDatabase.SaveAssets();
			Resources.UnloadUnusedAssets();
		}

		public override void DerivedFinish() {
			AssetDatabase.Refresh();
		}

	}
}
#endif