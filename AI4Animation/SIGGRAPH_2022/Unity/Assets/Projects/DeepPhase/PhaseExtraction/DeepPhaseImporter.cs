#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;

namespace AI4Animation {
	public class DeepPhaseImporter : EditorWindow {

		public static EditorWindow Window;
		public static Vector2 Scroll;

        public string SequencePath = string.Empty;
        public string PhasePath = string.Empty;
        public string Tag = string.Empty;

		private MotionEditor Editor = null;
        private EditorCoroutines.EditorCoroutine Coroutine = null;
        private int Count = 0;
        private int BatchSize = 1000;

		[MenuItem ("AI4Animation/Importer/Deep Phase Importer")]
		static void Init() {
			Window = EditorWindow.GetWindow(typeof(DeepPhaseImporter));
			Scroll = Vector3.zero;
		}

		public MotionEditor GetEditor() {
			if(Editor == null) {
				Editor = GameObjectExtensions.Find<MotionEditor>(true);
			}
			return Editor;
		}


        public void OnInspectorUpdate() {
            Repaint();
        }

		void OnGUI() {
			Scroll = EditorGUILayout.BeginScrollView(Scroll);

			if(GetEditor() == null) {
				EditorGUILayout.LabelField("No editor available in scene.");
				return;
			}

			Utility.SetGUIColor(UltiDraw.Black);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Grey);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();

					Utility.SetGUIColor(UltiDraw.Mustard);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.LabelField(this.GetType().ToString());
					}

                    EditorGUILayout.ObjectField("Editor", Editor, typeof(MotionEditor), true);

                    SequencePath = EditorGUILayout.TextField("Sequence Path", SequencePath);
                    PhasePath = EditorGUILayout.TextField("Phase Path", PhasePath);
                    Tag = EditorGUILayout.TextField("Tag", Tag);

                    if(Coroutine == null) {
                        if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
                            Coroutine = EditorCoroutines.StartCoroutine(Process(), this);
                        }
                    } else {
                        EditorGUILayout.LabelField("Read " + Count + " lines...");
                        if(Utility.GUIButton("Stop", UltiDraw.DarkGrey, UltiDraw.White)) {
                            Coroutine = null;
                        }
                    }
				}
			}

			EditorGUILayout.EndScrollView();
		}

        public IEnumerator Process() {
            StreamReader phaseFile = new StreamReader(PhasePath);
            StreamReader sequenceFile = new StreamReader(SequencePath);
            yield return new WaitForSeconds(0f);

            DeepPhaseModule module = null;

            Count = 0;
            while(Coroutine != null && !sequenceFile.EndOfStream) {
                string sLine = sequenceFile.ReadLine();
                string pLine = phaseFile.ReadLine();

                string[] tags = FileUtility.LineToArray(sLine, ' ');
                float[] features = FileUtility.LineToFloat(pLine, ' ');
                // tags.Print();
                string fileGUID = tags[4];
                bool fileMirrored = tags[2] == "Standard" ? false : true;
                int fileFrame = tags[1].ToInt();
                int channels = features.Length / 4;
                if(Editor.Asset != fileGUID) {
                    Editor.LoadSession(fileGUID);
                    MotionAsset asset = Editor.GetSession().Asset;
                    asset.RemoveModule<DeepPhaseModule>(Tag);
                    module = asset.AddModule<DeepPhaseModule>(Tag);
                    module.CreateChannels(channels);
                }

                for(int i=0; i<channels; i++) {
                    float phaseValue = Mathf.Repeat(features[0*channels+i], 1f);
                    float frequency = features[1*channels+i];
                    float amplitude = features[2*channels+i];
                    float offset = features[3*channels+i];
                    if(fileMirrored) {
                        module.Channels[i].MirroredPhaseValues[fileFrame-1] = phaseValue;
                        module.Channels[i].MirroredFrequencies[fileFrame-1] = frequency;
                        module.Channels[i].MirroredAmplitudes[fileFrame-1] = amplitude;
                        module.Channels[i].MirroredOffsets[fileFrame-1] = offset;
                    } else {
                        module.Channels[i].RegularPhaseValues[fileFrame-1] = phaseValue;
                        module.Channels[i].RegularFrequencies[fileFrame-1] = frequency;
                        module.Channels[i].RegularAmplitudes[fileFrame-1] = amplitude;
                        module.Channels[i].RegularOffsets[fileFrame-1] = offset;
                    }
                }

                Count += 1;
                if(Count % BatchSize == 0) {
                    yield return new WaitForSeconds(0f);
                }
            }
            yield return new WaitForSeconds(0f);
        }
	}
}
#endif