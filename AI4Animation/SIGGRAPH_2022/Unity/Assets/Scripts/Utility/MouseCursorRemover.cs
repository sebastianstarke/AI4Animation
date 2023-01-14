#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System.IO;
using System.Collections;
using System.Collections.Generic;

public class MouseCursorRemover : EditorWindow {
    public static EditorWindow Window;
	public static Vector2 Scroll;

    public string ReferenceImage = string.Empty;
    public string CursorIcon = string.Empty;
    public RectInt Rect = new RectInt();

	public string Source = string.Empty;
	public string Destination = string.Empty;

    public FileInfo[] Files = new FileInfo[0];

	private int Page = 1;
	private const int ItemsPerPage = 25;
    private bool Processing = false;
    private FileInfo CurrentFile = null;

    private Texture2D Image = null;
    private Texture2D Pattern = null;
    private Texture2D Extracted = null;
    private Texture2D Filtered = null;

	[MenuItem ("AI4Animation/Tools/Mouse Cursor Remover")]
	static void Init() {
		Window = EditorWindow.GetWindow(typeof(MouseCursorRemover));
		Scroll = Vector3.zero;
	}

	void OnGUI() {
		Scroll = EditorGUILayout.BeginScrollView(Scroll);

		Utility.SetGUIColor(UltiDraw.Black);
		using(new EditorGUILayout.VerticalScope ("Box")) {
			Utility.ResetGUIColor();

			Utility.SetGUIColor(UltiDraw.Grey);
			using(new EditorGUILayout.VerticalScope ("Box")) {
				Utility.ResetGUIColor();

				Utility.SetGUIColor(UltiDraw.Orange);
				using(new EditorGUILayout.VerticalScope ("Box")) {
					Utility.ResetGUIColor();
					EditorGUILayout.LabelField("Mouse Cursor Remover");
				}

				using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Reference Image");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					ReferenceImage = EditorGUILayout.TextField(ReferenceImage);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						ReferenceImage = EditorUtility.OpenFilePanel("Mouse Cursor Remover", ReferenceImage == string.Empty ? Application.dataPath : ReferenceImage, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.LabelField("Cursor Icon");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					CursorIcon = EditorGUILayout.TextField(CursorIcon);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						CursorIcon = EditorUtility.OpenFilePanel("Mouse Cursor Remover", CursorIcon == string.Empty ? Application.dataPath : CursorIcon, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

                    Rect = EditorGUILayout.RectIntField("Rect", Rect);

					if(Utility.GUIButton("Precompute", UltiDraw.DarkGrey, UltiDraw.White)) {
						Precompute();
					}

                    if(Pattern != null) {
                        EditorGUILayout.LabelField("Pattern");
                        Rect rect = EditorGUILayout.GetControlRect();
                        rect.width = (float)Pattern.width / (float)Pattern.height * rect.height;
                        GUI.DrawTexture(rect, Pattern);
                    }

                    if(Extracted != null) {
                        EditorGUILayout.LabelField("Extracted");
                        Rect rect = EditorGUILayout.GetControlRect();
                        rect.width = (float)Extracted.width / (float)Extracted.height * rect.height;
                        GUI.DrawTexture(rect, Extracted);
                    }

                    if(Filtered != null) {
                        EditorGUILayout.LabelField("Filtered");
                        Rect rect = EditorGUILayout.GetControlRect();
                        rect.width = (float)Filtered.width / (float)Filtered.height * rect.height;
                        GUI.DrawTexture(rect, Filtered);
                    }

                    if(Image != null) {
                        EditorGUILayout.LabelField("Reference Image");
                        Rect rect = EditorGUILayout.GetControlRect();
                        rect.width = (float)Image.width / (float)Image.height * rect.height;
                        GUI.DrawTexture(rect, Image);
                    }
                }

				if(!Processing) {
					if(Utility.GUIButton("Load Directory", UltiDraw.DarkGrey, UltiDraw.White)) {
						LoadDirectory();
					}
					if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
						this.StartCoroutine(Process());
					}
					if(Extracted != null && Pattern != null && Utility.GUIButton("Reapply Filter", UltiDraw.DarkGrey, UltiDraw.White)) {
						Filtered = FilterPatch(Extracted, Pattern);
					}
				} else {
					if(Utility.GUIButton("Stop", UltiDraw.DarkRed, UltiDraw.White)) {
						this.StopAllCoroutines();
						Processing = false;
					}
                    if(CurrentFile != null) {
                        EditorGUILayout.LabelField("Current File: " + CurrentFile.Name);
                    }
				}

                using(new EditorGUILayout.VerticalScope ("Box")) {
					EditorGUILayout.LabelField("Source");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					Source = EditorGUILayout.TextField(Source);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Source = EditorUtility.OpenFolderPanel("Mouse Cursor Remover", Source == string.Empty ? Application.dataPath : Source, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					EditorGUILayout.LabelField("Destination");
					EditorGUILayout.BeginHorizontal();
					EditorGUILayout.LabelField("<Path>", GUILayout.Width(50));
					Destination = EditorGUILayout.TextField(Destination);
					GUI.skin.button.alignment = TextAnchor.MiddleCenter;
					if(GUILayout.Button("O", GUILayout.Width(20))) {
						Destination = EditorUtility.OpenFolderPanel("Mouse Cursor Remover", Destination == string.Empty ? Application.dataPath : Destination, "");
						GUIUtility.ExitGUI();
					}
					EditorGUILayout.EndHorizontal();

					int start = (Page-1)*ItemsPerPage;
					int end = Mathf.Min(start+ItemsPerPage, Files.Length);
					int pages = Mathf.CeilToInt(Files.Length/ItemsPerPage)+1;
					Utility.SetGUIColor(UltiDraw.Orange);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Max(Page-1, 1);
						}
						EditorGUILayout.LabelField("Page " + Page + "/" + pages);
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White)) {
							Page = Mathf.Min(Page+1, pages);
						}
						EditorGUILayout.EndHorizontal();
					}

					for(int i=start; i<end; i++) {
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(20f));
							EditorGUILayout.LabelField(Files[i].Name);
							EditorGUILayout.EndHorizontal();
						}
					}
				}

			}
		}

		EditorGUILayout.EndScrollView();
	}

	private void LoadDirectory() {
		if(Directory.Exists(Source)) {
			DirectoryInfo info = new DirectoryInfo(Source);
			Files = info.GetFiles("*.jpg");
		} else {
			Files = new FileInfo[0];
		}
		Page = 1;
	}

    private void Precompute() {
        Pattern = LoadImage(CursorIcon);
        Image = LoadImage(ReferenceImage);
        Color[] patternPixels = Pattern.GetPixels();
        Rect = new RectInt(0, 0, Pattern.width, Pattern.height);
        float score = 0f;
        for(int x=0; x<Image.width-Pattern.width; x++) {
            for(int y=0; y<Image.height-Pattern.height; y++) {
                float pivotScore = CrossCorrelation(x, y);
                if(pivotScore > score) {
                    Rect.x = x;
                    Rect.y = y;
                    score = pivotScore;
                }
            }
        }

        Extracted = ReadPatch(Image, Rect);
        Filtered = FilterPatch(Extracted, Pattern);

        float CrossCorrelation(int x, int y) {
            float value = 0f;
            int count = 0;
            Color[] imagePixels = Image.GetPixels(x, y, Pattern.width, Pattern.height);
            for(int i=0; i<patternPixels.Length; i++) {
                float a = patternPixels[i].a;
                if(a == 1f) {
                    float r = Mathf.Abs(patternPixels[i].r - imagePixels[i].r);
                    float g = Mathf.Abs(patternPixels[i].g - imagePixels[i].g);
                    float b = Mathf.Abs(patternPixels[i].b - imagePixels[i].b);
                    value += (r+g+b) / 3f;
                    count += 1;
                }
            }
            return 1f - value / count;
        }
    }

    private Texture2D FilterPatch(Texture2D patch, Texture2D pattern) {
        int width = patch.width;
        Color[] patchColors = patch.GetPixels();
        Color[] patternColors = pattern.GetPixels();
        
        //Get all candidate neighbours
        List<Coord> tmpTruth = new List<Coord>();
        List<Coord> tmpFake = new List<Coord>();
        for(int x=0; x<patch.width; x++) {
            for(int y=0; y<patch.height; y++) {
                if(patternColors[GridToArray(x,y)].a == 0f) {
                    tmpTruth.Add(new Coord(x,y));
                } else {
                    tmpFake.Add(new Coord(x,y));
                }
            }
        }
        Coord[] truth = tmpTruth.ToArray();
        Coord[] fake = tmpFake.ToArray();

        //Interpolate
        Color[] result = new Color[patch.width * patch.height];
        for(int x=0; x<patch.width; x++) {
            for(int y=0; y<patch.height; y++) {
                if(patternColors[GridToArray(x,y)].a == 1f) {

                    System.Array.Sort(truth,
                        delegate(Coord a, Coord b) {
                            return a.Distance(x,y).CompareTo(b.Distance(x,y));
                        }
                    );

                    List<Vector2Int> pivots = new List<Vector2Int>();
                    int index = 0;
                    while(Mathf.Floor(truth[index].Distance(x,y)) <= Mathf.Floor(truth.First().Distance(x,y))) {
                        pivots.Add(truth[index].Coords);
                        index += 1;
                    }
                    pivots.AddRange(new Vector2Int[4]{ClosestLeft(x,y), ClosestRight(x,y), ClosestBottom(x,y), ClosestTop(x,y)});

                    float sum = 0f;
                    Vector3 color = Vector3.zero;
                    for(int i=0; i<pivots.Count; i++) {
                        float w = 1f/Vector2Int.Distance(new Vector2Int(x,y), pivots[i]);
                        sum += w;
                        Color c = patchColors[GridToArray(pivots[i].x, pivots[i].y)];
                        color.x += w * c.r;
                        color.y += w * c.g;
                        color.z += w * c.b;
                    }
                    color /= sum;

                    result[GridToArray(x,y)] = new Color(color.x, color.y, color.z, 1f);
                } else {
                    result[GridToArray(x,y)] = patchColors[GridToArray(x,y)];
                }
            }
        }

        Color[] smoothed = new Color[patch.width * patch.height];
        for(int x=0; x<patch.width; x++) {
            for(int y=0; y<patch.height; y++) {
                if(patternColors[GridToArray(x,y)].a == 1f) {
                    List<Color> colors = new List<Color>();
                    for(int i=-1; i<=1; i++) {
                        for(int j=-1; j<=1; j++) {
                            int coordX = x + i;
                            int coordY = y + j;
                            if(coordX > 0 && coordX < patch.width && coordY > 0 && coordY < patch.height) {
                                colors.Add(result[GridToArray(coordX,coordY)]);
                            }
                        }
                    }
                    Color avg = AverageColor(colors.ToArray());
                    smoothed[GridToArray(x,y)] = avg;
                } else {
                    smoothed[GridToArray(x,y)] = result[GridToArray(x,y)];
                }
            }
        }

        // Color[] smoothed = new Color[patch.width * patch.height];
        // Color[] colors = new Color[fake.Length];
        // for(int i=0; i<colors.Length; i++) {
        //     colors[i] = result[GridToArray(fake[i].Coords.x, fake[i].Coords.y)];
        // }
        // Color average = AverageColor(colors);
        // for(int x=0; x<patch.width; x++) {
        //     for(int y=0; y<patch.height; y++) {
        //         if(patternColors[GridToArray(x,y)].a == 1f) {

        //             System.Array.Sort(truth,
        //                 delegate(Coord a, Coord b) {
        //                     return a.Distance(x,y).CompareTo(b.Distance(x,y));
        //                 }
        //             );
        //             float distanceToBoundary = Vector2Int.Distance(new Vector2Int(x,y), truth.First().Coords);
        //             float w = distanceToBoundary;
        //             Debug.Log(w);
        //             smoothed[GridToArray(x,y)] = Color.Lerp(result[GridToArray(x,y)], average, w);
        //         } else {
        //             smoothed[GridToArray(x,y)] = result[GridToArray(x,y)];
        //         }
        //     }
        // }

        Texture2D filtered = new Texture2D(patch.width, patch.height);
        filtered.filterMode = FilterMode.Trilinear;
        filtered.SetPixels(smoothed);
        filtered.Apply();
        return filtered;

        int GridToArray(int x, int y) {
            return y * width + x;
        }

        Color AverageColor(Color[] values) {
            Vector3 avg = Vector3.zero;
            foreach(Color c in values) {
                avg.x += c.r;
                avg.y += c.g;
                avg.z += c.b;
            }
            avg /= values.Length;
            return new Color(avg.x, avg.y, avg.z, 1f);
        }

        Vector2Int ClosestLeft(int x, int y) {
            while(x > 0) {
                x -= 1;
                if(patternColors[GridToArray(x,y)].a == 0f) {
                    return new Vector2Int(x,y);
                }
            }
            Debug.Log("Closest left could not be found.");
            return new Vector2Int(x,y);
        }

        Vector2Int ClosestRight(int x, int y) {
            while(x < patch.width-1) {
                x += 1;
                if(patternColors[GridToArray(x,y)].a == 0f) {
                    return new Vector2Int(x,y);
                }
            }
            Debug.Log("Closest right could not be found.");
            return new Vector2Int(x,y);
        }

        Vector2Int ClosestBottom(int x, int y) {
            while(y > 0) {
                y -= 1;
                if(patternColors[GridToArray(x,y)].a == 0f) {
                    return new Vector2Int(x,y);
                }
            }
            Debug.Log("Closest bottom could not be found.");
            return new Vector2Int(x,y);
        }

        Vector2Int ClosestTop(int x, int y) {
            while(y < patch.height-1) {
                y += 1;
                if(patternColors[GridToArray(x,y)].a == 0f) {
                    return new Vector2Int(x,y);
                }
            }
            Debug.Log("Closest top could not be found.");
            return new Vector2Int(x,y);
        }
    }

    class Coord {
        public Vector2Int Coords;
        public Coord(int x, int y) {
            Coords = new Vector2Int(x,y);
        }
        public float Distance(int x, int y) {
            return Vector2Int.Distance(new Vector2Int(x,y), Coords);
        }
    }

    private void WritePatch(Texture2D tex, Texture2D patch, RectInt rect) {
        tex.SetPixels(rect.x, rect.y, rect.width, rect.height, patch.GetPixels());
        tex.Apply();
    }

    private Texture2D ReadPatch(Texture2D tex, RectInt rect) {
        Texture2D result = new Texture2D(rect.width, rect.height);
        result.SetPixels(tex.GetPixels(rect.x, rect.y, rect.width, rect.height));
        result.Apply();
        return result;
    }

    private IEnumerator Process() {
        Processing = true;
        for(int i=0; i<Files.Length; i++) {
            CurrentFile = Files[i];
            Texture2D tex = LoadImage(Files[i].FullName);
            Texture2D patch = ReadPatch(tex, Rect);
            Texture2D filtered = FilterPatch(patch, Pattern);
            WritePatch(tex, filtered, Rect);
            SaveImage(tex, Destination + "/" + Files[i].Name);
            yield return new WaitForSeconds(0f);
        }
        Processing = false;
    }

    private Texture2D LoadImage(string path) {
        Texture2D tex = null;
        byte[] fileData;
        if(File.Exists(path))     {
            fileData = File.ReadAllBytes(path);
            tex = new Texture2D(2, 2);
            tex.LoadImage(fileData);
        }
        return tex;
    }

    private void SaveImage(Texture2D tex, string path) {
        //Texture2D texture = new Texture2D(width, height, TextureFormat.RGB24, false);
        File.WriteAllBytes(path, tex.EncodeToJPG(100));
    }
}
#endif