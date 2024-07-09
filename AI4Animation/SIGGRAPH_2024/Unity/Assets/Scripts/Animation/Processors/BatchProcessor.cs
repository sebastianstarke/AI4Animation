#if UNITY_EDITOR
using UnityEngine;
using UnityEditor;
using System;
using System.Collections;
using System.Collections.Generic;

namespace AI4Animation {
	public abstract class BatchProcessor : EditorWindow {

		[System.Serializable]
		public class Item {
			public Item(string id, int index) {
				ID = id;
				Index = index;

				Selected = true;
				Processed = false;
				Inspect = false;
			}

			public string ID = string.Empty;
			public int Index = 0;

			public bool Selected = true;
			public bool Processed = false;
			public bool Inspect = false;
		}

		public static EditorWindow Window;
		public static Vector2 Scroll;
		public static int Page = 1;
		public static int BatchSize = 100;
		private static int ItemsPerPage = 25;

		[NonSerialized] private bool Initialized = false;

		[NonSerialized] private Item[] Items = new Item[0];

		[NonSerialized] private string Filter = string.Empty;
		[NonSerialized] private Item[] Instances = new Item[0];

		[NonSerialized] private bool Processing = false;
		[NonSerialized] private bool Working = false;
		[NonSerialized] private float Progress = 0f;

		public void OnInspectorUpdate() {
			if(!Initialized) {
				DerivedRefresh();
				Initialized = true;
			}
			Repaint();
		}

		public void LoadItems(string[] names) {
			Items = new Item[names.Length];
			for(int i=0; i<Items.Length; i++) {
				Items[i] = new Item(names[i], i);
			}
			ApplyFilter(string.Empty);
		}

		public void LoadItems(Item[] items) {
			Items = items;
			ApplyFilter(string.Empty);
		}

		public Item[] GetItems() {
			return Items;
		}

		public bool IsProcessing() {
			return Processing;
		}

		public bool IsAborting() {
			return !Processing && Working;
		}

		private void ApplyFilter(string filter) {
			Filter = filter;
			if(Filter == string.Empty) {
				Instances = Items;
			} else {
				List<Item> instances = new List<Item>();
				for(int i=0; i<Items.Length; i++) {
					if(GetID(Items[i]).ToLowerInvariant().Contains(Filter.ToLowerInvariant())) {
						instances.Add(Items[i]);
					}
				}
				Instances = instances.ToArray();
			}
			LoadPage(1);
		}

		private void LoadPage(int page) {
			Page = Mathf.Clamp(page, 1, GetPageCount());
		}

		private int GetPageCount() {
			return Mathf.CeilToInt(Instances.Length/ItemsPerPage)+1;
		}

		private int GetPageStart() {
			return (Page-1)*ItemsPerPage;
		}

		private int GetPageEnd() {
			return Mathf.Min(Page*ItemsPerPage, Instances.Length);
		}

		private int GetSelectedCount() {
			int count = 0;
			for(int i=0; i<Items.Length; i++) {
				count += Items[i].Selected ? 1 : 0;
			}
			return count;
		}

		void OnGUI() {
			Scroll = EditorGUILayout.BeginScrollView(Scroll);

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

					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						DerivedInspector();
					}

					Utility.SetGUIColor(UltiDraw.LightGrey);
					using(new EditorGUILayout.VerticalScope ("Box")) {
						Utility.ResetGUIColor();
						EditorGUILayout.BeginHorizontal();

						EditorGUILayout.LabelField("Page", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						int page = EditorGUILayout.IntField(Page, GUILayout.Width(40f));
						if(EditorGUI.EndChangeCheck()) {
							LoadPage(page);
						}
						EditorGUILayout.LabelField("/" + GetPageCount(), GUILayout.Width(40f));					
						EditorGUILayout.LabelField("Filter", GUILayout.Width(40f));
						EditorGUI.BeginChangeCheck();
						string filter = EditorGUILayout.TextField(Filter, GUILayout.Width(200f));
						if(EditorGUI.EndChangeCheck()) {
							ApplyFilter(filter);
						}
						if(Utility.GUIButton("<", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Max(Page-1, 1));
						}
						if(Utility.GUIButton(">", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							LoadPage(Mathf.Min(Page+1, GetPageCount()));
						}
						if(Utility.GUIButton("Enable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Item i in Instances) {
								i.Selected = true;
							}
						}
						if(Utility.GUIButton("Disable All", UltiDraw.DarkGrey, UltiDraw.White, 80f, 16f)) {
							foreach(Item i in Instances) {
								i.Selected = false;
							}
						}
						EditorGUILayout.LabelField("Selected: " + GetSelectedCount());
						EditorGUILayout.EndHorizontal();
					}

					if(Processing || IsAborting()) {
						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUI.BeginDisabledGroup(IsAborting());
							if(Utility.GUIButton(IsAborting() ? "Aborting" : "Stop", IsAborting() ? UltiDraw.Gold : UltiDraw.DarkRed, UltiDraw.White)) {
								Processing = false;
							}
							EditorGUI.EndDisabledGroup();
						}
						EditorGUI.DrawRect(
							new Rect(
								EditorGUILayout.GetControlRect().x, 
								EditorGUILayout.GetControlRect().y, 
								Progress * EditorGUILayout.GetControlRect().width, 25f
							), 
							UltiDraw.Green.Opacity(0.75f)
						);
					} else {
						Utility.SetGUIColor(UltiDraw.LightGrey);
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUI.BeginDisabledGroup(!CanProcess());
							if(Utility.GUIButton("Process", UltiDraw.DarkGrey, UltiDraw.White)) {
								this.StartCoroutine(Process());
							}
							EditorGUI.EndDisabledGroup();
						}
					}

					for(int i=GetPageStart(); i<GetPageEnd(); i++) {
						if(Instances[i].Processed) {
							Utility.SetGUIColor(UltiDraw.DarkGreen);
						} else if(Instances[i].Selected) {
							Utility.SetGUIColor(UltiDraw.Gold);
						} else {
							Utility.SetGUIColor(UltiDraw.DarkRed);
						}
						using(new EditorGUILayout.VerticalScope ("Box")) {
							Utility.ResetGUIColor();
							EditorGUILayout.BeginHorizontal();
							EditorGUILayout.LabelField((i+1).ToString(), GUILayout.Width(30f));
							EditorGUI.BeginChangeCheck();
							Instances[i].Selected = EditorGUILayout.Toggle(Instances[i].Selected, GUILayout.Width(20f));
							if(EditorGUI.EndChangeCheck()) {
								DerivedInspector(Instances[i]);
							}
							if(Utility.GUIButton("Inspect", Instances[i].Inspect ? UltiDraw.Cyan : UltiDraw.DarkGrey, Instances[i].Inspect ? UltiDraw.Black : UltiDraw.White, 60f, 18f)) {
								Instances[i].Inspect = !Instances[i].Inspect;
							}
							EditorGUILayout.LabelField(GetID(Instances[i]));
							EditorGUILayout.EndHorizontal();
							if(Instances[i].Inspect) {
								DerivedInspector(Instances[i]);
							}
						}
					}

				}
			}

			EditorGUILayout.EndScrollView();
		}
		
		public IEnumerator Process() {
			if(!CanProcess()) {
				yield return new WaitForSeconds(0f);
			} else {
				Processing = true;
				Progress = 0f;
				foreach(Item i in Items) {
					i.Processed = false;
				}
				Working = true;
				try {
					DerivedStart();
				} catch(Exception e) {
					Stop(e);
				}
				int batch = 0;
				for(int i=0; i<Items.Length; i++) {
					if(!Processing) {break;}
					if(Items[i].Selected) {
						EditorCoroutines.EditorCoroutine c = null;
						try {
							c = EditorCoroutines.StartCoroutine(DerivedProcess(Items[i]), this);
						} catch(Exception e) {
							Stop(e);
						}
						while(!c.finished) {
							if(!Processing) {break;}
							yield return new WaitForSeconds(0f);
						}
						Items[i].Processed = true;
						yield return new WaitForSeconds(0f);
						batch += 1;
						if(batch == BatchSize) {
							try {
								BatchCallback();
							} catch(Exception e) {
								Stop(e);
							}
							batch = 0;
						}
					}
					Progress = (float)(i+1) / (float)Items.Length;
				}
				try {
					BatchCallback();
				} catch(Exception e) {
					Stop(e);
				}
				try {
					DerivedFinish();
				} catch(Exception e) {
					Stop(e);
				}
				Working = false;
				foreach(Item i in Items) {
					i.Processed = false;
				}
				Processing = false;
				Progress = 0f;
			}
		}
		public void Stop(Exception exception=null) {
			if(exception != null) {
				Debug.LogError(exception);
			}
			EditorCoroutines.StopAllCoroutines(this);
			Working = false;
			Processing = false;
			Progress = 0f;
			foreach(Item i in Items) {
				i.Processed = false;
			}
		}

		public abstract string GetID(Item item);
		public abstract void DerivedRefresh();
		public abstract void DerivedInspector();
		public abstract void DerivedInspector(Item item);
		public abstract bool CanProcess();
		public abstract void DerivedStart();
		public abstract IEnumerator DerivedProcess(Item item);
		public abstract void BatchCallback();
		public abstract void DerivedFinish();
		
	}
}
#endif