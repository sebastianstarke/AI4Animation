using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System;

[Serializable]
public class Hierarchy {

	public Segment[] Segments;

	public void Build(Transform root) {
		Arrays.Clear(ref Segments);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = AddSegment(transform.name, parent);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	public void Build(Transform root, string[] names) {
		Arrays.Clear(ref Segments);
		Action<Transform, Segment> recursion = null;
		recursion = new Action<Transform, Segment>((transform, parent) => {
			Segment segment = Array.Exists(names, x => x == transform.name) ? AddSegment(transform.name, parent) : parent;
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i), segment);
			}
		});
		recursion(root, null);
	}

	public void Build(string[] names, string[] parents) {
		Arrays.Clear(ref Segments);
		for(int i=0; i<names.Length; i++) {
			if(FindSegment(names[i]) != null) {
				Debug.Log("Failed building hierarchy because there were multiple segments with name " + names[i] + ".");
				Arrays.Clear(ref Segments);
				return;
			}
			AddSegment(names[i], FindSegment(parents[i]));
		}
	}

	public bool RebuildRequired(Transform root) {
		if(Segments.Length == 0) {
			return false;
		}
		List<Transform> list = new List<Transform>();
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			list.Add(transform);
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
		Transform[] transforms = list.ToArray();
		for(int i=0; i<Segments.Length; i++) {
			if(!Array.Exists(transforms, x => x.name == Segments[i].GetName())) {
				return true;
			}
		}
		return false;
	}

	public Segment GetRoot() {
		if(Segments.Length == 0) {
			return null;
		}
		return Segments[0];
	}

	public string[] GetNames() {
		string[] names = new string[Segments.Length];
		for(int i=0; i<Segments.Length; i++) {
			names[i] = Segments[i].GetName();
		}
		return names;
	}

	public Segment FindSegment(string name) {
		return Array.Find(Segments, x => x.GetName() == name);
	}

	public void SetWorldTransformations(Matrix4x4[] transformations) {
		for(int i=0; i<Segments.Length; i++) {
			Segments[i].SetTransformation(transformations[i]);
		}
	}

	public Matrix4x4[] GetWorldTransformations() {
		Matrix4x4[] transformations = new Matrix4x4[Segments.Length];
		for(int i=0; i<Segments.Length; i++) {
			transformations[i] = Segments[i].GetTransformation();
		}
		return transformations;
	}

	public void SetLocalTransformations(Matrix4x4[] transformations) {
		for(int i=0; i<Segments.Length; i++) {
			Segment parent = Segments[i].GetParent(this);
			if(parent == null) {
				Segments[i].SetTransformation(transformations[i]);
			} else {
				Segments[i].SetTransformation(parent.GetTransformation() * transformations[i]);
			}
		}
	}

	public Matrix4x4[] GetLocalTransformations() {
		Matrix4x4[] transformations = new Matrix4x4[Segments.Length];
		for(int i=0; i<Segments.Length; i++) {
			Segment parent = Segments[i].GetParent(this);
			if(parent == null) {
				transformations[i] = Segments[i].GetTransformation();
			} else {
				transformations[i] = parent.GetTransformation().inverse * Segments[i].GetTransformation();
			}
		}
		return transformations;
	}

	public Matrix4x4[] WorldToLocal(Matrix4x4[] world) {
		Matrix4x4[] local = new Matrix4x4[Segments.Length];
		for(int i=0; i<Segments.Length; i++) {
			Segment parent = Segments[i].GetParent(this);
			if(parent == null) {
				local[i] = world[i];
			} else {
				local[i] = world[parent.GetIndex()].inverse * world[i];
			}
		}
		return local;
	}

	public Matrix4x4[] LocalToWorld(Matrix4x4[] local) {
		Matrix4x4[] world = new Matrix4x4[Segments.Length];
		for(int i=0; i<Segments.Length; i++) {
			Segment parent = Segments[i].GetParent(this);
			if(parent == null) {
				world[i] = local[i];
			} else {
				world[i] = world[parent.GetIndex()] * local[i];
			}
		}
		return world;
	}

	public void WriteTransformations(Transform root) {
		if(Segments.Length == 0) {
			return;
		}
		int index = 0;
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(index == Segments.Length) {
				return;
			}
			if(transform.name == Segments[index].GetName()) {
				transform.position = Segments[index].GetTransformation().GetPosition();
				transform.rotation = Segments[index].GetTransformation().GetRotation();
				index += 1;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
	}

	public void FetchTransformations(Transform root) {
		if(Segments.Length == 0) {
			return;
		}
		int index = 0;
		Action<Transform> recursion = null;
		recursion = new Action<Transform>((transform) => {
			if(index == Segments.Length) {
				return;
			}
			if(transform.name == Segments[index].GetName()) {
				Segments[index].SetTransformation(Matrix4x4.TRS(transform.position, transform.rotation, Vector3.one));
				index += 1;
			}
			for(int i=0; i<transform.childCount; i++) {
				recursion(transform.GetChild(i));
			}
		});
		recursion(root);
	}

	private Segment AddSegment(string name, Segment parent) {
		Segment segment = new Segment(name, parent, Segments.Length);
		Arrays.Add(ref Segments, segment);
		return segment;
	}
	
	[Serializable]
	public class Segment {
		[SerializeField] private string Name = "Empty";
		[SerializeField] private int Index = -1;
		[SerializeField] private int Parent = -1;
		[SerializeField] private int[] Childs = new int[0];
		[SerializeField] private Matrix4x4 Transformation;

		public Segment(string name, Segment parent, int index) {
			Name = name;
			Index = index;
			if(parent != null) {
				Parent = parent.Index;
				Arrays.Add(ref parent.Childs, index);
			}
			Childs = new int[0];
			Transformation = Matrix4x4.identity;
		}

		public string GetName() {
			return Name;
		}

		public int GetIndex() {
			return Index;
		}

		public Segment GetParent(Hierarchy hierarchy) {
			if(Parent == -1) {
				return null;
			} else {
				return hierarchy.Segments[Parent];
			}
		}

		public Segment GetChild(Hierarchy hierarchy, int index) {
			return hierarchy.Segments[Childs[index]];
		}
		
		public int GetChildCount() {
			return Childs.Length;
		}

		public void SetTransformation(Matrix4x4 transformation) {
			Transformation = transformation;
		}

		public Matrix4x4 GetTransformation() {
			return Transformation;
		}
	}

}
