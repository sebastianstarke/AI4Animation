using UnityEngine;
using System.Collections.Generic;

public static class MeshSlicer
{
	private static Mesher leftSideMesher = new Mesher();
	private static Mesher rightSideMesher = new Mesher();

	private static Plane bladePlane;
	private static Mesh targetMesh;

	private static List<Vector3> verticesNew = new List<Vector3>();

	private static int _capMatSub = 1;


	/// <summary>
	/// Cut the specified target into 2 pieces
	/// </summary>
	public static GameObject[] CutIntoPieces(GameObject target, Vector3 anchorPoint, Vector3 normalDirection, Material capMaterial)
	{
		// set the blade relative to victim
		bladePlane = new Plane(target.transform.InverseTransformDirection(-normalDirection),
			target.transform.InverseTransformPoint(anchorPoint));

		// get the target mesh
		targetMesh = target.GetComponent<MeshFilter>().mesh;

		// reset values
		verticesNew.Clear();

		leftSideMesher = new Mesher();
		rightSideMesher = new Mesher();

		bool[] sides = new bool[3];
		int[] indices;
		int p1, p2, p3;

		// go throught the submeshes
		for (int sub = 0; sub < targetMesh.subMeshCount; sub++)
		{

			indices = targetMesh.GetTriangles(sub);

			for (int i = 0; i < indices.Length; i += 3)
			{

				p1 = indices[i];
				p2 = indices[i + 1];
				p3 = indices[i + 2];

				sides[0] = bladePlane.GetSide(targetMesh.vertices[p1]);
				sides[1] = bladePlane.GetSide(targetMesh.vertices[p2]);
				sides[2] = bladePlane.GetSide(targetMesh.vertices[p3]);


				// whole triangle
				if (sides[0] == sides[1] && sides[0] == sides[2])
				{

					if (sides[0])
					{ // left side

						leftSideMesher.AddTriangle(
							new Vector3[] { targetMesh.vertices[p1], targetMesh.vertices[p2], targetMesh.vertices[p3] },
							new Vector3[] { targetMesh.normals[p1], targetMesh.normals[p2], targetMesh.normals[p3] },
							new Vector2[] { targetMesh.uv[p1], targetMesh.uv[p2], targetMesh.uv[p3] },
							new Vector4[] { targetMesh.tangents[p1], targetMesh.tangents[p2], targetMesh.tangents[p3] },
							sub);
					}
					else
					{

						rightSideMesher.AddTriangle(
							new Vector3[] { targetMesh.vertices[p1], targetMesh.vertices[p2], targetMesh.vertices[p3] },
							new Vector3[] { targetMesh.normals[p1], targetMesh.normals[p2], targetMesh.normals[p3] },
							new Vector2[] { targetMesh.uv[p1], targetMesh.uv[p2], targetMesh.uv[p3] },
							new Vector4[] { targetMesh.tangents[p1], targetMesh.tangents[p2], targetMesh.tangents[p3] },
							sub);
					}

				}
				else
				{ // cut the triangle

					Cut_this_Face(
						new Vector3[] { targetMesh.vertices[p1], targetMesh.vertices[p2], targetMesh.vertices[p3] },
						new Vector3[] { targetMesh.normals[p1], targetMesh.normals[p2], targetMesh.normals[p3] },
						new Vector2[] { targetMesh.uv[p1], targetMesh.uv[p2], targetMesh.uv[p3] },
						new Vector4[] { targetMesh.tangents[p1], targetMesh.tangents[p2], targetMesh.tangents[p3] },
						sub);
				}
			}
		}

		// The capping Material will be at the end
		Material[] mats = target.GetComponent<MeshRenderer>().sharedMaterials;
		if(capMaterial != null) {
			Material[] newMats = new Material[mats.Length + 1];
			mats.CopyTo(newMats, 0);
			newMats[mats.Length] = capMaterial;
			mats = newMats;
		}
		_capMatSub = mats.Length - 1; // for later use

		// cap the opennings
		Capping();

		// Left Mesh
		Mesh left_HalfMesh = leftSideMesher.GetMesh();
		left_HalfMesh.name = "Split Mesh Left";

		// Right Mesh
		Mesh right_HalfMesh = rightSideMesher.GetMesh();
		right_HalfMesh.name = "Split Mesh Right";

		// assign the game objects
		GameObject leftSideObj = new GameObject("left side", typeof(MeshFilter), typeof(MeshRenderer));
		leftSideObj.transform.position = target.transform.position;
		leftSideObj.transform.rotation = target.transform.rotation;
		leftSideObj.transform.localScale = target.transform.localScale;
		leftSideObj.GetComponent<MeshFilter>().mesh = left_HalfMesh;

		GameObject rightSideObj = new GameObject("right side", typeof(MeshFilter), typeof(MeshRenderer));
		rightSideObj.transform.position = target.transform.position;
		rightSideObj.transform.rotation = target.transform.rotation;
		rightSideObj.transform.localScale = target.transform.localScale;
		rightSideObj.GetComponent<MeshFilter>().mesh = right_HalfMesh;

		// assign mats
		leftSideObj.GetComponent<MeshRenderer>().materials = mats;
		rightSideObj.GetComponent<MeshRenderer>().materials = mats;

		return new GameObject[] { leftSideObj, rightSideObj, target };
	}

	private static void Cut_this_Face(
		Vector3[] vertices,
		Vector3[] normals,
		Vector2[] uvs,
		Vector4[] tangents,
		int submesh)
	{

		bool[] sides = new bool[3];
		sides[0] = bladePlane.GetSide(vertices[0]); // true = left
		sides[1] = bladePlane.GetSide(vertices[1]);
		sides[2] = bladePlane.GetSide(vertices[2]);


		Vector3[] leftPoints = new Vector3[2];
		Vector3[] leftNormals = new Vector3[2];
		Vector2[] leftUvs = new Vector2[2];
		Vector4[] leftTangents = new Vector4[2];
		Vector3[] rightPoints = new Vector3[2];
		Vector3[] rightNormals = new Vector3[2];
		Vector2[] rightUvs = new Vector2[2];
		Vector4[] rightTangents = new Vector4[2];

		bool didset_left = false;
		bool didset_right = false;

		for (int i = 0; i < 3; i++)
		{

			if (sides[i])
			{
				if (!didset_left)
				{
					didset_left = true;

					leftPoints[0] = vertices[i];
					leftPoints[1] = leftPoints[0];
					leftUvs[0] = uvs[i];
					leftUvs[1] = leftUvs[0];
					leftNormals[0] = normals[i];
					leftNormals[1] = leftNormals[0];
					leftTangents[0] = tangents[i];
					leftTangents[1] = leftTangents[0];

				}
				else
				{

					leftPoints[1] = vertices[i];
					leftUvs[1] = uvs[i];
					leftNormals[1] = normals[i];
					leftTangents[1] = tangents[i];

				}
			}
			else
			{
				if (!didset_right)
				{
					didset_right = true;

					rightPoints[0] = vertices[i];
					rightPoints[1] = rightPoints[0];
					rightUvs[0] = uvs[i];
					rightUvs[1] = rightUvs[0];
					rightNormals[0] = normals[i];
					rightNormals[1] = rightNormals[0];
					rightTangents[0] = tangents[i];
					rightTangents[1] = rightTangents[0];

				}
				else
				{

					rightPoints[1] = vertices[i];
					rightUvs[1] = uvs[i];
					rightNormals[1] = normals[i];
					rightTangents[1] = tangents[i];

				}
			}
		}


		float normalizedDistance = 0.0f;
		float distance = 0;
		bladePlane.Raycast(new Ray(leftPoints[0], (rightPoints[0] - leftPoints[0]).normalized), out distance);

		normalizedDistance = distance / (rightPoints[0] - leftPoints[0]).magnitude;
		Vector3 newVertex1 = Vector3.Lerp(leftPoints[0], rightPoints[0], normalizedDistance);
		Vector2 newUv1 = Vector2.Lerp(leftUvs[0], rightUvs[0], normalizedDistance);
		Vector3 newNormal1 = Vector3.Lerp(leftNormals[0], rightNormals[0], normalizedDistance);
		Vector4 newTangent1 = Vector3.Lerp(leftTangents[0], rightTangents[0], normalizedDistance);

		verticesNew.Add(newVertex1);

		bladePlane.Raycast(new Ray(leftPoints[1], (rightPoints[1] - leftPoints[1]).normalized), out distance);

		normalizedDistance = distance / (rightPoints[1] - leftPoints[1]).magnitude;
		Vector3 newVertex2 = Vector3.Lerp(leftPoints[1], rightPoints[1], normalizedDistance);
		Vector2 newUv2 = Vector2.Lerp(leftUvs[1], rightUvs[1], normalizedDistance);
		Vector3 newNormal2 = Vector3.Lerp(leftNormals[1], rightNormals[1], normalizedDistance);
		Vector4 newTangent2 = Vector3.Lerp(leftTangents[1], rightTangents[1], normalizedDistance);


		verticesNew.Add(newVertex2);


		Vector3[] final_verts;
		Vector3[] final_norms;
		Vector2[] final_uvs;
		Vector4[] final_tangents;

		// first triangle

		final_verts = new Vector3[] { leftPoints[0], newVertex1, newVertex2 };
		final_norms = new Vector3[] { leftNormals[0], newNormal1, newNormal2 };
		final_uvs = new Vector2[] { leftUvs[0], newUv1, newUv2 };
		final_tangents = new Vector4[] { leftTangents[0], newTangent1, newTangent2 };

		if (final_verts[0] != final_verts[1] && final_verts[0] != final_verts[2])
		{

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			leftSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents, submesh);
		}

		// second triangle

		final_verts = new Vector3[] { leftPoints[0], leftPoints[1], newVertex2 };
		final_norms = new Vector3[] { leftNormals[0], leftNormals[1], newNormal2 };
		final_uvs = new Vector2[] { leftUvs[0], leftUvs[1], newUv2 };
		final_tangents = new Vector4[] { leftTangents[0], leftTangents[1], newTangent2 };

		if (final_verts[0] != final_verts[1] && final_verts[0] != final_verts[2])
		{

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			leftSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents, submesh);
		}

		// third triangle

		final_verts = new Vector3[] { rightPoints[0], newVertex1, newVertex2 };
		final_norms = new Vector3[] { rightNormals[0], newNormal1, newNormal2 };
		final_uvs = new Vector2[] { rightUvs[0], newUv1, newUv2 };
		final_tangents = new Vector4[] { rightTangents[0], newTangent1, newTangent2 };

		if (final_verts[0] != final_verts[1] && final_verts[0] != final_verts[2])
		{

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			rightSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents, submesh);
		}

		// fourth triangle

		final_verts = new Vector3[] { rightPoints[0], rightPoints[1], newVertex2 };
		final_norms = new Vector3[] { rightNormals[0], rightNormals[1], newNormal2 };
		final_uvs = new Vector2[] { rightUvs[0], rightUvs[1], newUv2 };
		final_tangents = new Vector4[] { rightTangents[0], rightTangents[1], newTangent2 };

		if (final_verts[0] != final_verts[1] && final_verts[0] != final_verts[2])
		{

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			rightSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents, submesh);
		}

	}

	private static void FlipFace(
		Vector3[] verts,
		Vector3[] norms,
		Vector2[] uvs,
		Vector4[] tangents)
	{

		Vector3 temp = verts[2];
		verts[2] = verts[0];
		verts[0] = temp;

		temp = norms[2];
		norms[2] = norms[0];
		norms[0] = temp;

		Vector2 temp2 = uvs[2];
		uvs[2] = uvs[0];
		uvs[0] = temp2;

		Vector4 temp3 = tangents[2];
		tangents[2] = tangents[0];
		tangents[0] = temp3;

	}

	private static List<Vector3> capVertTracker = new List<Vector3>();
	private static List<Vector3> capVertpolygon = new List<Vector3>();

	private static void Capping()
	{

		capVertTracker.Clear();

		for (int i = 0; i < verticesNew.Count-1; i++)
			if (!capVertTracker.Contains(verticesNew[i]))
			{
				capVertpolygon.Clear();
				capVertpolygon.Add(verticesNew[i]);
				capVertpolygon.Add(verticesNew[i + 1]);

				capVertTracker.Add(verticesNew[i]);
				capVertTracker.Add(verticesNew[i + 1]);


				bool isDone = false;
				while (!isDone)
				{
					isDone = true;

					for (int k = 0; k < verticesNew.Count; k += 2)
					{ // go through the pairs

						if (verticesNew[k] == capVertpolygon[capVertpolygon.Count - 1] && !capVertTracker.Contains(verticesNew[k + 1]))
						{ // if so add the other

							isDone = false;
							capVertpolygon.Add(verticesNew[k + 1]);
							capVertTracker.Add(verticesNew[k + 1]);

						}
						else if (verticesNew[k + 1] == capVertpolygon[capVertpolygon.Count - 1] && !capVertTracker.Contains(verticesNew[k]))
						{// if so add the other

							isDone = false;
							capVertpolygon.Add(verticesNew[k]);
							capVertTracker.Add(verticesNew[k]);
						}
					}
				}

				FillCap(capVertpolygon);

			}

	}

	private static void FillCap(List<Vector3> vertices)
	{


		// center of the cap
		Vector3 center = Vector3.zero;
		foreach (Vector3 point in vertices)
			center += point;

		center = center / vertices.Count;

		// you need an axis based on the cap
		Vector3 upward = Vector3.zero;
		// 90 degree turn
		upward.x = bladePlane.normal.y;
		upward.y = -bladePlane.normal.x;
		upward.z = bladePlane.normal.z;
		Vector3 left = Vector3.Cross(bladePlane.normal, upward);

		Vector3 displacement = Vector3.zero;
		Vector2 newUV1 = Vector2.zero;
		Vector2 newUV2 = Vector2.zero;

		for (int i = 0; i < vertices.Count; i++)
		{

			displacement = vertices[i] - center;
			newUV1 = Vector3.zero;
			newUV1.x = 0.5f + Vector3.Dot(displacement, left);
			newUV1.y = 0.5f + Vector3.Dot(displacement, upward);
			//newUV1.z = 0.5f + Vector3.Dot(displacement, _blade.normal);

			displacement = vertices[(i + 1) % vertices.Count] - center;
			newUV2 = Vector3.zero;
			newUV2.x = 0.5f + Vector3.Dot(displacement, left);
			newUV2.y = 0.5f + Vector3.Dot(displacement, upward);
			//newUV2.z = 0.5f + Vector3.Dot(displacement, _blade.normal);

			Vector3[] final_verts = new Vector3[] { vertices[i], vertices[(i + 1) % vertices.Count], center };
			Vector3[] final_norms = new Vector3[] { -bladePlane.normal, -bladePlane.normal, -bladePlane.normal };
			Vector2[] final_uvs = new Vector2[] { newUV1, newUV2, new Vector2(0.5f, 0.5f) };
			Vector4[] final_tangents = new Vector4[] { Vector4.zero, Vector4.zero, Vector4.zero };

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			leftSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents,
				_capMatSub);


			final_norms = new Vector3[] { bladePlane.normal, bladePlane.normal, bladePlane.normal };

			if (Vector3.Dot(Vector3.Cross(final_verts[1] - final_verts[0], final_verts[2] - final_verts[0]), final_norms[0]) < 0)
			{
				FlipFace(final_verts, final_norms, final_uvs, final_tangents);
			}

			rightSideMesher.AddTriangle(final_verts, final_norms, final_uvs, final_tangents,
				_capMatSub);


		}


	}


	private class Mesher
	{
		// Mesh Values
		private List<Vector3> _vertices = new List<Vector3>();
		private List<Vector3> _normals = new List<Vector3>();
		private List<Vector2> _uvs = new List<Vector2>();
		private List<Vector4> _tangents = new List<Vector4>();
		private List<List<int>> _subIndices = new List<List<int>>();


		public void AddTriangle(Vector3[] vertices, Vector3[] normals, Vector2[] uvs, int submesh)
		{
			int vertCount = _vertices.Count;

			_vertices.Add(vertices[0]);
			_vertices.Add(vertices[1]);
			_vertices.Add(vertices[2]);

			_normals.Add(normals[0]);
			_normals.Add(normals[1]);
			_normals.Add(normals[2]);

			_uvs.Add(uvs[0]);
			_uvs.Add(uvs[1]);
			_uvs.Add(uvs[2]);

			if (_subIndices.Count < submesh + 1)
			{
				for (int i = _subIndices.Count; i < submesh + 1; i++)
				{
					_subIndices.Add(new List<int>());
				}
			}

			_subIndices[submesh].Add(vertCount);
			_subIndices[submesh].Add(vertCount + 1);
			_subIndices[submesh].Add(vertCount + 2);

		}

		public void AddTriangle(
			Vector3[] vertices,
			Vector3[] normals,
			Vector2[] uvs,
			Vector4[] tangents,
			int submesh)
		{


			int vertCount = _vertices.Count;

			_vertices.Add(vertices[0]);
			_vertices.Add(vertices[1]);
			_vertices.Add(vertices[2]);

			_normals.Add(normals[0]);
			_normals.Add(normals[1]);
			_normals.Add(normals[2]);

			_uvs.Add(uvs[0]);
			_uvs.Add(uvs[1]);
			_uvs.Add(uvs[2]);

			_tangents.Add(tangents[0]);
			_tangents.Add(tangents[1]);
			_tangents.Add(tangents[2]);

			if (_subIndices.Count < submesh + 1)
			{
				for (int i = _subIndices.Count; i < submesh + 1; i++)
				{
					_subIndices.Add(new List<int>());
				}
			}

			_subIndices[submesh].Add(vertCount);
			_subIndices[submesh].Add(vertCount + 1);
			_subIndices[submesh].Add(vertCount + 2);

		}


		public void RemoveDoubles()
		{
			int dubCount = 0;

			Vector3 vertex = Vector3.zero;
			Vector3 normal = Vector3.zero;
			Vector2 uv = Vector2.zero;
			Vector4 tangent = Vector4.zero;

			int i = 0;
			while (i < _vertices.Count)
			{

				vertex = _vertices[i];
				normal = _normals[i];
				uv = _uvs[i];

				// look backward for a match
				for (int b = i - 1; b >= 0; b--)
				{

					if (vertex == _vertices[b] &&
						normal == _normals[b] &&
						uv == _uvs[b])
					{
						dubCount++;
						DoubleFound(b, i);
						i--;
						break; // there should only be one
					}
				}

				i++;

			}
		}

		private void DoubleFound(int first, int duplicate)
		{

			// go through all indices an replace them
			for (int h = 0; h < _subIndices.Count; h++)
			{
				for (int i = 0; i < _subIndices[h].Count; i++)
				{

					if (_subIndices[h][i] > duplicate) // knock it down
						_subIndices[h][i]--;
					else if (_subIndices[h][i] == duplicate) // replace
						_subIndices[h][i] = first;
				}
			}

			_vertices.RemoveAt(duplicate);
			_normals.RemoveAt(duplicate);
			_uvs.RemoveAt(duplicate);

			if (_tangents.Count > 0)
				_tangents.RemoveAt(duplicate);
		}

		/// <summary>
		/// Creates and returns a new mesh
		/// </summary>
		public Mesh GetMesh()
		{
			Mesh shape = new Mesh();
			shape.name = "Generated Mesh";
			shape.SetVertices(_vertices);
			shape.SetNormals(_normals);
			shape.SetUVs(0, _uvs);
			shape.SetUVs(1, _uvs);

			if (_tangents.Count > 1)
				shape.SetTangents(_tangents);

			shape.subMeshCount = _subIndices.Count;

			for (int i = 0; i < _subIndices.Count; i++)
				shape.SetTriangles(_subIndices[i], i);

			return shape;
		}
#if UNITY_EDITOR
		/// <summary>
		/// Creates and returns a new mesh with generated lightmap uvs
		/// </summary>
		public Mesh GetMesh_GenerateSecondaryUVSet()
		{
			Mesh mesh = GetMesh();

			UnityEditor.Unwrapping.GenerateSecondaryUVSet(mesh);

			return mesh;
		}
		
		/// <summary>
		/// Creates and returns a new mesh with generated lightmap uvs
		/// </summary>
		public Mesh GetMesh_GenerateSecondaryUVSet(UnityEditor.UnwrapParam param)
		{
			Mesh mesh = GetMesh();


			UnityEditor.Unwrapping.GenerateSecondaryUVSet(mesh, param);


			return mesh;
		}
#endif
	}

}