using System.Collections;
using System.Collections.Generic;

using UnityEngine;
using UnityEngine.Rendering;

namespace VoxelSystem
{

    public class VoxelMesh {

		public static Mesh Build(Voxel_t[] voxels, float unit, bool useUV = false) {
			var vertices = new List<Vector3>();
			var uvs = new List<Vector2>();
			var triangles = new List<int>();
			var normals = new List<Vector3>();
			var centers = new List<Vector4>();

			var up = Vector3.up * unit;
			var hup = up * 0.5f;
			var hbottom = -hup;

			var right = Vector3.right * unit;
			var hright = right * 0.5f;

			var left = -right;
			var hleft = left * 0.5f;

			var forward = Vector3.forward * unit;
			var hforward = forward * 0.5f;
			var back = -forward;
			var hback = back * 0.5f;

			for(int i = 0, n = voxels.Length; i < n; i++) {
				var v = voxels[i];
				if(v.fill > 0) {
					// back
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hback, right, up, Vector3.back
					);

					// right
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hright, forward, up, Vector3.right
					);

					// forward
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hforward, left, up, Vector3.forward
					);

					// left
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hleft, back, up, Vector3.left
					);

					// up
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hup, right, forward, Vector3.up
					);

					// down
					CalculatePlane(
						vertices, normals, centers, uvs, triangles,
						v, useUV, hbottom, right, back, Vector3.down
					);

				}
			}

			var mesh = new Mesh();
			mesh.indexFormat = IndexFormat.UInt32;
			mesh.vertices = vertices.ToArray();
			mesh.uv = uvs.ToArray();
			mesh.normals = normals.ToArray();
			mesh.tangents = centers.ToArray();
			mesh.SetTriangles(triangles.ToArray(), 0);
			mesh.RecalculateBounds();
			return mesh;
		}

		static void CalculatePlane (
			List<Vector3> vertices, List<Vector3> normals, List<Vector4> centers, List<Vector2> uvs, List<int> triangles,
			Voxel_t voxel, bool useUV, Vector3 offset, Vector3 right, Vector3 up, Vector3 normal, int rSegments = 2, int uSegments = 2
		) {
			float rInv = 1f / (rSegments - 1);
			float uInv = 1f / (uSegments - 1);

			int triangleOffset = vertices.Count;
            var center = voxel.position;

			var transformed = center + offset;
			for(int y = 0; y < uSegments; y++) {
				float ru = y * uInv;
				for(int x = 0; x < rSegments; x++) {
					float rr = x * rInv;
					vertices.Add(transformed + right * (rr - 0.5f) + up * (ru - 0.5f));
					normals.Add(normal);
					centers.Add(center);
                    if(useUV)
                    {
					    uvs.Add(voxel.uv);
                    } else
                    {
					    uvs.Add(new Vector2(rr, ru));
                    }
				}

				if(y < uSegments - 1) {
					var ioffset = y * rSegments + triangleOffset;
					for(int x = 0, n = rSegments - 1; x < n; x++) {
						triangles.Add(ioffset + x);
						triangles.Add(ioffset + x + rSegments);
						triangles.Add(ioffset + x + 1);

						triangles.Add(ioffset + x + 1);
						triangles.Add(ioffset + x + rSegments);
						triangles.Add(ioffset + x + 1 + rSegments);
					}
				}
			}
		}

    }

}


