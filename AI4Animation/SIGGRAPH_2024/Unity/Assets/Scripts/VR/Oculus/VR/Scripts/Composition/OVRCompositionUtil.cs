/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * Licensed under the Oculus SDK License Agreement (the "License");
 * you may not use the Oculus SDK except in compliance with the License,
 * which is provided at the time of installation or download, or which
 * otherwise accompanies this software in either electronic or hard copy form.
 *
 * You may obtain a copy of the License at
 *
 * https://developer.oculus.com/licenses/oculussdk/
 *
 * Unless required by applicable law or agreed to in writing, the Oculus SDK
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using UnityEngine;
using System.Collections.Generic;

#if UNITY_EDITOR_WIN || UNITY_STANDALONE_WIN || UNITY_ANDROID

internal class OVRCompositionUtil {

	public static void SafeDestroy(GameObject obj)
	{
		if (Application.isPlaying)
		{
			GameObject.Destroy(obj);
		}
		else
		{
			GameObject.DestroyImmediate(obj);
		}
	}

	public static void SafeDestroy(ref GameObject obj)
	{
		if (obj != null)
		{
			SafeDestroy(obj);
			obj = null;
		}
	}

	[System.Obsolete("GetWorldPosition should be invoked with an explicit camera parameter")]
	public static Vector3 GetWorldPosition(Vector3 trackingSpacePosition)
	{
		return GetWorldPosition(Camera.main, trackingSpacePosition);
	}

	public static Vector3 GetWorldPosition(Camera camera, Vector3 trackingSpacePosition)
	{
		OVRPose tsPose;
		tsPose.position = trackingSpacePosition;
		tsPose.orientation = Quaternion.identity;
		OVRPose wsPose = OVRExtensions.ToWorldSpacePose(tsPose, camera);
		Vector3 pos = wsPose.position;
		return pos;
	}

	public static float GetMaximumBoundaryDistance(Camera camera, OVRBoundary.BoundaryType boundaryType)
	{
		if (!OVRManager.boundary.GetConfigured())
		{
			return float.MaxValue;
		}

		Vector3[] geometry = OVRManager.boundary.GetGeometry(boundaryType);
		if (geometry.Length == 0)
		{
			return float.MaxValue;
		}

		float maxDistance = -float.MaxValue;
		foreach (Vector3 v in geometry)
		{
			Vector3 pos = GetWorldPosition(camera, v);
			float distance = Vector3.Dot(camera.transform.forward, pos);
			if (maxDistance < distance)
			{
				maxDistance = distance;
			}
		}
		return maxDistance;
	}

	public static Mesh BuildBoundaryMesh(OVRBoundary.BoundaryType boundaryType, float topY, float bottomY)
	{
		if (!OVRManager.boundary.GetConfigured())
		{
			return null;
		}

		List<Vector3> geometry = new List<Vector3>(OVRManager.boundary.GetGeometry(boundaryType));
		if (geometry.Count == 0)
		{
			return null;
		}

		geometry.Add(geometry[0]);
		int numPoints = geometry.Count;

		Vector3[] vertices = new Vector3[numPoints * 2];
		Vector2[] uvs = new Vector2[numPoints * 2];
		for (int i = 0; i < numPoints; ++i)
		{
			Vector3 v = geometry[i];
			vertices[i] = new Vector3(v.x, bottomY, v.z);
			vertices[i + numPoints] = new Vector3(v.x, topY, v.z);
			uvs[i] = new Vector2((float)i / (numPoints - 1), 0.0f);
			uvs[i + numPoints] = new Vector2(uvs[i].x, 1.0f);
		}

		int[] triangles = new int[(numPoints - 1) * 2 * 3];
		for (int i = 0; i < numPoints - 1; ++i)
		{
			// the geometry is built clockwised. only the back faces should be rendered in the camera frame mask

			triangles[i * 6 + 0] = i;
			triangles[i * 6 + 1] = i + numPoints;
			triangles[i * 6 + 2] = i + 1 + numPoints;

			triangles[i * 6 + 3] = i;
			triangles[i * 6 + 4] = i + 1 + numPoints;
			triangles[i * 6 + 5] = i + 1;
		}

		Mesh mesh = new Mesh();
		mesh.vertices = vertices;
		mesh.uv = uvs;
		mesh.triangles = triangles;
		return mesh;
	}

}

#endif
