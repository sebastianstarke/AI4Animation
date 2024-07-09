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

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using OVRSimpleJSON;
using Unity.Jobs;
using Unity.Collections;

public enum OVRGLTFType
{
	NONE,
	SCALAR,
	VEC2,
	VEC3,
	VEC4,
	MAT4,
}

public enum OVRGLTFComponentType
{
	BYTE = 5120,
	UNSIGNED_BYTE = 5121,
	SHORT = 5122,
	UNSIGNED_SHORT = 5123,
	UNSIGNED_INT = 5125,
	FLOAT = 5126,
}

public class OVRGLTFAccessor
{
	// Buffer View parameters
	private int byteOffset;
	private int byteLength;
	private int byteStride;
	private int bufferId;
	private int bufferLength;

	// Accessor parameters
	private int additionalOffset;
	private OVRGLTFType dataType;
	private OVRGLTFComponentType componentType;
	private int dataCount;

	public OVRGLTFAccessor(JSONNode node, JSONNode root, bool bufferViewOnly = false)
	{
		JSONNode jsonBufferView = node;
		if (!bufferViewOnly)
		{
			additionalOffset = node["byteOffset"].AsInt;
			dataType = ToOVRType(node["type"].Value);
			componentType = (OVRGLTFComponentType)node["componentType"].AsInt;
			dataCount = node["count"].AsInt;

			int bufferViewId = node["bufferView"].AsInt;
			jsonBufferView = root["bufferViews"][bufferViewId];
		}

		int bufferId = jsonBufferView["buffer"].AsInt;
		byteOffset = jsonBufferView["byteOffset"].AsInt;
		byteLength = jsonBufferView["byteLength"].AsInt;
		byteStride = jsonBufferView["byteStride"].AsInt;

		var jsonBuffer = root["buffers"][bufferId];
		bufferLength = jsonBuffer["byteLength"].AsInt;
	}

	public int GetDataCount()
	{
		return dataCount;
	}

	private static OVRGLTFType ToOVRType(string type)
	{
		switch(type)
		{
			case "SCALAR":
				return OVRGLTFType.SCALAR;
			case "VEC2":
				return OVRGLTFType.VEC2;
			case "VEC3":
				return OVRGLTFType.VEC3;
			case "VEC4":
				return OVRGLTFType.VEC4;
			case "MAT4":
				return OVRGLTFType.MAT4;
			default:
				Debug.LogError("Unsupported accessor type.");
				return OVRGLTFType.NONE;
		}
	}

	public void ReadAsInt(OVRBinaryChunk chunk, ref int[] data, int offset)
	{
		if (dataType != OVRGLTFType.SCALAR)
		{
			Debug.LogError("Tried to read non-scalar data as a uint array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int stride = byteStride > 0 ? byteStride : GetStrideForType(componentType);
		for(int i = 0; i < dataCount; i++)
		{
			data[offset + i] = (int)ReadElementAsUint(bufferData, i * stride, componentType);
		}
	}

	public void ReadAsFloat(OVRBinaryChunk chunk, ref float[] data, int offset)
	{
		if (dataType != OVRGLTFType.SCALAR)
		{
			Debug.LogError("Tried to read non-scalar data as a uint array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int stride = byteStride > 0 ? byteStride : GetStrideForType(componentType);
		for (int i = 0; i < dataCount; i++)
		{
			data[offset + i] = ReadElementAsFloat(bufferData, i * stride);
		}
	}

	public void ReadAsVector2(OVRBinaryChunk chunk, ref Vector2[] data, int offset)
	{
		if (dataType != OVRGLTFType.VEC2)
		{
			Debug.LogError("Tried to read non-vec3 data as a vec2 array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * 2;
		for (int i = 0; i < dataCount; i++)
		{
			if (componentType == OVRGLTFComponentType.FLOAT)
			{
				data[offset + i].x = ReadElementAsFloat(bufferData, i * stride);
				data[offset + i].y = ReadElementAsFloat(bufferData, i * stride + dataTypeSize);
			}
		}
	}

	public void ReadAsVector3(OVRBinaryChunk chunk, ref Vector3[] data, int offset, Vector3 conversionScale)
	{
		if (dataType != OVRGLTFType.VEC3)
		{
			Debug.LogError("Tried to read non-vec3 data as a vec3 array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * 3;
		for (int i = 0; i < dataCount; i++)
		{
			if (componentType == OVRGLTFComponentType.FLOAT)
			{
				data[offset + i].x = ReadElementAsFloat(bufferData, i * stride);
				data[offset + i].y = ReadElementAsFloat(bufferData, i * stride + dataTypeSize);
				data[offset + i].z = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 2);
			}
			else
			{
				data[offset + i].x = ReadElementAsUint(bufferData, i * stride, componentType);
				data[offset + i].y = ReadElementAsUint(bufferData, i * stride + dataTypeSize, componentType);
				data[offset + i].z = ReadElementAsUint(bufferData, i * stride + dataTypeSize * 2, componentType);
			}
			data[offset + i].Scale(conversionScale);
		}
	}

	public void ReadAsVector4(OVRBinaryChunk chunk, ref Vector4[] data, int offset, Vector4 conversionScale)
	{
		if (dataType != OVRGLTFType.VEC4)
		{
			Debug.LogError("Tried to read non-vec4 data as a vec4 array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * 4;
		for (int i = 0; i < dataCount; i++)
		{
			if (componentType == OVRGLTFComponentType.FLOAT)
			{
				data[offset + i].x = ReadElementAsFloat(bufferData, i * stride);
				data[offset + i].y = ReadElementAsFloat(bufferData, i * stride + dataTypeSize);
				data[offset + i].z = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 2);
				data[offset + i].w = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 3);
			}
			else
			{
				data[offset + i].x = ReadElementAsUint(bufferData, i * stride, componentType);
				data[offset + i].y = ReadElementAsUint(bufferData, i * stride + dataTypeSize, componentType);
				data[offset + i].z = ReadElementAsUint(bufferData, i * stride + dataTypeSize * 2, componentType);
				data[offset + i].w = ReadElementAsUint(bufferData, i * stride + dataTypeSize * 3, componentType);
			}
			data[offset + i].Scale(conversionScale);
		}
	}

	public void ReadAsColor(OVRBinaryChunk chunk, ref Color[] data, int offset)
	{
		if (dataType != OVRGLTFType.VEC4 && dataType != OVRGLTFType.VEC3)
		{
			Debug.LogError("Tried to read non-color type as a color array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int vecSize = dataType == OVRGLTFType.VEC3 ? 3 : 4;
		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * vecSize;
		float maxValue = GetMaxValueForType(componentType);
		for (int i = 0; i < dataCount; i++)
		{
			if (componentType == OVRGLTFComponentType.FLOAT)
			{
				data[offset + i].r = ReadElementAsFloat(bufferData, i * stride);
				data[offset + i].g = ReadElementAsFloat(bufferData, i * stride + dataTypeSize);
				data[offset + i].b = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 2);
				data[offset + i].a = dataType == OVRGLTFType.VEC3 ? 1.0f : ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 3);
			}
			else
			{
				data[offset + i].r = ReadElementAsUint(bufferData, i * stride, componentType) / maxValue;
				data[offset + i].g = ReadElementAsUint(bufferData, i * stride + dataTypeSize, componentType) / maxValue;
				data[offset + i].b = ReadElementAsUint(bufferData, i * stride + dataTypeSize * 2, componentType) / maxValue;
				data[offset + i].a = dataType == OVRGLTFType.VEC3 ? 1.0f : ReadElementAsUint(bufferData, i * stride + dataTypeSize * 3, componentType) / maxValue;
			}
		}
	}

	public void ReadAsMatrix4x4(OVRBinaryChunk chunk, ref Matrix4x4[] data, int offset, Vector3 conversionScale)
	{
		if (dataType != OVRGLTFType.MAT4)
		{
			Debug.LogError("Tried to read non-vec3 data as a vec3 array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * 16;

		Matrix4x4 scale = Matrix4x4.Scale(conversionScale);
		for (int i = 0; i < dataCount; i++)
		{
			for (int m = 0; m < 16; m++)
			{
				data[offset + i][m] = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * m);
			}
			data[offset + i] = scale * data[offset + i] * scale;
		}
	}

	public byte[] ReadAsKtxTexture(OVRBinaryChunk chunk)
	{
		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return null;
		}

		byte[] bufferData = new byte[byteLength];
		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		return bufferData;
	}

	public void ReadAsBoneWeights(OVRBinaryChunk chunk, ref Vector4[] data, int offset)
	{
		if (dataType != OVRGLTFType.VEC4)
		{
			Debug.LogError("Tried to read bone weights data as a non-vec4 array.");
			return;
		}

		if (chunk.chunkLength != bufferLength)
		{
			Debug.LogError("Chunk length is not equal to buffer length.");
			return;
		}

		byte[] bufferData = new byte[byteLength];

		chunk.chunkStream.Seek(chunk.chunkStart + byteOffset + additionalOffset, SeekOrigin.Begin);
		chunk.chunkStream.Read(bufferData, 0, byteLength);

		int dataTypeSize = GetStrideForType(componentType);
		int stride = byteStride > 0 ? byteStride : dataTypeSize * 4;
		for (int i = 0; i < dataCount; i++)
		{
			data[offset + i].x = ReadElementAsFloat(bufferData, i * stride);
			data[offset + i].y = ReadElementAsFloat(bufferData, i * stride + dataTypeSize);
			data[offset + i].z = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 2);
			data[offset + i].w = ReadElementAsFloat(bufferData, i * stride + dataTypeSize * 3);

			float weightSum = data[offset + i].x + data[offset + i].y + data[offset + i].z + data[offset + i].w;
			if (!Mathf.Approximately(weightSum, 0))
			{
				data[offset + i] /= weightSum;
			}
		}
	}

	private int GetStrideForType(OVRGLTFComponentType type)
	{
		switch (type)
		{
			case OVRGLTFComponentType.BYTE:
				return sizeof(sbyte);
			case OVRGLTFComponentType.UNSIGNED_BYTE:
				return sizeof(byte);
			case OVRGLTFComponentType.SHORT:
				return sizeof(short);
			case OVRGLTFComponentType.UNSIGNED_SHORT:
				return sizeof(ushort);
			case OVRGLTFComponentType.UNSIGNED_INT:
				return sizeof(uint);
			case OVRGLTFComponentType.FLOAT:
				return sizeof(float);
			default:
				return 0;
		}
	}

	private float GetMaxValueForType(OVRGLTFComponentType type)
	{
		switch (type)
		{
			case OVRGLTFComponentType.BYTE:
				return sbyte.MaxValue;
			case OVRGLTFComponentType.UNSIGNED_BYTE:
				return byte.MaxValue;
			case OVRGLTFComponentType.SHORT:
				return short.MaxValue;
			case OVRGLTFComponentType.UNSIGNED_SHORT:
				return ushort.MaxValue;
			case OVRGLTFComponentType.UNSIGNED_INT:
				return uint.MaxValue;
			case OVRGLTFComponentType.FLOAT:
				return float.MaxValue;
			default:
				return 0;
		}
	}

	private uint ReadElementAsUint(byte[] data, int index, OVRGLTFComponentType type)
	{
		switch(type)
		{
			case OVRGLTFComponentType.BYTE:
				return (uint)Convert.ToSByte(data[index]);
			case OVRGLTFComponentType.UNSIGNED_BYTE:
				return data[index];
			case OVRGLTFComponentType.SHORT:
				return (uint)BitConverter.ToInt16(data, index);
			case OVRGLTFComponentType.UNSIGNED_SHORT:
				return BitConverter.ToUInt16(data, index);
			case OVRGLTFComponentType.UNSIGNED_INT:
				return BitConverter.ToUInt32(data, index);
			default:
				Debug.Log(String.Format("Failed to read Component Type {0} as a uint.", type));
				return 0;
		}
	}

	private float ReadElementAsFloat(byte[] data, int index)
	{
		return BitConverter.ToSingle(data, index);
	}
}
