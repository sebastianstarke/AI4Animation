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

using System;
using System.Runtime.InteropServices;
using Unity.Collections.LowLevel.Unsafe;
using UnityEngine;

//-------------------------------------------------------------------------------------
/// <summary>
/// Collection of helper methods to facilitate data deserialization
/// </summary>
internal static class OVRDeserialize
{
    public static T ByteArrayToStructure<T>(byte[] bytes) where T: struct
    {
        T stuff;
        GCHandle handle = GCHandle.Alloc(bytes, GCHandleType.Pinned);
        try
        {
            stuff = (T)Marshal.PtrToStructure(handle.AddrOfPinnedObject(), typeof(T));
        }
        finally
        {
            handle.Free();
        }
        return stuff;
    }

    public struct DisplayRefreshRateChangedData
    {
        public float FromRefreshRate;
        public float ToRefreshRate;
    }

    public struct SpaceQueryResultsData
    {
        public UInt64 RequestId;
    }

    public struct SpaceQueryCompleteData
    {
        public UInt64 RequestId;
        public int Result;
    }

    public struct SceneCaptureCompleteData
    {
        public UInt64 RequestId;
        public int Result;
    }


    public struct SpatialAnchorCreateCompleteData
    {
        public UInt64 RequestId;
        public int Result;
        public UInt64 Space;
        public Guid Uuid;
    }

    public struct SpaceSetComponentStatusCompleteData
    {
        public UInt64 RequestId;
        public int Result;
        public UInt64 Space;
        public Guid Uuid;
        public OVRPlugin.SpaceComponentType ComponentType;
        public int Enabled;
    }

    public struct SpaceSaveCompleteData
    {
        public UInt64 RequestId;
        public UInt64 Space;
        public int Result;
        public Guid Uuid;
    }

    public struct SpaceEraseCompleteData
    {
        public UInt64 RequestId;
        public int Result;
        public Guid Uuid;
        public OVRPlugin.SpaceStorageLocation Location;
    }

    public struct SpaceShareResultData
    {
        public UInt64 RequestId;

        public int Result;
    }

    public struct SpaceListSaveResultData
    {
        public UInt64 RequestId;

        public int Result;
    }

}
