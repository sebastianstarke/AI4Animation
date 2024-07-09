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

/// <summary>
/// Represents a "space" in the Oculus Runtime.
/// </summary>
public readonly struct OVRSpace : IEquatable<OVRSpace>
{
    /// <summary>
    /// Represents a storage location for an <see cref="OVRSpace"/>.
    /// </summary>
    public enum StorageLocation
    {
        /// <summary>
        /// The storage location is local to the device.
        /// </summary>
        Local,

        /// <summary>
        /// The storage location is in the cloud.
        /// </summary>
        Cloud,
    }

    /// <summary>
    /// A runtime handle associated with this <see cref="OVRSpace"/>. This could change between subsequent sessions
    /// or apps.
    /// </summary>
    public ulong Handle { get; }

    /// <summary>
    /// Retrieve the universally unique identifier (UUID) associated with this <see cref="OVRSpace"/>.
    /// </summary>
    /// <remarks>
    /// Every space that can be persisted will have a UUID associated with it. UUIDs are consistent across different
    /// sessions and apps.
    ///
    /// The UUID of a space does not change over time, but not all spaces are guaranteed to have a UUID.
    /// </remarks>
    /// <param name="uuid">If successful, the uuid associated with this <see cref="OVRSpace"/>, otherwise, `Guid.Empty`.
    /// </param>
    /// <returns>Returns `true` if the uuid could be retrieved, otherwise `false`.</returns>
    public bool TryGetUuid(out Guid uuid) => OVRPlugin.GetSpaceUuid(Handle, out uuid);

    /// <summary>
    /// Indicates whether this <see cref="OVRSpace"/> represents a valid space (vs a default constructed
    /// <see cref="OVRSpace"/>).
    /// </summary>
    public bool Valid => Handle != 0;

    /// <summary>
    /// Constructs an <see cref="OVRSpace"/> object from an existing runtime handle and UUID.
    /// </summary>
    /// <remarks>
    /// This constructor does not create a new space. An <see cref="OVRSpace"/> is a wrapper for low-level functionality
    /// in the Oculus Runtime. To create a new spatial anchor, use <see cref="OVRSpatialAnchor"/>.
    /// </remarks>
    /// <param name="handle">The runtime handle associated with the space.</param>
    public OVRSpace(ulong handle) => Handle = handle;

    /// <summary>
    /// Generates a string representation of this <see cref="OVRSpace"/> of the form
    /// "0xYYYYYYYY"
    /// where "Y" are the hexadecimal digits of the <see cref="Handle"/>.
    /// </summary>
    /// <returns>Returns a string representation of this <see cref="OVRSpace"/>.</returns>
    public override string ToString() => $"0x{Handle:x16}";

    public bool Equals(OVRSpace other) => Handle == other.Handle;

    public override bool Equals(object obj) => obj is OVRSpace other && Equals(other);

    public override int GetHashCode() => Handle.GetHashCode();

    public static bool operator== (OVRSpace lhs, OVRSpace rhs) => lhs.Handle == rhs.Handle;

    public static bool operator!= (OVRSpace lhs, OVRSpace rhs) => lhs.Handle != rhs.Handle;

    public static implicit operator OVRSpace(ulong handle) => new OVRSpace(handle);

    public static implicit operator ulong(OVRSpace space) => space.Handle;
}

public static partial class OVRExtensions
{
    internal static OVRPlugin.SpaceStorageLocation ToSpaceStorageLocation(this OVRSpace.StorageLocation storageLocation)
    {
        switch (storageLocation)
        {
            case OVRSpace.StorageLocation.Local: return OVRPlugin.SpaceStorageLocation.Local;
            case OVRSpace.StorageLocation.Cloud: return OVRPlugin.SpaceStorageLocation.Cloud;
            default:
                throw new NotSupportedException($"{storageLocation} is not a supported {nameof(OVRPlugin.SpaceStorageLocation)}");
        }
    }
}
