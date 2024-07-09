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
using System.Collections.Generic;

/// <summary>
/// Utility to assist with queries for <see cref="OVRSpace"/>s.
/// </summary>
internal static class OVRSpaceQuery
{
    /// <summary>
    /// Components that can be enabled on an <see cref="OVRSpace"/>.
    /// </summary>
    [Flags]
    public enum ComponentType : uint
    {
        /// <summary>
        /// No components.
        /// </summary>
        None = 0,

        /// <summary>
        /// The space is locatable.
        /// </summary>
        Locatable = 1 << OVRPlugin.SpaceComponentType.Locatable,

        /// <summary>
        /// The space is storable.
        /// </summary>
        Storable = 1 << OVRPlugin.SpaceComponentType.Storable,

        /// <summary>
        /// The space is sharable.
        /// </summary>
        Sharable = 1 << OVRPlugin.SpaceComponentType.Sharable,

        /// <summary>
        /// The space represents a 2D plane.
        /// </summary>
        Bounded2D = 1 << OVRPlugin.SpaceComponentType.Bounded2D,

        /// <summary>
        /// The space represents a 3D volume.
        /// </summary>
        Bounded3D = 1 << OVRPlugin.SpaceComponentType.Bounded3D,

        /// <summary>
        /// The space has semantic labels associated with it.
        /// </summary>
        SemanticLabels = 1 << OVRPlugin.SpaceComponentType.SemanticLabels,

        /// <summary>
        /// The space represents a room layout.
        /// </summary>
        RoomLayout = 1 << OVRPlugin.SpaceComponentType.RoomLayout,

        /// <summary>
        /// The space is a container for other spaces.
        /// </summary>
        SpaceContainer = 1 << OVRPlugin.SpaceComponentType.SpaceContainer,
    }

    /// <summary>
    /// Represents options used to generate an <see cref="OVRSpaceQuery"/>.
    /// </summary>
    public struct Options
    {
        /// <summary>
        /// The maximum number of UUIDs which can be used in a <see cref="UuidFilter"/>.
        /// </summary>
        public const int MaxUuidCount = OVRPlugin.SpaceFilterInfoIdsMaxSize;

        private static readonly Guid[] Ids = new Guid[MaxUuidCount];

        private static readonly OVRPlugin.SpaceComponentType[] ComponentTypes =
            new OVRPlugin.SpaceComponentType[OVRPlugin.SpaceFilterInfoComponentsMaxSize];

        /// <summary>
        /// The maximum number of results the query can return.
        /// </summary>
        public int MaxResults { get; set; }

        /// <summary>
        /// The timeout, in seconds for the query.
        /// </summary>
        /// <remarks>
        /// Zero indicates the query does not timeout.
        /// </remarks>
        public double Timeout { get; set; }

        /// <summary>
        /// The storage location to query.
        /// </summary>
        public OVRSpace.StorageLocation Location { get; set; }

        /// <summary>
        /// The type of query to perform.
        /// </summary>
        public OVRPlugin.SpaceQueryType QueryType { get; set; }

        /// <summary>
        /// The type of action to perform.
        /// </summary>
        public OVRPlugin.SpaceQueryActionType ActionType { get; set; }

        private ComponentType _componentFilter;

        private IReadOnlyList<Guid> _uuidFilter;

        /// <summary>
        /// The components which must be present on the space in order to match the query.
        /// </summary>
        /// <remarks>
        /// The query will be limited to spaces that have this set of components. You may filter by component type or
        /// UUID (see <see cref="UuidFilter"/>), but not both at the same time.
        ///
        /// Currently, only one component is allowed at a time.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if <see cref="UuidFilter"/> is not `null`.</exception>
        /// <exception cref="NotSupportedException">Thrown if more than one <see cref="ComponentType"/> is set.</exception>
        public ComponentType ComponentFilter
        {
            get => _componentFilter;
            set
            {
                if (value != 0 && _uuidFilter != null)
                    throw new InvalidOperationException($"Cannot have both a component and uuid filter.");

                // Count the number of set bits
                var v = (uint)value;
                var numBitsSet = 0;
                while (v != 0)
                {
                    v &= v - 1;
                    numBitsSet++;
                }

                if (numBitsSet > 1)
                    throw new NotSupportedException($"Only one component is supported, but {numBitsSet} are set.");

                _componentFilter = value;
            }
        }

        /// <summary>
        /// A set of UUIDs used to filter the query.
        /// </summary>
        /// <remarks>
        /// The query will look for this set of UUIDs and only return matching UUIDs up to <see cref="MaxResults"/>.
        /// You may filter by component type (see <see cref="ComponentFilter"/>) or UUIDs, but not both at the same
        /// time.
        /// </remarks>
        /// <exception cref="InvalidOperationException">Thrown if <see cref="ComponentFilter"/> is not
        /// <see cref="ComponentType.None"/>.</exception>
        /// <exception cref="ArgumentException">Thrown if <see cref="UuidFilter"/> is set to a value that contains more
        /// than <seealso cref="MaxUuidCount"/> UUIDs.</exception>
        public IReadOnlyList<Guid> UuidFilter
        {
            get => _uuidFilter;
            set
            {
                if (value != null && _componentFilter != 0)
                    throw new InvalidOperationException($"{nameof(ComponentFilter)} must be {nameof(ComponentType.None)} to query by UUID.");

                if (value?.Count > MaxUuidCount)
                    throw new ArgumentException($"There must not be more than {MaxUuidCount} UUIDs specified by the {nameof(UuidFilter)} (new value contains {value.Count} UUIDs).", nameof(value));

                _uuidFilter = value;
            }
        }

        /// <summary>
        /// Creates a copy of <paramref name="other"/>.
        /// </summary>
        /// <param name="other">The options to copy.</param>
        public Options(Options other)
        {
            QueryType = other.QueryType;
            MaxResults = other.MaxResults;
            Timeout = other.Timeout;
            Location = other.Location;
            ActionType = other.ActionType;
            _componentFilter = other._componentFilter;
            _uuidFilter = other._uuidFilter;
        }

        /// <summary>
        /// Initiates a space query.
        /// </summary>
        /// <param name="requestId">When this method returns, <paramref name="requestId"/> will represent a valid
        /// request if successful, or an invalid request if not. This parameter is passed initialized.</param>
        /// <returns>`true` if the query was successfully started; otherwise, `false`.</returns>
        public bool TryQuerySpaces(out ulong requestId)
        {
            var filterType = OVRPlugin.SpaceQueryFilterType.None;
            var numIds = 0;
            var numComponents = 0;
            if (_uuidFilter != null)
            {
                filterType = OVRPlugin.SpaceQueryFilterType.Ids;
                numIds = Math.Min(_uuidFilter.Count, MaxUuidCount);
                for (var i = 0; i < numIds; i++)
                {
                    Ids[i] = _uuidFilter[i];
                }
            }
            else if (_componentFilter != 0)
            {
                filterType = OVRPlugin.SpaceQueryFilterType.Components;
                if ((_componentFilter & ComponentType.Locatable) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.Locatable;
                if ((_componentFilter & ComponentType.Storable) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.Storable;
                if ((_componentFilter & ComponentType.Sharable) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.Sharable;
                if ((_componentFilter & ComponentType.Bounded2D) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.Bounded2D;
                if ((_componentFilter & ComponentType.Bounded3D) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.Bounded3D;
                if ((_componentFilter & ComponentType.SemanticLabels) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.SemanticLabels;
                if ((_componentFilter & ComponentType.RoomLayout) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.RoomLayout;
                if ((_componentFilter & ComponentType.SpaceContainer) != 0)
                    ComponentTypes[numComponents++] = OVRPlugin.SpaceComponentType.SpaceContainer;
            }

            var queryInfo = new OVRPlugin.SpaceQueryInfo
            {
                QueryType = QueryType,
                MaxQuerySpaces = MaxResults,
                Timeout = Timeout,
                Location = Location.ToSpaceStorageLocation(),
                ActionType = ActionType,
                FilterType = filterType,
                IdInfo = new OVRPlugin.SpaceFilterInfoIds
                {
                    Ids = Ids,
                    NumIds = numIds
                },
                ComponentsInfo = new OVRPlugin.SpaceFilterInfoComponents
                {
                    Components = ComponentTypes,
                    NumComponents = numComponents,
                }
            };

            return OVRPlugin.QuerySpaces(queryInfo, out requestId);
        }
    }
}
