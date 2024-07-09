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
using UnityEngine;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine.Serialization;
using Debug = UnityEngine.Debug;

/// <summary>
/// A manager for <see cref="OVRSceneAnchor"/>s created using the Guardian's Room Capture feature.
/// </summary>
public class OVRSceneManager : MonoBehaviour
{
    /// <summary>
    /// A prefab that will be used to instantiate any Plane found when querying the Scene model
    /// </summary>
    [FormerlySerializedAs("planePrefab")]
    [Tooltip("A prefab that will be used to instantiate any Plane found when querying the Scene model")]
    public OVRSceneAnchor PlanePrefab;

    /// <summary>
    /// A prefab that will be used to instantiate any Volume found when querying the Scene model
    /// </summary>
    [FormerlySerializedAs("volumePrefab")]
    [Tooltip("A prefab that will be used to instantiate any Volume found when querying the Scene model")]
    public OVRSceneAnchor VolumePrefab;

    /// <summary>
    /// Overrides the instantiation of the generic Plane and Volume prefabs with specialized ones
    /// </summary>
    [FormerlySerializedAs("prefabOverrides")]
    [Tooltip("Overrides the instantiation of the generic Plane/Volume prefabs with specialized ones")]
    public List<OVRScenePrefabOverride> PrefabOverrides = new List<OVRScenePrefabOverride>();

    /// <summary>
    /// When true, verbose debug logs will be emitted.
    /// </summary>
    [FormerlySerializedAs("verboseLogging")]
    [Tooltip("When enabled, verbose debug logs will be emitted.")]
    public bool VerboseLogging;

    /// <summary>
    /// The maximum number of scene anchors that will be updated each frame.
    /// </summary>
    [Tooltip("The maximum number of scene anchors that will be updated each frame.")]
    public int MaxSceneAnchorUpdatesPerFrame = 3;


    #region Events

    /// <summary>
    /// This event fires when the OVR Scene Manager has correctly loaded the scene definition and
    /// instantiated the prefabs for the planes and volumes. Trap it to know that the logic of the
    /// experience can now continue.
    /// </summary>
    public Action SceneModelLoadedSuccessfully;

    /// <summary>
    ///This event fires when a query load the Scene Model returns no result. It can indicate that the,
    /// user never used the Room Capture in the space they are in.
    /// </summary>
    public Action NoSceneModelToLoad;

    /// <summary>
    /// This event will fire after the Room Capture successfully returns. It can be trapped to load the
    /// scene Model.
    /// </summary>
    public Action SceneCaptureReturnedWithoutError;

    /// <summary>
    /// This event will fire if an error occurred while trying to send the user to Room Capture.
    /// </summary>
    public Action UnexpectedErrorWithSceneCapture;


    #endregion

    /// <summary>
    /// Represents the available classifications for each <see cref="OVRSceneAnchor"/>.
    /// </summary>
    public static class Classification
    {
        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a floor.
        /// </summary>
        public const string Floor = "FLOOR";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a ceiling.
        /// </summary>
        public const string Ceiling = "CEILING";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a wall face.
        /// </summary>
        public const string WallFace = "WALL_FACE";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a desk.
        /// </summary>
        public const string Desk = "DESK";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a couch.
        /// </summary>
        public const string Couch = "COUCH";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a door frame.
        /// </summary>
        public const string DoorFrame = "DOOR_FRAME";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as a window frame.
        /// </summary>
        public const string WindowFrame = "WINDOW_FRAME";

        /// <summary>
        /// Represents an <see cref="OVRSceneAnchor"/> that is classified as other.
        /// </summary>
        public const string Other = "OTHER";

        /// <summary>
        /// The list of possible semantic labels.
        /// </summary>
        public static IReadOnlyList<string> List { get; } = new[]
        {
            Floor,
            Ceiling,
            WallFace,
            Desk,
            Couch,
            DoorFrame,
            WindowFrame,
            Other
        };
    }

    /// <summary>
    /// A container for the set of <see cref="OVRSceneAnchor"/>s representing a room.
    /// </summary>
    public class RoomLayoutInformation
    {
        /// <summary>
        /// The <see cref="OVRScenePlane"/> representing the floor of the room.
        /// </summary>
        public OVRScenePlane Floor;

        /// <summary>
        /// The <see cref="OVRScenePlane"/> representing the ceiling of the room.
        /// </summary>
        public OVRScenePlane Ceiling;

        /// <summary>
        /// The set of <see cref="OVRScenePlane"/> representing the walls of the room.
        /// </summary>
        public List<OVRScenePlane> Walls = new List<OVRScenePlane>();
    }

    /// <summary>
    /// Describes the room layout stored by the Guardian.
    /// </summary>
    public RoomLayoutInformation RoomLayout;

    #region Private Vars

    private enum QueryMode
    {
        QueryAllAnchors,                                // Get entire Scene
        QueryByUuid,                                    // Get a specific entity
        QueryAllBounded2DEnabled,                       // Get all planar entities
        QueryAllRoomLayoutEnabledForAllEntitiesInside,  // Get Ceiling/Floor/Walls + other entities in Space Container.
        QueryAllRoomLayoutEnabledForRoomBox,            // Get Ceiling/Floor/Walls only.
    }


    private readonly Dictionary<Guid, int> _orderedRoomGuids = new Dictionary<Guid, int>();

    private Comparison<OVRScenePlane> _wallOrderComparer;

    // Maintain UUIDs to be used.
    private List<Guid> _uuidsToQuery;

    private QueryMode _currentQueryMode = QueryMode.QueryAllAnchors;

    // We use this to store the request id when attempting to load the scene
    private UInt64 _sceneCaptureRequestId = UInt64.MaxValue;

    private HashSet<UInt64> _individualRequestIds = new HashSet<UInt64>();

    // Spatial entities that we know about but are waiting for their "locatable" component to be enabled.
    private readonly Dictionary<OVRSpace, OVRPlugin.SpaceQueryResult> _pendingLocatable =
        new Dictionary<OVRSpace, OVRPlugin.SpaceQueryResult>();

    private OVRCameraRig _cameraRig;
    private int _sceneAnchorUpdateIndex;
    private HashSet<Guid> _uuidSet = new HashSet<Guid>();
    private bool _sceneLoadComplete;


    #endregion

    internal struct LogForwarder
    {
        public void Log(string context, string message) => Debug.Log($"[{context}] {message}");
        public void LogWarning(string context, string message) => Debug.LogWarning($"[{context}] {message}");
        public void LogError(string context, string message) => Debug.LogError($"[{context}] {message}");
    }

    internal LogForwarder? Verbose => VerboseLogging ? new LogForwarder() : (LogForwarder?)null;

    internal static class Development
    {
        [Conditional("DEVELOPMENT_BUILD")]
        [Conditional("UNITY_EDITOR")]
        public static void Log(string context, string message) => Debug.Log($"[{context}] {message}");

        [Conditional("DEVELOPMENT_BUILD")]
        [Conditional("UNITY_EDITOR")]
        public static void LogWarning(string context, string message) => Debug.LogWarning($"[{context}] {message}");

        [Conditional("DEVELOPMENT_BUILD")]
        [Conditional("UNITY_EDITOR")]
        public static void LogError(string context, string message) => Debug.LogError($"[{context}] {message}");
    }

    void Awake()
    {
        _wallOrderComparer = (planeA, planeB) =>
        {
            bool TryGetUuid(OVRScenePlane plane, out int index)
            {
                var guid = plane.GetComponent<OVRSceneAnchor>().Uuid;
                if (_orderedRoomGuids.TryGetValue(guid, out index)) return true;

                Development.LogWarning(nameof(OVRSceneManager),
                    $"{nameof(OVRScenePlane)} {guid} does not belong to the current room layout.");
                return false;
            }

            if (!TryGetUuid(planeA, out var indexA)) return 0;
            if (!TryGetUuid(planeB, out var indexB)) return 0;

            return indexA.CompareTo(indexB);
        };

        // Only allow one instance at runtime.
        if (FindObjectsOfType<OVRSceneManager>().Length > 1)
        {
            new LogForwarder().LogError(nameof(OVRSceneManager),
                $"Found multiple {nameof(OVRSceneManager)}s. Destroying '{name}'.");
            enabled = false;
            DestroyImmediate(this);
        }
    }

    private static void OnTrackingSpaceChanged(Transform trackingSpace)
    {
        // Tracking space changed, update all scene anchors using their cache
        UpdateAllSceneAnchors();
    }

    private void Update()
    {
        UpdateSomeSceneAnchors();

    }


    private static void UpdateAllSceneAnchors()
    {
        foreach (var sceneAnchor in OVRSceneAnchor.SceneAnchors.Values)
        {
            sceneAnchor.TryUpdateTransform(true);
        }
    }

    private void UpdateSomeSceneAnchors()
    {
        for (var i = 0; i < Math.Min(OVRSceneAnchor.SceneAnchorsList.Count, MaxSceneAnchorUpdatesPerFrame); i++)
        {
            _sceneAnchorUpdateIndex %= OVRSceneAnchor.SceneAnchorsList.Count;
            var anchor = OVRSceneAnchor.SceneAnchorsList[_sceneAnchorUpdateIndex++];
            anchor.TryUpdateTransform(false);
        }
    }

    /// <summary>
    /// Loads the scene model from the Guardian.
    /// </summary>
    /// <remarks>
    /// When running on Quest, Scene is queried to retrieve the entities describing the Scene Model. In the Editor,
    /// the Scene Model is loaded over Link.
    /// </remarks>
    /// <returns>Returns true if the query was successfully registered</returns>
    public bool LoadSceneModel()
    {
        _currentQueryMode = QueryMode.QueryAllRoomLayoutEnabledForAllEntitiesInside;
        return LoadSpatialEntities();
    }

    /// <summary>
    /// Requests scene capture from the Guardian.
    /// </summary>
    /// <returns>Returns true if scene capture succeeded, otherwise false.</returns>
    public bool RequestSceneCapture() => RequestSceneCapture("");

    /// <summary>
    /// Requests scene capture with specified types of <see cref="OVRSceneAnchor"/> from the Guardian.
    /// </summary>
    /// <param name="requestedAnchorClassifications">A list of <see cref="OVRSceneManager.Classification"/>.</param>
    /// <returns>Returns true if scene capture succeeded, otherwise false.</returns>
    public bool RequestSceneCapture(IEnumerable<string> requestedAnchorClassifications)
    {
        if(requestedAnchorClassifications == null)
        {
            throw new ArgumentNullException(nameof(requestedAnchorClassifications));
        }

        var anchorClassifications = requestedAnchorClassifications.ToList();
        foreach (var classification in anchorClassifications)
        {
            if (!Classification.List.Contains(classification))
            {
                throw new ArgumentException(
                    $"{nameof(requestedAnchorClassifications)} contains invalid anchor {nameof(Classification)} {nameof(classification)}.");
            }
        }

        return RequestSceneCapture(String.Join(",", anchorClassifications));
    }

    #region Private Methods

    private bool RequestSceneCapture(string requestString)
    {
#if !UNITY_EDITOR
        return OVRPlugin.RequestSceneCapture(requestString, out _sceneCaptureRequestId);
#elif UNITY_EDITOR_WIN
        Development.LogWarning(nameof(OVRSceneManager),
            "Scene Capture does not work over Link. Please capture a scene with the HMD in standalone mode, then access the scene model over Link.");
        return false;
#else
        return false;
#endif
    }

    private void OnEnable()
    {
        // Bind events
        OVRManager.SpaceQueryComplete += OVRManager_SpaceQueryComplete;
        OVRManager.SceneCaptureComplete += OVRManager_SceneCaptureComplete;
        OVRManager.SpaceSetComponentStatusComplete += OVRManager_SpaceSetComponentStatusComplete;

        if (OVRManager.display != null)
        {
            OVRManager.display.RecenteredPose += UpdateAllSceneAnchors;
        }

        if (!_cameraRig)
        {
            _cameraRig = FindObjectOfType<OVRCameraRig>();
        }

        if (_cameraRig)
        {
            _cameraRig.TrackingSpaceChanged += OnTrackingSpaceChanged;
        }
    }

    private void OnDisable()
    {
        // Unbind events
        OVRManager.SpaceQueryComplete -= OVRManager_SpaceQueryComplete;
        OVRManager.SceneCaptureComplete -= OVRManager_SceneCaptureComplete;
        OVRManager.SpaceSetComponentStatusComplete -= OVRManager_SpaceSetComponentStatusComplete;

        if (OVRManager.display != null)
        {
            OVRManager.display.RecenteredPose -= UpdateAllSceneAnchors;
        }

        if (_cameraRig)
        {
            _cameraRig.TrackingSpaceChanged -= OnTrackingSpaceChanged;
        }
    }

    private bool LoadSpatialEntities()
    {
        // Remove all the scene entities in memory. Update with scene entities from new query.
        var sceneAnchors = FindObjectsOfType<OVRSceneAnchor>();
        foreach (var sceneAnchor in sceneAnchors)
        {
            Destroy(sceneAnchor.gameObject);
        }

        RoomLayout = new RoomLayoutInformation();
        var options = new OVRSpaceQuery.Options
        {
            QueryType = OVRPlugin.SpaceQueryType.Action,
            MaxResults = 100,
            Timeout = 0,
            Location = OVRSpace.StorageLocation.Local,
            ActionType = OVRPlugin.SpaceQueryActionType.Load,
        };

        if (_currentQueryMode == QueryMode.QueryByUuid)
        {
            options.UuidFilter = _uuidsToQuery;
        }
        else if (_currentQueryMode == QueryMode.QueryAllRoomLayoutEnabledForAllEntitiesInside ||
                 _currentQueryMode == QueryMode.QueryAllBounded2DEnabled ||
                 _currentQueryMode == QueryMode.QueryAllRoomLayoutEnabledForRoomBox)
        {
            if (_currentQueryMode == QueryMode.QueryAllRoomLayoutEnabledForAllEntitiesInside ||
                _currentQueryMode == QueryMode.QueryAllRoomLayoutEnabledForRoomBox)
            {
                options.MaxResults = 1;
                options.ComponentFilter = OVRSpaceQuery.ComponentType.RoomLayout;
            }
            else
            {
                options.ComponentFilter = OVRSpaceQuery.ComponentType.Bounded2D;
            }
        }

        if (options.TryQuerySpaces(out var requestId))
        {
            // We save this request id to ensure that when we trap a SpaceQueryResults event
            // it's indeed one of our requests.
            _individualRequestIds.Add(requestId);
            Verbose?.Log(nameof(OVRSceneManager),
                $"{nameof(LoadSpatialEntities)}() calling {nameof(OVRSpaceQuery)}.{nameof(OVRSpaceQuery.Options)}.{nameof(options.TryQuerySpaces)}(). Request id [{requestId}] added to the request list.");

            return true;
        }

        Verbose?.LogWarning(nameof(OVRSceneManager),
            $"{nameof(LoadSpatialEntities)}() {nameof(OVRSpaceQuery)}.{nameof(OVRSpaceQuery.Options)}.{nameof(options.TryQuerySpaces)}() failed.");
        return false;
    }

    /// <summary>
    /// Tests whether <paramref name="componentType"/> is enabled and, if not, requests that it be enabled.
    /// </summary>
    /// <returns>Returns the current state of the component.</returns>
    private bool EnableComponentIfNecessary(OVRSpace space, Guid uuid, OVRPlugin.SpaceComponentType componentType)
    {
        OVRPlugin.GetSpaceComponentStatus(space, componentType, out bool componentEnabled, out _);
        if (componentEnabled)
        {
            Verbose?.Log(nameof(OVRSceneManager),
                $"[{uuid}] {nameof(EnableComponentIfNecessary)}() component [{componentType}] is already enabled.");

            return true;
        }

        double dTimeout = 10 * 1000f;
        OVRPlugin.SetSpaceComponentStatus(space, componentType, true, dTimeout, out var requestId);
        Verbose?.Log(nameof(OVRSceneManager),
            $"[{uuid}] {nameof(EnableComponentIfNecessary)}() component [{componentType}] requested with requestId [{requestId}].");

        return false;
    }

    #endregion

    #region ActionFunctions

    private void OVRManager_SceneCaptureComplete(UInt64 requestId, bool result)
    {
        if (requestId != _sceneCaptureRequestId)
        {
            Verbose?.LogWarning(nameof(OVRSceneManager),
                $"Scene Room Capture with requestId: [{requestId}] was ignored, as it was not issued by this Scene Load request.");
            return;
        }

        Development.Log(nameof(OVRSceneManager),
            $"{nameof(OVRManager_SceneCaptureComplete)}() requestId: [{requestId}] result: [{result}]");

        if (result)
        {
            // Either the user created a room, or they confirmed that the existing room is up to date. We can now load it.
            Development.Log(nameof(OVRSceneManager),
                $"The Room Capture returned without errors. Invoking {nameof(SceneCaptureReturnedWithoutError)}.");
            SceneCaptureReturnedWithoutError?.Invoke();
        }
        else
        {
            Development.LogError(nameof(OVRSceneManager),
                $"An error occurred when sending the user to the Room Capture. Invoking {nameof(UnexpectedErrorWithSceneCapture)}.");
            UnexpectedErrorWithSceneCapture?.Invoke();
        }
    }

    private static bool IsComponentEnabled(OVRSpace space, OVRPlugin.SpaceComponentType componentType) =>
        OVRPlugin.GetSpaceComponentStatus(space, componentType, out var enabled, out _) && enabled;

    internal OVRSceneAnchor InstantiateSceneAnchor(OVRSpace space, Guid uuid, OVRSceneAnchor prefab)
    {
        // Query for the semantic classification of the object
        var hasSemanticLabels = OVRPlugin.GetSpaceSemanticLabels(space, out var labelString);
        var labels = hasSemanticLabels
            ? labelString.Split(',')
            : Array.Empty<string>();

        // Search the prefab override for a matching label, and if found override the prefab
        if (PrefabOverrides.Count > 0)
        {
            foreach (var label in labels)
            {
                // Skip empty labels
                if (string.IsNullOrEmpty(label)) continue;

                // Search the prefab override for an entry matching the label
                foreach (var @override in PrefabOverrides)
                {
                    if (@override.Prefab &&
                        @override.ClassificationLabel == label)
                    {
                        prefab = @override.Prefab;
                        break;
                    }
                }
            }
        }

        // This can occur if neither the prefab nor any matching override prefab is set in the inspector
        if (prefab == null)
        {
            Verbose?.Log(nameof(OVRSceneManager),
                $"No prefab was provided for space: [{space}]"
                + (labels.Length > 0 ? $" with semantic label {labels[0]}" : ""));
            return null;
        }

        var sceneAnchor = Instantiate(prefab, Vector3.zero, Quaternion.identity);
        sceneAnchor.Initialize(space, uuid);

        var plane = sceneAnchor.GetComponent<OVRScenePlane>();
        if (plane)
        {
            if (RoomLayout == null)
            {
                RoomLayout = new RoomLayoutInformation();
            }
            // Populate RoomLayoutInformation
            foreach (var label in labels)
            {
                switch (label)
                {
                    case Classification.Floor:
                        RoomLayout.Floor = plane;
                        break;
                    case Classification.Ceiling:
                        RoomLayout.Ceiling = plane;
                        break;
                    case Classification.WallFace:
                        RoomLayout.Walls.Add(plane);
                        break;
                }
            }
        }

        return sceneAnchor;
    }

    private void OVRManager_SpaceQueryComplete(UInt64 requestId, bool result)
    {
        if (!_individualRequestIds.Contains(requestId))
        {
            Verbose?.LogWarning(nameof(OVRSceneManager),
                $"requestId: [{requestId}] was ignored as it's not part of the Scene Load requests.");
            return;
        }

        Verbose?.Log(nameof(OVRSceneManager),
            $"{nameof(OVRManager_SpaceQueryComplete)}() requestId: [{requestId}] result: [{result}]");

        _individualRequestIds.Remove(requestId);


        if (!result)
        {
            Development.LogError(nameof(OVRSceneManager),
                $"{nameof(OVRPlugin.QuerySpaces)}() asynchronously returned a failed result. " +
                $"Invoking {nameof(NoSceneModelToLoad)}.");
            NoSceneModelToLoad?.Invoke();
            return;
        }

        if (!OVRPlugin.RetrieveSpaceQueryResults(requestId, out var results))
        {
            Development.LogError(nameof(OVRSceneManager),
                $"{nameof(OVRPlugin.RetrieveSpaceQueryResults)}() could not retrieve results.");
            return;
        }

        if (results == null || results.Length == 0)
        {
            Development.LogWarning(nameof(OVRSceneManager),
                "Loading the Scene definition yielded no result. "
                + "Typically, this means the user has not captured the room they are in yet. "
                + "Alternatively, an internal error may be preventing this app from accessing scene. "
                + $"Invoking {nameof(NoSceneModelToLoad)}.");

            NoSceneModelToLoad?.Invoke();
            return;
        }


        foreach (var queryResult in results)
        {
            ProcessQueryResult(queryResult);
        }


        CheckForCompletion();
    }

    private void CheckForCompletion()
    {
        // Requests can be nested, so we have to wait for the last one to be complete before applying
        // any judgement on the final outcome.
        if (_individualRequestIds.Count == 0 && _pendingLocatable.Count == 0 && !_sceneLoadComplete)
        {
            Development.Log(nameof(OVRSceneManager),
                $"Scene Model was loaded successfully. Invoking {nameof(SceneModelLoadedSuccessfully)}.");
            RoomLayout?.Walls.Sort(_wallOrderComparer);
            SceneModelLoadedSuccessfully?.Invoke();
            _sceneLoadComplete = true;
        }
    }

    private void OVRManager_SpaceSetComponentStatusComplete(UInt64 requestId, bool result, OVRSpace space, Guid uuid,
        OVRPlugin.SpaceComponentType componentType, bool isEnabled)
    {
        if (!result)
        {
#if DEVELOPMENT_BUILD
            if (_pendingLocatable.ContainsKey(space))
            {
                Development.LogError(nameof(OVRSceneManager),
                    $"[{uuid}] {nameof(OVRManager)}.{nameof(OVRManager.SpaceSetComponentStatusComplete)} failed for component {componentType}.");
            }
#endif
            return;
        }

        if (componentType == OVRPlugin.SpaceComponentType.Locatable &&
            isEnabled &&
            _pendingLocatable.TryGetValue(space, out var spaceQueryResult))
        {
            Development.Log(nameof(OVRSceneManager), $"[{uuid}] is now locatable.");
            _pendingLocatable.Remove(space);
            ProcessQueryResult(spaceQueryResult);
            CheckForCompletion();
        }
    }

    private void ProcessQueryResult(OVRPlugin.SpaceQueryResult queryResult)
    {
        var space = queryResult.space;
        var uuid = queryResult.uuid;

        OVRPlugin.GetSpaceComponentStatus(space, OVRPlugin.SpaceComponentType.Bounded3D, out bool bounded3dEnabled, out _);
        OVRPlugin.GetSpaceComponentStatus(space, OVRPlugin.SpaceComponentType.Bounded2D, out bool bounded2dEnabled, out _);
        OVRPlugin.GetSpaceComponentStatus(space, OVRPlugin.SpaceComponentType.RoomLayout, out bool roomLayoutEnabled, out _);

        IEnumerable<string> EnabledComponents()
        {
            if (IsComponentEnabled(space, OVRPlugin.SpaceComponentType.Locatable))
                yield return nameof(OVRPlugin.SpaceComponentType.Locatable);
            if (bounded2dEnabled) yield return nameof(OVRPlugin.SpaceComponentType.Bounded2D);
            if (bounded3dEnabled) yield return nameof(OVRPlugin.SpaceComponentType.Bounded3D);
            if (IsComponentEnabled(space, OVRPlugin.SpaceComponentType.SemanticLabels))
            {
                if (OVRPlugin.GetSpaceSemanticLabels(space, out var labels))
                {
                    yield return $"{nameof(OVRPlugin.SpaceComponentType.SemanticLabels)} ({labels})";
                }
                else
                {
                    yield return $"{nameof(OVRPlugin.SpaceComponentType.SemanticLabels)} (none)";
                }
            }

            if (roomLayoutEnabled) yield return nameof(OVRPlugin.SpaceComponentType.RoomLayout);
        }

        Verbose?.Log(nameof(OVRSceneManager),
            $"[{uuid}] {nameof(OVRManager_SpaceQueryComplete)}() Enabled components: {string.Join(", ", EnabledComponents())}");

        if (bounded2dEnabled || bounded3dEnabled)
        {
            // Validate only allowed components are set
            if (roomLayoutEnabled)
            {
                Development.LogError(nameof(OVRSceneManager),
                    $"[{uuid}] {nameof(OVRManager_SpaceQueryComplete)}() Anchor has incompatible components. {nameof(OVRPlugin.SpaceComponentType.RoomLayout)} should not be enabled with {nameof(OVRPlugin.SpaceComponentType.Bounded2D)} or {nameof(OVRPlugin.SpaceComponentType.Bounded3D)}.");
                return;
            }

            // Enable Locatable component, as it is not enabled when the space is loaded from storage for the first time.
            var locatableEnabled = EnableComponentIfNecessary(space, uuid, OVRPlugin.SpaceComponentType.Locatable);
            if (!locatableEnabled)
            {
                Development.Log(nameof(OVRSceneManager),
                    $"[{uuid}] Waiting for spatial entity to become {nameof(OVRPlugin.SpaceComponentType.Locatable)}.");
                _pendingLocatable[queryResult.space] = queryResult;
                return;
            }

            if (OVRSceneAnchor.SceneAnchors.ContainsKey(uuid))
            {
                return;
            }

            // Ignoring this anchor because it has been destroyed.
            if (OVRSceneAnchor.DestroyedSceneAnchors.Contains(uuid))
            {
                return;
            }

            InstantiateSceneAnchor(space, uuid, bounded2dEnabled ? PlanePrefab : VolumePrefab);
        }
        else if (roomLayoutEnabled)
        {
            bool roomLayoutSuccess = OVRPlugin.GetSpaceRoomLayout(space, out var roomLayout);
            if (!roomLayoutSuccess)
            {
                Development.LogError(nameof(OVRSceneManager),
                    $"[{uuid}] has component {nameof(OVRPlugin.SpaceComponentType.RoomLayout)} but {nameof(OVRPlugin.GetSpaceRoomLayout)} failed. Ignoring room.");
                return;
            }

            _uuidSet.Clear();
            if (!roomLayout.floorUuid.Equals(Guid.Empty))
            {
                _uuidSet.Add(roomLayout.floorUuid);
                Verbose?.Log(nameof(OVRSceneManager),
                    $"{nameof(OVRPlugin.GetSpaceRoomLayout)}: floor [{roomLayout.floorUuid}]");
            }

            if (!roomLayout.ceilingUuid.Equals(Guid.Empty))
            {
                _uuidSet.Add(roomLayout.ceilingUuid);
                Verbose?.Log(nameof(OVRSceneManager),
                    $"{nameof(OVRPlugin.GetSpaceRoomLayout)}: ceiling [{roomLayout.ceilingUuid}]");
            }

            _orderedRoomGuids.Clear();
            int validWallsCount = 0;
            foreach (var wallUuid in roomLayout.wallUuids)
            {
                if (!wallUuid.Equals(Guid.Empty))
                {
                    _uuidSet.Add(wallUuid);
                    Verbose?.Log(nameof(OVRSceneManager),
                        $"{nameof(OVRPlugin.GetSpaceRoomLayout)}: wall [{wallUuid}]");
                    _orderedRoomGuids[wallUuid] = validWallsCount++;
                }
            }

            Verbose?.Log(nameof(OVRSceneManager),
                $"{nameof(OVRPlugin.GetSpaceRoomLayout)}: wall count [{validWallsCount}]");

            bool containerSuccess = OVRPlugin.GetSpaceContainer(space, out var containerUuids);
            Verbose?.Log(nameof(OVRSceneManager),
                $"{nameof(OVRPlugin.GetSpaceContainer)}: success [{containerSuccess}], count [{containerUuids.Length}]");

            if (containerSuccess)
            {
                foreach (var containerUuid in containerUuids)
                {
                    Verbose?.Log(nameof(OVRSceneManager),
                        $"{nameof(OVRPlugin.GetSpaceContainer)}: UUID [{containerUuid}]");

                    if (!containerUuid.Equals(Guid.Empty))
                    {
                        _uuidSet.Add(containerUuid);
                    }
                }
            }

            _uuidsToQuery = _uuidSet.ToList();
            _currentQueryMode = QueryMode.QueryByUuid;
            LoadSpatialEntities();
        }
    }

    #endregion
}

