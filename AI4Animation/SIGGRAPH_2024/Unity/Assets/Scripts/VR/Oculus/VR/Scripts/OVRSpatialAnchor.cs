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
using System.Diagnostics;
#if DEVELOPMENT_BUILD
using System.Linq;
#endif
using Unity.Collections;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// Represents a spatial anchor.
/// </summary>
/// <remarks>
/// This component can be used in two ways: to create a new spatial anchor or to bind to an existing spatial anchor.
///
/// To create a new spatial anchor, simply add this component to any GameObject. The transform of the GameObject is used
/// to create a new spatial anchor in the Oculus Runtime. Afterwards, the GameObject's transform will be updated
/// automatically. The creation operation is asynchronous, and, if it fails, this component will be destroyed.
///
/// To load previously saved anchors and bind them to an <see cref="OVRSpatialAnchor"/>, see
/// <see cref="LoadUnboundAnchors"/>.
/// </remarks>
[DisallowMultipleComponent]
public class OVRSpatialAnchor : MonoBehaviour
{
	private bool _startCalled;

	private ulong _requestId;

	private readonly SaveOptions _defaultSaveOptions = new SaveOptions
	{
		Storage = OVRSpace.StorageLocation.Local,
	};

	private readonly EraseOptions _defaultEraseOptions = new EraseOptions
	{
		Storage = OVRSpace.StorageLocation.Local,
	};

	/// <summary>
	/// The space associated with this spatial anchor.
	/// </summary>
	/// <remarks>
	/// The <see cref="OVRSpace"/> represents the runtime instance of the spatial anchor and will change across
	/// different sessions.
	/// </remarks>
	public OVRSpace Space { get; private set; }

	/// <summary>
	/// The UUID associated with this spatial anchor.
	/// </summary>
	/// <remarks>
	/// UUIDs persist across sessions and applications. If you load a persisted anchor, you can use the UUID to identify
	/// it.
	/// </remarks>
	public Guid Uuid { get; private set; }

	/// <summary>
	/// Whether the spatial anchor has been created.
	/// </summary>
	/// <remarks>
	/// Creation is asynchronous and may take several frames. If creation fails, this component is destroyed.
	/// </remarks>
	public bool Created => Space.Valid;

	/// <summary>
	/// Whether the spatial anchor is pending creation.
	/// </summary>
	public bool PendingCreation => _requestId != 0;

	/// <summary>
	/// Whether the spatial anchor has been localized.
	/// </summary>
	/// <remarks>
	/// When you create a new spatial anchor, it may take a few frames before it is localized. Once localized,
	/// its transform will update automatically.
	/// </remarks>
	public bool Localized => Space.Valid &&
	                         OVRPlugin.GetSpaceComponentStatus(Space, OVRPlugin.SpaceComponentType.Locatable,
		                         out var isEnabled, out _) && isEnabled;

	/// <summary>
	/// Initializes this component from an existing space handle and uuid, e.g., the result of a call to
	/// <see cref="OVRPlugin.QuerySpaces"/>.
	/// </summary>
	/// <remarks>
	/// This method allows you to associate this component with an existing spatial anchor, e.g., one that was saved in
	/// a previous session. Do not call this method to create a new spatial anchor.
	///
	/// If you call this method, you must do so prior to the component's `Start` method. You cannot change the spatial
	/// anchor associated with this component after that.
	/// </remarks>
	/// <param name="space">The existing <see cref="OVRSpace"/> to associate with this spatial anchor.</param>
	/// <param name="uuid">The universally unique identifier to associate with this spatial anchor.</param>
	/// <exception cref="InvalidOperationException">Thrown if `Start` has already been called on this component.</exception>
	/// <exception cref="ArgumentException">Thrown if <paramref name="space"/> is not <see cref="OVRSpace.Valid"/>.</exception>
	public void InitializeFromExisting(OVRSpace space, Guid uuid)
	{
		if (_startCalled)
			throw new InvalidOperationException($"Cannot call {nameof(InitializeFromExisting)} after {nameof(Start)}. This must be set once upon creation.");

		try
		{
			if (!space.Valid)
				throw new ArgumentException($"Invalid space {space}.", nameof(space));

			ThrowIfBound(uuid);
		}
		catch
		{
			Destroy(this);
			throw;
		}

		InitializeUnchecked(space, uuid);
	}

	/// <summary>
	/// Saves the <see cref="OVRSpatialAnchor"/> to local persistent storage.
	/// </summary>
	/// <remarks>
	/// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
	///
	/// When saved, an <see cref="OVRSpatialAnchor"/> can be loaded by a different session or application. Use the
	/// <see cref="Uuid"/> to identify the same <see cref="OVRSpatialAnchor"/> at a future time.
	///
	/// This operation fully succeeds or fails; that is, either all anchors are successfully saved,
	/// or the operation fails.
	/// </remarks>
	/// <param name="onComplete">
	/// Invoked when the save operation completes. May be null. Parameters are
	/// - <see cref="OVRSpatialAnchor"/>: The anchor being saved.
	/// - `bool`: A value indicating whether the save operation succeeded.
	/// </param>
	public void Save(Action<OVRSpatialAnchor, bool> onComplete = null)
	{
		Save(_defaultSaveOptions, onComplete);
	}

	private static NativeArray<ulong> ToNativeArray(ICollection<OVRSpatialAnchor> anchors)
	{
		var count = anchors.Count;
		var spaces = new NativeArray<ulong>(count, Allocator.Temp);
		if (anchors is IReadOnlyList<OVRSpatialAnchor> list)
		{
			// We can save a GC alloc (the Enumerator<T>) if it can be used as a list.
			for (var i = 0; i < count; i++)
			{
				spaces[i] = list[i] ? list[i].Space : 0;
			}
		}
		else
		{
			var i = 0;
			foreach (var anchor in anchors)
			{
				spaces[i++] = anchor ? anchor.Space : 0;
			}
		}

		return spaces;
	}

	/// <summary>
	/// Saves the <see cref="OVRSpatialAnchor"/> with specified <see cref="SaveOptions"/>.
	/// </summary>
	/// <remarks>
	/// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
	/// When saved, the <see cref="OVRSpatialAnchor"/> can be loaded by a different session or application. Use the
	/// <see cref="Uuid"/> to identify the same <see cref="OVRSpatialAnchor"/> at a future time.
	///
	/// This operation fully succeeds or fails; that is, either all anchors are successfully saved,
	/// or the operation fails.
	/// </remarks>
	/// <param name="saveOptions">Options how the anchor will be saved.</param>
	/// <param name="onComplete">
	/// Invoked when the save operation completes. May be null. Parameters are
	/// - <see cref="OVRSpatialAnchor"/>: The anchor being saved.
	/// - `bool`: A value indicating whether the save operation succeeded.
	/// </param>
	public void Save(SaveOptions saveOptions, Action<OVRSpatialAnchor, bool> onComplete = null)
	{
		var saved = OVRPlugin.SaveSpace(Space, saveOptions.Storage.ToSpaceStorageLocation(),
			OVRPlugin.SpaceStoragePersistenceMode.Indefinite, out var requestId);


		void Callback(OVRSpatialAnchor anchor, bool success)
		{
			onComplete?.Invoke(anchor, success);
		}

		if (saved)
		{
			Development.LogRequest(requestId, $"[{Uuid}] Saving spatial anchor...");
			SingleAnchorCompletionDelegates[requestId] = new SingleAnchorDelegatePair
			{
				Anchor = this,
				Delegate = Callback
			};
		}
		else
		{
			Development.LogError($"[{Uuid}] {nameof(OVRPlugin)}.{nameof(OVRPlugin.SaveSpace)} failed.");
			Callback(this, false);
		}
	}

	/// <summary>
	/// Saves a collection of <see cref="OVRSpatialAnchor"/> to specified storage.
	/// </summary>
	/// <remarks>
	/// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
	/// When saved, an <see cref="OVRSpatialAnchor"/> can be loaded by a different session or application. Use the
	/// <see cref="Uuid"/> to identify the same <see cref="OVRSpatialAnchor"/> at a future time.
	/// </remarks>
	/// <param name="anchors">Collection of anchors</param>
	/// <param name="saveOptions">Options how the anchors will be saved.</param>
	/// <param name="onComplete">
	/// Invoked when the save operation completes. May be null. <paramref name="onComplete"/> receives two parameters:
	/// - `ICollection&lt;OVRSpatialAnchor&gt;`: The same collection as in <paramref name="anchors"/> parameter
	/// - `OperationResult`: An error code indicating whether the save operation succeeded or not.
	/// </param>
	/// <exception cref="ArgumentNullException">Thrown if <paramref name="anchors"/> is `null`.</exception>
	public static void Save(ICollection<OVRSpatialAnchor> anchors, SaveOptions saveOptions, Action<ICollection<OVRSpatialAnchor>, OperationResult> onComplete = null)
    {
        if (anchors == null)
            throw new ArgumentNullException(nameof(anchors));

        using (var spaces = ToNativeArray(anchors))
        {
            var saveResult = OVRPlugin.SaveSpaceList(spaces, saveOptions.Storage.ToSpaceStorageLocation(), out var requestId);
            if (saveResult.IsSuccess())
            {
                Development.LogRequest(requestId, $"Saving spatial anchors...");
                if (onComplete != null)
                {
                    MultiAnchorCompletionDelegates[requestId] = new MultiAnchorDelegatePair
                    {
                        Anchors = anchors,
                        Delegate = onComplete
                    };
                }
            }
            else
            {
                Development.LogError($"{nameof(OVRPlugin)}.{nameof(OVRPlugin.SaveSpaceList)} failed with error {saveResult}.");
                onComplete?.Invoke(anchors, (OperationResult)saveResult);
            }
        }
    }

	/// <summary>
	/// Shares a collection of <see cref="OVRSpatialAnchor"/> to specified users.
	/// Specified users will be able to download, track, and share specified anchors.
	/// </summary>
	/// <remarks>
	/// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
	///
	/// This operation fully succeeds or fails; that is, either all anchors are successfully shared,
	/// or the operation fails.
	/// </remarks>
	/// <param name="anchors">The collection of anchors to share.</param>
	/// <param name="users">An array of Oculus users to share these anchors with.</param>
	/// <param name="onComplete">
	/// Invoked when the share operation completes. May be null. Delegate parameter is
	/// - `ICollection&lt;OVRSpatialAnchor&gt;`: The collection of anchors being shared.
	/// - `OperationResult`: An error code indicating whether the share operation succeeded or not.
	/// </param>
	/// <exception cref="ArgumentNullException">Thrown if <paramref name="anchors"/> is `null`.</exception>
	public static void Share(ICollection<OVRSpatialAnchor> anchors, ICollection<OVRSpaceUser> users, Action<ICollection<OVRSpatialAnchor>, OperationResult> onComplete = null)
    {
        if (anchors == null)
            throw new ArgumentNullException(nameof(anchors));

        using var spaces = ToNativeArray(anchors);

        var handles = new NativeArray<ulong>(users.Count, Allocator.Temp);
        using var disposer = handles;
        int i = 0;
        foreach (var user in users)
        {
            handles[i++] = user._handle;
        }

        var shareResult = OVRPlugin.ShareSpaces(spaces, handles, out var requestId);
        if (shareResult.IsSuccess())
        {
            Development.LogRequest(requestId, $"Sharing {(uint)spaces.Length} spatial anchors...");
            if (onComplete != null)
            {
                MultiAnchorCompletionDelegates[requestId] = new MultiAnchorDelegatePair
                {
                    Anchors = anchors,
                    Delegate = onComplete
                };
            }
        }
        else
        {
            Development.LogError($"{nameof(OVRPlugin)}.{nameof(OVRPlugin.ShareSpaces)}  failed with error {shareResult}.");
            onComplete?.Invoke(anchors, (OperationResult)shareResult);
        }
    }

    /// <summary>
    /// Erases the <see cref="OVRSpatialAnchor"/> from persistent storage.
    /// </summary>
    /// <remarks>
    /// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
    /// Erasing an <see cref="OVRSpatialAnchor"/> does not destroy the anchor.
    /// </remarks>
    /// <param name="onComplete">
    /// Invoked when the erase operation completes. May be null. Parameters are
    /// - <see cref="OVRSpatialAnchor"/>: The anchor being erased.
    /// - `bool`: A value indicating whether the erase operation succeeded.
    /// </param>
    public void Erase(Action<OVRSpatialAnchor, bool> onComplete = null)
    {
        Erase(_defaultEraseOptions, onComplete);
    }

    /// <summary>
    /// Erases the <see cref="OVRSpatialAnchor"/> from specified storage.
    /// </summary>
    /// <remarks>
    /// This method is asynchronous; use <paramref name="onComplete"/> to be notified of completion.
    /// Erasing an <see cref="OVRSpatialAnchor"/> does not destroy the anchor.
    /// </remarks>
    /// <param name="eraseOptions">Options how the anchor should be erased.</param>
    /// <param name="onComplete">
    /// Invoked when the erase operation completes. May be null. Parameters are
    /// - <see cref="OVRSpatialAnchor"/>: The anchor being erased.
    /// - `bool`: A value indicating whether the erase operation succeeded.
    /// </param>
    public void Erase(EraseOptions eraseOptions, Action<OVRSpatialAnchor, bool> onComplete = null)
    {
	    var erased = OVRPlugin.EraseSpace(Space, eraseOptions.Storage.ToSpaceStorageLocation(), out var requestId);


	    void Callback(OVRSpatialAnchor anchor, bool success)
	    {
		    onComplete?.Invoke(anchor, success);
	    }

	    if (erased)
	    {
		    Development.LogRequest(requestId, $"[{Uuid}] Erasing spatial anchor...");
		    SingleAnchorCompletionDelegates[requestId] = new SingleAnchorDelegatePair
		    {
			    Anchor = this,
			    Delegate = Callback
		    };
	    }
	    else
	    {
		    Development.LogError($"[{Uuid}] {nameof(OVRPlugin)}.{nameof(OVRPlugin.EraseSpace)} failed.");
		    Callback(this, false);
	    }
    }

    private static void ThrowIfBound(Guid uuid)
    {
        if (SpatialAnchors.ContainsKey(uuid))
            throw new InvalidOperationException($"Spatial anchor with uuid {uuid} is already bound to an {nameof(OVRSpatialAnchor)}.");
    }

    // Initializes this component without checking preconditions
    private void InitializeUnchecked(OVRSpace space, Guid uuid)
    {
        SpatialAnchors.Add(uuid, this);
        _requestId = 0;
        Space = space;
        Uuid = uuid;
        OVRPlugin.SetSpaceComponentStatus(Space, OVRPlugin.SpaceComponentType.Locatable, true, 0, out _);
        OVRPlugin.SetSpaceComponentStatus(Space, OVRPlugin.SpaceComponentType.Storable, true, 0, out _);
        OVRPlugin.SetSpaceComponentStatus(Space, OVRPlugin.SpaceComponentType.Sharable, true, 0, out _);

        // Try to update the pose as soon as we can.
        UpdateTransform();
    }

    private void Start()
    {
        _startCalled = true;

        if (Space.Valid)
        {
            Development.Log($"[{Uuid}] Created spatial anchor from existing an existing space.");
        }
        else
        {
            CreateSpatialAnchor();
        }
    }

    private void Update()
    {
        if (Space.Valid)
        {
            UpdateTransform();
        }
    }

    private void OnDestroy()
    {
        if (Space.Valid)
        {
            OVRPlugin.DestroySpace(Space);
        }

        SpatialAnchors.Remove(Uuid);
    }

    private OVRPose GetTrackingSpacePose()
    {
        var mainCamera = Camera.main;
        if (mainCamera)
        {
            return transform.ToTrackingSpacePose(mainCamera);
        }

        Development.LogWarning($"No main camera found. Using world-space pose.");
        return transform.ToOVRPose(isLocal: false);
    }

    private void CreateSpatialAnchor()
    {
	    var created = OVRPlugin.CreateSpatialAnchor(new OVRPlugin.SpatialAnchorCreateInfo
	    {
		    BaseTracking = OVRPlugin.GetTrackingOriginType(),
		    PoseInSpace = GetTrackingSpacePose().ToPosef(),
		    Time = OVRPlugin.GetTimeInSeconds(),
	    }, out _requestId);


	    if (created)
	    {
		    Development.LogRequest(_requestId, $"Creating spatial anchor...");
		    CreationRequests[_requestId] = this;
	    }
	    else
	    {
		    Development.LogError($"{nameof(OVRPlugin)}.{nameof(OVRPlugin.CreateSpatialAnchor)} failed. Destroying {nameof(OVRSpatialAnchor)} component.");
		    Destroy(this);
	    }
    }

    internal static bool TryGetPose(OVRSpace space, out OVRPose pose)
    {
        if (!OVRPlugin.TryLocateSpace(space, OVRPlugin.GetTrackingOriginType(), out var posef))
        {
            pose = OVRPose.identity;
            return false;
        }

        pose = posef.ToOVRPose();
        var mainCamera = Camera.main;
        if (mainCamera)
        {
            pose = pose.ToWorldSpacePose(mainCamera);
        }

        return true;
    }

    private void UpdateTransform()
    {
        if (TryGetPose(Space, out var pose))
        {
            transform.SetPositionAndRotation(pose.position, pose.orientation);
        }
    }

    private static bool TryExtractValue<TKey, TValue>(Dictionary<TKey, TValue> dict, TKey key, out TValue value) =>
        dict.TryGetValue(key, out value) && dict.Remove(key);

    private struct SingleAnchorDelegatePair
    {
        public OVRSpatialAnchor Anchor;
        public Action<OVRSpatialAnchor, bool> Delegate;
    }

    private struct MultiAnchorDelegatePair
    {
        public ICollection<OVRSpatialAnchor> Anchors;
        public Action<ICollection<OVRSpatialAnchor>, OperationResult> Delegate;
    }

    internal static readonly Dictionary<Guid, OVRSpatialAnchor> SpatialAnchors =
        new Dictionary<Guid, OVRSpatialAnchor>();

    private static readonly Dictionary<ulong, OVRSpatialAnchor> CreationRequests =
        new Dictionary<ulong, OVRSpatialAnchor>();

    private static readonly Dictionary<ulong, SingleAnchorDelegatePair> SingleAnchorCompletionDelegates =
        new Dictionary<ulong, SingleAnchorDelegatePair>();

    private static readonly Dictionary<ulong, MultiAnchorDelegatePair> MultiAnchorCompletionDelegates =
        new Dictionary<ulong, MultiAnchorDelegatePair>();

    private static readonly Dictionary<ulong, Action<UnboundAnchor, bool>> LocalizationDelegates =
        new Dictionary<ulong, Action<UnboundAnchor, bool>>();

    private static readonly Dictionary<ulong, Action<UnboundAnchor[]>> Queries =
        new Dictionary<ulong, Action<UnboundAnchor[]>>();

    private static readonly List<UnboundAnchor> UnboundAnchorBuffer = new List<UnboundAnchor>();

    private static readonly OVRPlugin.SpaceComponentType[] ComponentTypeBuffer = new OVRPlugin.SpaceComponentType[32];

    [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
    static void InitializeOnLoad()
    {
        CreationRequests.Clear();
        SingleAnchorCompletionDelegates.Clear();
        MultiAnchorCompletionDelegates.Clear();
        LocalizationDelegates.Clear();
        Queries.Clear();
        UnboundAnchorBuffer.Clear();
        SpatialAnchors.Clear();
    }

    static OVRSpatialAnchor()
    {
        OVRManager.SpatialAnchorCreateComplete += OnSpatialAnchorCreateComplete;
        OVRManager.SpaceSaveComplete += OnSpaceSaveComplete;
        OVRManager.SpaceListSaveComplete += OnSpaceListSaveComplete;
        OVRManager.ShareSpacesComplete += OnShareSpacesComplete;
        OVRManager.SpaceEraseComplete += OnSpaceEraseComplete;
        OVRManager.SpaceQueryComplete += OnSpaceQueryComplete;
        OVRManager.SpaceSetComponentStatusComplete += OnSpaceSetComponentStatusComplete;
    }

    private static void InvokeSingleAnchorDelegate(ulong requestId, bool result)
    {
        if (TryExtractValue(SingleAnchorCompletionDelegates, requestId, out var value))
        {
            value.Delegate(value.Anchor, result);
        }
    }

    private static void InvokeMultiAnchorDelegate(ulong requestId, OperationResult result)
    {
        if (TryExtractValue(MultiAnchorCompletionDelegates, requestId, out var value))
        {
            value.Delegate(value.Anchors, result);
        }
    }

    private static void OnSpatialAnchorCreateComplete(ulong requestId, bool success, OVRSpace space, Guid uuid)
    {
        Development.LogRequestResult(requestId, success,
            $"[{uuid}] Spatial anchor created.",
            $"Failed to create spatial anchor. Destroying {nameof(OVRSpatialAnchor)} component.");

        if (!TryExtractValue(CreationRequests, requestId, out var anchor)) return;


        if (success && anchor)
        {
            // All good; complete setup of OVRSpatialAnchor component.
            anchor.InitializeUnchecked(space, uuid);
        }
        else if (success && !anchor)
        {
            // Creation succeeded, but the OVRSpatialAnchor component was destroyed before the callback completed.
            OVRPlugin.DestroySpace(space);
        }
        else if (!success && anchor)
        {
            // The OVRSpatialAnchor component exists but creation failed.
            Destroy(anchor);
        }
        // else if creation failed and the OVRSpatialAnchor component was destroyed, nothing to do.
    }

    private static void OnSpaceSaveComplete(ulong requestId, OVRSpace space, bool result, Guid uuid)
    {
        Development.LogRequestResult(requestId, result,
            $"[{uuid}] Saved.",
            $"[{uuid}] Save failed.");

        InvokeSingleAnchorDelegate(requestId, result);
    }

    private static void OnSpaceEraseComplete(ulong requestId, bool result, Guid uuid, OVRPlugin.SpaceStorageLocation location)
    {
        Development.LogRequestResult(requestId, result,
            $"[{uuid}] Erased.",
            $"[{uuid}] Erase failed.");

        InvokeSingleAnchorDelegate(requestId, result);
    }

    /// <summary>
    /// Options for loading unbound spatial anchors used by <see cref="OVRSpatialAnchor.LoadUnboundAnchors"/>.
    /// </summary>
    public struct LoadOptions
    {
        /// <summary>
        /// The maximum number of uuids that may be present in the <see cref="Uuids"/> collection.
        /// </summary>
        public const int MaxSupported = OVRSpaceQuery.Options.MaxUuidCount;

        /// <summary>
        /// The storage location from which to query spatial anchors.
        /// </summary>
        public OVRSpace.StorageLocation StorageLocation { get; set; }

        /// <summary>
        /// (Obsolete) The maximum number of anchors to query.
        /// </summary>
        /// <remarks>
        /// In prior SDK versions, it was mandatory to set this property in order to receive any
        /// results. However, this property is now obsolete. If <see cref="MaxAnchorCount"/> is zero,
        /// i.e., the default initialized value, it will automatically be set to the count of
        /// <see cref="Uuids"/>.
        ///
        /// If non-zero, the number of anchors in the result will be limited to
        /// <see cref="MaxAnchorCount"/>, preserving the previous behavior.
        /// </remarks>
        [Obsolete("This property is no longer required. MaxAnchorCount will be automatically set to the number of uuids to load.")]
        public int MaxAnchorCount { get; set; }

        /// <summary>
        /// The timeout, in seconds, for the query operation.
        /// </summary>
        /// <remarks>
        /// A value of zero indicates no timeout.
        /// </remarks>
        public double Timeout { get; set; }

        /// <summary>
        /// The set of spatial anchors to query, identified by their UUIDs.
        /// </summary>
        /// <remarks>
        /// The UUIDs are copied by the <see cref="OVRSpatialAnchor.LoadUnboundAnchors"/> method and no longer
        /// referenced internally afterwards.
        ///
        /// You must supply a list of UUIDs. <see cref="OVRSpatialAnchor.LoadUnboundAnchors"/> will throw if this
        /// property is null.
        /// </remarks>
        /// <exception cref="System.ArgumentException">Thrown if <see cref="Uuids"/> contains more
        ///     than <see cref="MaxSupported"/> elements.</exception>
        public IReadOnlyList<Guid> Uuids
        {
            get => _uuids;
            set
            {
                if (value?.Count > OVRSpaceQuery.Options.MaxUuidCount)
                    throw new ArgumentException($"There must not be more than {MaxSupported} UUIDs (new value contains {value.Count} UUIDs).", nameof(value));

                _uuids = value;
            }
        }
        private IReadOnlyList<Guid> _uuids;

        internal OVRSpaceQuery.Options ToQueryOptions() => new OVRSpaceQuery.Options
        {
            Location = StorageLocation,
#pragma warning disable CS0618
			MaxResults = MaxAnchorCount == 0 ? Uuids?.Count ?? 0 : MaxAnchorCount,
#pragma warning restore CS0618
			Timeout = Timeout,
			UuidFilter = Uuids,
			QueryType = OVRPlugin.SpaceQueryType.Action,
			ActionType = OVRPlugin.SpaceQueryActionType.Load,
		};
	}

	/// <summary>
	/// A spatial anchor that has not been bound to an <see cref="OVRSpatialAnchor"/>.
	/// </summary>
	/// <remarks>
	/// Use this object to bind an unbound spatial anchor to an <see cref="OVRSpatialAnchor"/>.
	/// </remarks>
	public readonly struct UnboundAnchor
	{
		internal readonly OVRSpace _space;

		/// <summary>
		/// The universally unique identifier associated with this anchor.
		/// </summary>
		public Guid Uuid { get; }

		/// <summary>
		/// Whether the anchor has been localized.
		/// </summary>
		/// <remarks>
		/// Prior to localization, the anchor's <see cref="Pose"/> cannot be determined.
		/// </remarks>
		/// <seealso cref="Localized"/>
		/// <seealso cref="Localizing"/>
		public bool Localized => OVRPlugin.GetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Locatable,
			out var enabled, out _) && enabled;

		/// <summary>
		/// Whether the anchor is in the process of being localized.
		/// </summary>
		/// <seealso cref="Localized"/>
		/// <seealso cref="Localize"/>
		public bool Localizing => OVRPlugin.GetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Locatable,
			out var enabled, out var pending) && !enabled && pending;

		/// <summary>
		/// The world space pose of the spatial anchor.
		/// </summary>
		public Pose Pose
		{
			get
			{
				if (!TryGetPose(_space, out var pose))
					throw new InvalidOperationException($"[{Uuid}] Anchor must be localized before obtaining its pose.");

				return new Pose(pose.position, pose.orientation);
			}
		}

		/// <summary>
		/// Localizes an anchor.
		/// </summary>
		/// <remarks>
		/// The delegate supplied to <see cref="OVRSpatialAnchor.LoadUnboundAnchors"/> receives an array of unbound
		/// spatial anchors. You can choose whether to localize each one and be notified when localization completes.
		///
		/// The <paramref name="onComplete"/> delegate receives two arguments:
		/// - <see cref="UnboundAnchor"/>: The anchor to bind
		/// - `bool`: Whether localization was successful
		///
		/// Upon successful localization, your delegate should instantiate an <see cref="OVRSpatialAnchor"/>, then bind
		/// the <see cref="UnboundAnchor"/> to the <see cref="OVRSpatialAnchor"/> by calling
		/// <see cref="UnboundAnchor.BindTo"/>. Once an <see cref="UnboundAnchor"/> is bound to an
		/// <see cref="OVRSpatialAnchor"/>, it cannot be used again; that is, it cannot be bound to multiple
		/// <see cref="OVRSpatialAnchor"/> components.
		/// </remarks>
		/// <param name="onComplete">A delegate invoked when localization completes (which may fail). The delegate
		/// receives two arguments:
		/// - <see cref="UnboundAnchor"/>: The anchor to bind
		/// - `bool`: Whether localization was successful
		/// </param>
		/// <param name="timeout">The timeout, in seconds, to attempt localization, or zero to indicate no timeout.</param>
		/// <exception cref="InvalidOperationException">Thrown if
		/// - The anchor does not support localization, e.g., because it is invalid.
		/// - The anchor has already been localized.
		/// - The anchor is being localized, e.g., because <see cref="Localize"/> was previously called.
		/// </exception>
		public void Localize(Action<UnboundAnchor, bool> onComplete = null, double timeout = 0)
		{
			if (!OVRPlugin.GetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Locatable, out var enabled, out var changePending))
				throw new InvalidOperationException($"[{Uuid}] {nameof(UnboundAnchor)} does not support localization.");

			if (enabled)
				throw new InvalidOperationException($"[{Uuid}] Anchor has already been localized.");

			if (changePending)
				throw new InvalidOperationException($"[{Uuid}] Anchor is currently being localized.");

			var setStatus = OVRPlugin.SetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Locatable, true, timeout, out var requestId);


			void Callback(UnboundAnchor anchor, bool success)
			{

				onComplete?.Invoke(anchor, success);
			}

			if (!setStatus)
			{
				Development.LogError($"[{Uuid}] {nameof(OVRPlugin.SetSpaceComponentStatus)} failed.");
				Callback(this, false);
				return;
			}

			Development.LogRequest(requestId,
				$"[{Uuid}] {nameof(OVRPlugin.SetSpaceComponentStatus)} enable {nameof(OVRPlugin.SpaceComponentType.Locatable)}.");

			if (onComplete != null)
			{
				LocalizationDelegates[requestId] = Callback;
			}

			OVRPlugin.SetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Storable, true, 0, out _);
			OVRPlugin.SetSpaceComponentStatus(_space, OVRPlugin.SpaceComponentType.Sharable, true, 0, out _);
		}

		/// <summary>
		/// Binds an unbound anchor to an <see cref="OVRSpatialAnchor"/> component.
		/// </summary>
		/// <remarks>
		/// Use this to bind an unbound anchor to an <see cref="OVRSpatialAnchor"/>. After <see cref="BindTo"/> is used
		/// to bind an <see cref="UnboundAnchor"/> to an <see cref="OVRSpatialAnchor"/>, the
		/// <see cref="UnboundAnchor"/> is no longer valid; that is, it cannot be bound to another
		/// <see cref="OVRSpatialAnchor"/>.
		/// </remarks>
		/// <param name="spatialAnchor">The component to which this unbound anchor should be bound.</param>
		/// <exception cref="InvalidOperationException">Thrown if this <see cref="UnboundAnchor"/> does not refer to a valid anchor.</exception>
		/// <exception cref="ArgumentNullException">Thrown if <paramref name="spatialAnchor"/> is `null`.</exception>
		/// <exception cref="ArgumentException">Thrown if an anchor is already bound to <paramref name="spatialAnchor"/>.</exception>
		/// <exception cref="ArgumentException">Thrown if <paramref name="spatialAnchor"/> is pending creation (see <see cref="OVRSpatialAnchor.PendingCreation"/>).</exception>
		/// <exception cref="InvalidOperationException">Thrown if this <see cref="UnboundAnchor"/> is already bound to an <see cref="OVRSpatialAnchor"/>.</exception>
		public void BindTo(OVRSpatialAnchor spatialAnchor)
		{
			if (!_space.Valid)
				throw new InvalidOperationException($"{nameof(UnboundAnchor)} does not refer to a valid anchor.");

			if (spatialAnchor == null)
				throw new ArgumentNullException(nameof(spatialAnchor));

			if (spatialAnchor.Created)
				throw new ArgumentException($"Cannot bind {Uuid} to {nameof(spatialAnchor)} because {nameof(spatialAnchor)} is already bound to {spatialAnchor.Uuid}.", nameof(spatialAnchor));

			if (spatialAnchor.PendingCreation)
				throw new ArgumentException($"Cannot bind {Uuid} to {nameof(spatialAnchor)} because {nameof(spatialAnchor)} is being used to create a new spatial anchor.", nameof(spatialAnchor));

			ThrowIfBound(Uuid);

			spatialAnchor.InitializeUnchecked(_space, Uuid);
		}

		internal UnboundAnchor(OVRSpace space, Guid uuid)
		{
			_space = space;
			Uuid = uuid;
		}
	}

	/// <summary>
	/// Performs a query for anchors with the specified <paramref name="options"/>.
	/// </summary>
	/// <remarks>
	/// Use this method to find anchors that were previously persisted with
	/// <see cref="Save(Action{OVRSpatialAnchor, bool}"/>. The query is asynchronous; when the query completes,
	/// <paramref name="onComplete"/> is invoked with an array of <see cref="UnboundAnchor"/>s for which tracking
	/// may be requested.
	/// </remarks>
	/// <param name="options">Options that affect the query.</param>
	/// <param name="onComplete">A delegate invoked when the query completes. The delegate accepts one argument:
	/// - `UnboundAnchor[]`: An array of unbound anchors.
	///
	/// If the operation fails, <paramref name="onComplete"/> is invoked with `null`.</param>
	/// <returns>Returns `true` if the operation could be initiated; otherwise `false`.</returns>
	/// <exception cref="ArgumentNullException">Thrown if <paramref name="onComplete"/> is `null`.</exception>
	/// <exception cref="InvalidOperationException">Thrown if <see cref="LoadOptions.Uuids"/> of <paramref name="options"/> is `null`.</exception>
	public static bool LoadUnboundAnchors(LoadOptions options, Action<UnboundAnchor[]> onComplete)
	{
		if (onComplete == null)
			throw new ArgumentNullException(nameof(onComplete));

		if (options.Uuids == null)
			throw new InvalidOperationException($"{nameof(LoadOptions)}.{nameof(LoadOptions.Uuids)} must not be null.");

		var queried = options.ToQueryOptions().TryQuerySpaces(out var requestId);


		if (queried)
		{
			void Callback(UnboundAnchor[] anchors)
			{
				onComplete.Invoke(anchors);
			}

			Development.LogRequest(requestId, $"{nameof(OVRPlugin.QuerySpaces)}: Query created.");
			Queries[requestId] = Callback;
			return true;
		}


		Development.LogError($"{nameof(OVRPlugin.QuerySpaces)} failed.");
		return false;
	}

	private static void OnSpaceQueryComplete(ulong requestId, bool queryResult)
	{
		Development.LogRequestResult(requestId, queryResult,
			$"{nameof(OVRPlugin.QuerySpaces)}: Query succeeded.",
			$"{nameof(OVRPlugin.QuerySpaces)}: Query failed.");

		if (!TryExtractValue(Queries, requestId, out var callback)) return;

		if (!queryResult)
		{
			callback(null);
			return;
		}

		if (OVRPlugin.RetrieveSpaceQueryResults(requestId, out var results, Allocator.Temp))
		{
			Development.Log($"{nameof(OVRPlugin.RetrieveSpaceQueryResults)}({requestId}): Retrieved {results.Length} results.");
		}
		else
		{
			Development.LogError($"{nameof(OVRPlugin.RetrieveSpaceQueryResults)}({requestId}): Failed to retrieve results.");
			callback(null);
			return;
		}

		using var disposer = results;

		UnboundAnchorBuffer.Clear();
		foreach (var result in results)
		{
			if (SpatialAnchors.ContainsKey(result.uuid))
			{
				Development.Log($"[{result.uuid}] Anchor is already bound to an {nameof(OVRSpatialAnchor)}. Ignoring.");
				continue;
			}

			// See if it supports localization
			if (!OVRPlugin.EnumerateSpaceSupportedComponents(result.space, out var numSupportedComponents,
				    ComponentTypeBuffer))
			{
				Development.LogWarning($"[{result.uuid}] Unable to enumerate supported component types. Ignoring.");
				continue;
			}

			var supportsLocatable = false;
			for (var i = 0; i < numSupportedComponents; i++)
			{
				supportsLocatable |= ComponentTypeBuffer[i] == OVRPlugin.SpaceComponentType.Locatable;
			}

#if DEVELOPMENT_BUILD
            var supportedComponentTypesMsg =
                $"[{result.uuid}] Supports {numSupportedComponents} component type(s): {(numSupportedComponents == 0 ? "(none)" : string.Join(", ", ComponentTypeBuffer.Take((int)numSupportedComponents).Select(c => c.ToString())))}";
#endif

			if (!supportsLocatable)
			{
#if DEVELOPMENT_BUILD
                Development.Log($"{supportedComponentTypesMsg} -- ignoring because it does not support localization.");
#endif
				continue;
			}

#if DEVELOPMENT_BUILD
            Development.Log($"{supportedComponentTypesMsg}.");
#endif

			OVRPlugin.GetSpaceComponentStatus(result.space, OVRPlugin.SpaceComponentType.Locatable, out var enabled, out var changePending);
			Debug.Log($"{result.uuid}: locatable enabled? {enabled} changePending? {changePending}");

			UnboundAnchorBuffer.Add(new UnboundAnchor(result.space, result.uuid));
		}

		Development.Log($"Invoking callback with {UnboundAnchorBuffer.Count} unbound anchor{(UnboundAnchorBuffer.Count == 1 ? "" : "s")}.");
		callback(UnboundAnchorBuffer.Count == 0 ? Array.Empty<UnboundAnchor>() : UnboundAnchorBuffer.ToArray());
	}

	private static void OnSpaceSetComponentStatusComplete(ulong requestId, bool result, OVRSpace space, Guid uuid,
		OVRPlugin.SpaceComponentType componentType, bool enabled)
	{
		Development.LogRequestResult(requestId, result,
			$"[{uuid}] {componentType} {(enabled ? "enabled" : "disabled")}.",
			$"[{uuid}] Failed to set {componentType} status.");

		if (TryExtractValue(LocalizationDelegates, requestId, out var onComplete))
		{
			onComplete(new UnboundAnchor(space, uuid), result);
		}
	}


	private static void OnSpaceListSaveComplete(ulong requestId, OperationResult result)
	{
		Development.LogRequestResult(requestId, result >= 0,
			$"Spaces saved.",
			$"Spaces save failed with error {result}.");

		InvokeMultiAnchorDelegate(requestId, result);
	}

	private static void OnShareSpacesComplete(ulong requestId, OperationResult result)
	{
		Development.LogRequestResult(requestId, result >= 0,
			$"Spaces shared.",
			$"Spaces share failed with error {result}.");

		InvokeMultiAnchorDelegate(requestId, result);
	}

	private static class Development
	{
		[Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
		public static void Log(string message) => Debug.Log($"[{nameof(OVRSpatialAnchor)}] {message}");

		[Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
		public static void LogWarning(string message) => Debug.LogWarning($"[{nameof(OVRSpatialAnchor)}] {message}");

		[Conditional("DEVELOPMENT_BUILD"), Conditional("UNITY_EDITOR")]
		public static void LogError(string message) => Debug.LogError($"[{nameof(OVRSpatialAnchor)}] {message}");

#if DEVELOPMENT_BUILD
        private static readonly HashSet<ulong> _requests = new HashSet<ulong>();
#endif // DEVELOPMENT_BUILD

		[Conditional("DEVELOPMENT_BUILD")]
		public static void LogRequest(ulong requestId, string message)
		{
#if DEVELOPMENT_BUILD
            _requests.Add(requestId);
#endif // DEVELOPMENT_BUILD
			Log($"({requestId}) {message}");
		}

		[Conditional("DEVELOPMENT_BUILD")]
		public static void LogRequestResult(ulong requestId, bool result, string successMessage, string failureMessage)
		{
#if DEVELOPMENT_BUILD
            // Not a request we're tracking
            if (!_requests.Remove(requestId)) return;
#endif // DEVELOPMENT_BUILD
			if (result)
			{
				Log($"({requestId}) {successMessage}");
			}
			else
			{
				LogError($"({requestId}) {failureMessage}");
			}
		}
	}

	/// <summary>
	/// Represents options for saving <see cref="OVRSpatialAnchor"/>.
	/// </summary>
	public struct SaveOptions
	{
		/// <summary>
		/// Location where <see cref="OVRSpatialAnchor"/> will be saved.
		/// </summary>
		public OVRSpace.StorageLocation Storage;
	}

	/// <summary>
	/// Represents options for erasing <see cref="OVRSpatialAnchor"/>.
	/// </summary>
	public struct EraseOptions
	{
		/// <summary>
		/// Location from where <see cref="OVRSpatialAnchor"/> will be erased.
		/// </summary>
		public OVRSpace.StorageLocation Storage;
	}

	public enum OperationResult
	{
		/// Success
		Success = 0,

		/// Failure
		Failure = -1000,

		/// Space error cases
		Failure_SpaceCloudStorageDisabled = -2000,
		Failure_SpaceMappingInsufficient = -2001,
		Failure_SpaceLocalizationFailed = -2002,
		Failure_SpaceNetworkTimeout = -2003,
		Failure_SpaceNetworkRequestFailed = -2004,
	}
}

/// <summary>
/// Represents a user for purposes of sharing scene anchors
/// </summary>
public struct OVRSpaceUser : IDisposable
{
	internal ulong _handle;

	/// <summary>
	/// Checks if the user is valid
	/// </summary>
	public bool Valid => _handle != 0 && Id != 0;

	/// <summary>
	/// Creates a space user handle for given Facebook user id
	/// </summary>
	/// <param name="spaceUserId">The Facebook user id obtained from the other party over the network</param>
	public OVRSpaceUser(ulong spaceUserId)
	{
		OVRPlugin.CreateSpaceUser(spaceUserId, out _handle);
	}

	/// <summary>
	/// The user id associated with this <see cref="OVRSpaceUser"/>.
	/// </summary>
	public ulong Id => OVRPlugin.GetSpaceUserId(_handle, out var userId) ? userId : 0;

	/// <summary>
	/// Disposes of the <see cref="OVRSpaceUser"/>.
	/// </summary>
	/// <remarks>
	/// Note: This does not destroy the user account, just this handle used to reference it.
	/// </remarks>
	public void Dispose()
	{
		OVRPlugin.DestroySpaceUser(_handle);
		_handle = 0;
	}
}
