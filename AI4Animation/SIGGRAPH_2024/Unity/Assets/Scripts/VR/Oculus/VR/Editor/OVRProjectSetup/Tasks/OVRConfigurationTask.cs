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
using UnityEditor;
using UnityEngine;

internal class OVRConfigurationTask
{
    internal static readonly string ConsoleLinkHref = "OpenOculusProjectSettings";

    public enum TaskLevel
    {
        Optional = 0,
        Recommended = 1,
        Required = 2
    }

    public enum TaskGroup
    {
        All = 0,
        Compatibility = 1,
        Rendering = 2,
        Quality = 3,
        Physics = 4,
        Packages = 5,
        Features = 6
    }

    public Hash128 Uid { get; }
    public TaskGroup Group { get; }
    public BuildTargetGroup Platform { get; }

    public OptionalLambdaType<BuildTargetGroup, bool> Valid { get; }
    public OptionalLambdaType<BuildTargetGroup, TaskLevel> Level { get; }
    public OptionalLambdaType<BuildTargetGroup, string> Message { get; }
    public OptionalLambdaType<BuildTargetGroup, string> FixMessage { get; }
    public OptionalLambdaType<BuildTargetGroup, string> URL { get; }
    public OVRConfigurationTaskSourceCode SourceCode { get; set; }

    private Func<BuildTargetGroup, bool> _isDone;
    public Func<BuildTargetGroup, bool> IsDone
    {
        get => GetDoneState;
        private set => _isDone = value;
    }
    public Action<BuildTargetGroup> FixAction { get; }

    private readonly Dictionary<BuildTargetGroup, OVRProjectSetupSettingBool> _ignoreSettings =
        new Dictionary<BuildTargetGroup, OVRProjectSetupSettingBool>();

    private readonly Dictionary<BuildTargetGroup, bool> _isDoneCache = new Dictionary<BuildTargetGroup, bool>();

    public OVRConfigurationTask(
        TaskGroup group,
        BuildTargetGroup platform,
        Func<BuildTargetGroup, bool> isDone,
        Action<BuildTargetGroup> fix,
        OptionalLambdaType<BuildTargetGroup, TaskLevel> level,
        OptionalLambdaType<BuildTargetGroup, string> message,
        OptionalLambdaType<BuildTargetGroup, string> fixMessage,
        OptionalLambdaType<BuildTargetGroup, string> url,
        OptionalLambdaType<BuildTargetGroup, bool> valid)
    {
	    Platform = platform;
	    Group = group;
	    IsDone = isDone;
        FixAction = fix;
	    Level = level;
	    Message = message;

	    // If parameters are null, we're creating a OptionalLambdaType that points to default values
	    // We don't want a null OptionalLambdaType, but we may be okay with an OptionalLambdaType containing a null value
	    // For the URL for instance
	    // Mandatory parameters will be checked on the Validate method down below
	    URL = url ?? new OptionalLambdaTypeWithoutLambda<BuildTargetGroup, string>(null);
        FixMessage = fixMessage ?? new OptionalLambdaTypeWithoutLambda<BuildTargetGroup, string>(null);
	    Valid = valid ?? new OptionalLambdaTypeWithoutLambda<BuildTargetGroup, bool>(true);

	    // We may want to throw in case of some invalid parameters
	    Validate();

        var hash = new Hash128();
        hash.Append(Message.Default);
        Uid = hash;

        SourceCode = new OVRConfigurationTaskSourceCode(this);
    }

    private void Validate()
    {
	    if (Group == TaskGroup.All)
	    {
		    throw new ArgumentException(
			    $"[{nameof(OVRConfigurationTask)}] {nameof(TaskGroup.All)} is not meant to be used as a {nameof(TaskGroup)} type");
	    }

	    if (_isDone == null)
	    {
		    throw new ArgumentNullException(nameof(_isDone));
	    }


	    if (Level == null)
	    {
		    throw new ArgumentNullException(nameof(Level));
	    }


	    if (Message == null || !Message.Valid || string.IsNullOrEmpty(Message.Default))
	    {
		    throw new ArgumentNullException(nameof(Message));
	    }
    }

    public void InvalidateCache(BuildTargetGroup buildTargetGroup)
    {
	    Level.InvalidateCache(buildTargetGroup);
	    Message.InvalidateCache(buildTargetGroup);
	    URL.InvalidateCache(buildTargetGroup);
	    Valid.InvalidateCache(buildTargetGroup);
    }

    public bool IsIgnored(BuildTargetGroup buildTargetGroup)
    {
        return GetIgnoreSetting(buildTargetGroup).Value;
    }

    public void SetIgnored(BuildTargetGroup buildTargetGroup, bool ignored)
    {
        GetIgnoreSetting(buildTargetGroup).Value = ignored;
    }

    public bool Fix(BuildTargetGroup buildTargetGroup)
    {
	    try
	    {
		    FixAction(buildTargetGroup);
	    }
	    catch (OVRConfigurationTaskException exception)
	    {
		    Debug.LogWarning(
			    $"[Oculus Settings] Failed to fix task \"{Message.GetValue(buildTargetGroup)}\" : {exception}");
	    }

	    var hasChanged = UpdateAndGetStateChanged(buildTargetGroup);
	    if (hasChanged)
	    {
		    var fixMessage = FixMessage.GetValue(buildTargetGroup);
		    Debug.Log(
			    fixMessage != null
				    ? $"[Oculus Settings] Fixed task \"{Message.GetValue(buildTargetGroup)}\" : {fixMessage}"
				    : $"[Oculus Settings] Fixed task \"{Message.GetValue(buildTargetGroup)}\"");
	    }

	    var isDone = IsDone(buildTargetGroup);
	    return isDone;
    }

    private OVRProjectSetupSettingBool GetIgnoreSetting(BuildTargetGroup buildTargetGroup)
    {
        if (!_ignoreSettings.TryGetValue(buildTargetGroup, out var item))
        {
            var key = $"{OVRProjectSetup.KeyPrefix}.{GetType().Name}.{Uid}.Ignored.{buildTargetGroup.ToString()}";
            item = new OVRProjectSetupProjectSettingBool(key, "", false);
            _ignoreSettings.Add(buildTargetGroup, item);
        }

        return item;
    }

    internal bool UpdateAndGetStateChanged(BuildTargetGroup buildTargetGroup)
    {
        var newState = _isDone(buildTargetGroup);
        var didStateChange = true;
        if (_isDoneCache.TryGetValue(buildTargetGroup, out var previousState))
        {
            didStateChange = newState != previousState;
        }

        _isDoneCache[buildTargetGroup] = newState;

        return didStateChange;
    }

    internal void LogMessage(BuildTargetGroup buildTargetGroup)
    {
        var logMessage = GetFullLogMessage(buildTargetGroup);

        switch (Level.GetValue(buildTargetGroup))
        {
            case TaskLevel.Optional:
                break;
            case TaskLevel.Recommended:
                Debug.LogWarning(logMessage);
                break;
            case TaskLevel.Required:
                if (OVRProjectSetup.RequiredThrowErrors.Value)
                {
                    Debug.LogError(logMessage);
                }
                else
                {
                    Debug.LogWarning(logMessage);
                }
                break;
            default:
                throw new ArgumentOutOfRangeException();
        }
    }

    internal string GetFullLogMessage(BuildTargetGroup buildTargetGroup)
    {
        return $"{GetLogMessage(buildTargetGroup)}.\nYou can fix this by going to <a href=\"{ConsoleLinkHref}\">Edit > Project Settings > {OVRProjectSetupSettingsProvider.SettingsName}</a>";
    }

    internal string GetLogMessage(BuildTargetGroup buildTargetGroup)
    {
        return $"[{Group}] {Message.GetValue(buildTargetGroup)}";
    }

    private bool GetDoneState(BuildTargetGroup buildTargetGroup)
    {
        if (_isDoneCache.TryGetValue(buildTargetGroup, out var cachedState))
        {
            return cachedState;
        }

        return _isDone(buildTargetGroup);
    }
}
