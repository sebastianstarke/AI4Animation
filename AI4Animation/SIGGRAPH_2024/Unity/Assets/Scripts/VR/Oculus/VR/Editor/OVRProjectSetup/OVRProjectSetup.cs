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

using System.Collections.Generic;
using UnityEditor;
using System;
using System.Linq;

/// <summary>
/// Core System for the OVRProjectSetup Tool
/// </summary>
/// <remarks>
/// This static class manages <see cref="OVRConfigurationTask"/> that can be added at any point.
/// Use the AddTask method to add and register <see cref="OVRConfigurationTask"/>.
/// </remarks>
internal static class OVRProjectSetup
{
	internal static readonly OVRConfigurationTaskRegistry _principalRegistry;

    internal static OVRConfigurationTaskRegistry Registry { get; private set; }
    internal static OVRConfigurationTaskProcessorQueue ProcessorQueue { get; }

    internal const string KeyPrefix = "OVRProjectSetup";
    public static OVRProjectSetupSettingBool Enabled;
    public static OVRProjectSetupSettingBool RequiredThrowErrors;
    public static readonly OVRProjectSetupSettingBool AllowLogs = new OVRProjectSetupProjectSettingBool("AllowLogs", "Log outstanding issues", false);
    public static readonly OVRProjectSetupSettingBool ShowStatusIcon = new OVRProjectSetupProjectSettingBool("ShowStatusIcon", "Show Status Icon", true);

    private static readonly HashSet<BuildTargetGroup> SupportedPlatforms = new HashSet<BuildTargetGroup>{BuildTargetGroup.Android, BuildTargetGroup.Standalone};

    static OVRProjectSetup()
    {
	    _principalRegistry = new OVRConfigurationTaskRegistry();
        ProcessorQueue = new OVRConfigurationTaskProcessorQueue();
        ConsoleLinkEventHandler.OnConsoleLink += OnConsoleLink;
        RestoreRegistry();
    }

    internal static void SetupTemporaryRegistry()
    {
	    Registry = new OVRConfigurationTaskRegistry();
	    Enabled = new OVRProjectSetupConstSettingBool("Enabled", "Enabled", true);
	    RequiredThrowErrors = new OVRProjectSetupConstSettingBool("RequiredThrowErrors", "Required throw errors", false);
    }

    internal static void RestoreRegistry()
    {
	    Registry = _principalRegistry;
	    Enabled =
// An additional OVR_INTERNAL_CODE check that should be kept in as this setting item is not meant for public
	    new OVRProjectSetupConstSettingBool("Enabled", "Enabled", true);
	    RequiredThrowErrors = new OVRProjectSetupProjectSettingBool("RequiredThrowErrors", "Required throw errors", false);
    }

    private static void OnConsoleLink(Dictionary<string, string> infos)
    {
        if (infos.TryGetValue("href", out var href))
        {
            if (href == OVRConfigurationTask.ConsoleLinkHref)
            {
                OVRProjectSetupSettingsProvider.OpenSettingsWindow();
            }
        }
    }

    internal static IEnumerable<OVRConfigurationTask> GetTasks(BuildTargetGroup buildTargetGroup, bool refresh)
    {
        return Registry.GetTasks(buildTargetGroup, refresh);
    }

    /// <summary>
    /// Add an <see cref="OVRConfigurationTask"/> to the Setup Tool.
    /// </summary>
    /// <remarks>
    /// This methods adds and registers an already created <see cref="OVRConfigurationTask"/> to the SetupTool.
    /// We recommend the use of the other AddTask method with all the required parameters to create the task.
    /// </remarks>
    /// <param name="task">The task that will get registered to the Setup Tool.</param>
    /// <exception cref="ArgumentException">Possible causes :
    /// - a task with the same unique ID already has been registered (conflict in hash generated from description message).</exception>
    public static void AddTask(OVRConfigurationTask task)
    {
        Registry.AddTask(task);
    }

    /// <summary>
    /// Add an <see cref="OVRConfigurationTask"/> to the Setup Tool.
    /// </summary>
    /// <remarks>
    /// This methods creates, adds and registers an <see cref="OVRConfigurationTask"/> to the SetupTool.
    /// Please note that the Message or ConditionalMessage parameters are used to generated a unique hash that serves as an Unique ID for the task.
    /// Those tasks, once added, are not meant to be removed from the Setup Tool, and will get checked at some key points.
    /// This method is the one entry point for developers to add their own sanity checks, technical requirements or other recommendations.
    /// You can use the conditional parameters that accepts lambdas or delegates for more complex behaviours if needed.
    /// </remarks>
    /// <param name="group">Category that fits the task. Feel free to add more to the enum if relevant. Do not use "All".</param>
    /// <param name="isDone">Func/Delegates/Lambda that checks if the Configuration Task is validated or not.</param>
    /// <param name="fix">Action/Delegates/Lambda that actually validate the Configuration Task.</param>
    /// <param name="platform">Platform for which this Configuration Task applies. Use "Unknown" for any.</param>
    /// <param name="level">Level/Severity/Priority/Behaviour of the Configuration Task</param>
    /// <param name="conditionalLevel">Use this delegate for more control or complex behaviours over the level parameter.</param>
    /// <param name="message">Description of the Configuration Task</param>
    /// <param name="conditionalMessage">Use this delegate for more control or complex behaviours over the message parameter.</param>
    /// /// <param name="fixMessage">Description of the actual fix for the Task</param>
    /// <param name="conditionalFixMessage">Use this delegate for more control or complex behaviours over the fixMessage parameter.</param>
    /// <param name="url">Url to more information about the Configuration Task</param>
    /// <param name="conditionalUrl">Use this delegate for more control or complex behaviours over the url parameter.</param>
    /// <param name="validity">Checks if the task is valid. If not, it will be ignored by the Setup Tool.</param>
    /// <param name="conditionalValidity">Use this delegate for more control or complex behaviours over the validity parameter.</param>
    /// <exception cref="ArgumentNullException">Possible causes :
    /// - If either message or conditionalMessage do not provide a valid non null string
    /// - isDone is null
    /// - fix is null</exception>
    /// <exception cref="ArgumentException">Possible causes :
    /// - group is set to "All". This category is not meant to be used to describe a task.
    /// - a task with the same unique ID already has been registered (conflict in hash generated from description message).</exception>
    public static void AddTask(
        OVRConfigurationTask.TaskGroup group,
        Func<BuildTargetGroup, bool> isDone,
        BuildTargetGroup platform = BuildTargetGroup.Unknown,
        Action<BuildTargetGroup> fix = null,
        OVRConfigurationTask.TaskLevel level = OVRConfigurationTask.TaskLevel.Recommended,
        Func<BuildTargetGroup, OVRConfigurationTask.TaskLevel> conditionalLevel = null,
        string message = null,
        Func<BuildTargetGroup, string> conditionalMessage = null,
        string fixMessage = null,
        Func<BuildTargetGroup, string> conditionalFixMessage = null,
        string url = null,
        Func<BuildTargetGroup, string> conditionalUrl = null,
        bool validity = true,
        Func<BuildTargetGroup, bool> conditionalValidity = null
        )
    {
        var optionalLevel = OptionalLambdaType<BuildTargetGroup, OVRConfigurationTask.TaskLevel>.Create(level, conditionalLevel, true);
        var optionalMessage = OptionalLambdaType<BuildTargetGroup, string>.Create(message, conditionalMessage, true);
        var optionalFixMessage = OptionalLambdaType<BuildTargetGroup, string>.Create(fixMessage, conditionalFixMessage, true);
        var optionalUrl = OptionalLambdaType<BuildTargetGroup, string>.Create(url, conditionalUrl, true);
        var optionalValidity = OptionalLambdaType<BuildTargetGroup, bool>.Create(validity, conditionalValidity, true);
        AddTask(new OVRConfigurationTask(group, platform, isDone, fix, optionalLevel, optionalMessage, optionalFixMessage, optionalUrl, optionalValidity));
    }

    public static bool IsPlatformSupported(BuildTargetGroup buildTargetGroup)
    {
        return SupportedPlatforms.Contains(buildTargetGroup);
    }

    internal enum LogMessages
    {
        Disabled = 0,
        Summary = 1,
        Changed = 2,
        All = 3,
    }

    private const int LoopExitCount = 4;

    internal static void FixTasks(
	    BuildTargetGroup buildTargetGroup,
        Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter = null,
	    LogMessages logMessages = LogMessages.Disabled,
        bool blocking = true,
	    Action<OVRConfigurationTaskProcessor> onCompleted = null)
    {
	    var fixer = new OVRConfigurationTaskFixer(Registry, buildTargetGroup, filter, logMessages, blocking, onCompleted);
	    ProcessorQueue.Request(fixer);
    }

    internal static void FixTask(
	    BuildTargetGroup buildTargetGroup,
	    OVRConfigurationTask task,
	    LogMessages logMessages = LogMessages.Disabled,
	    bool blocking = true,
	    Action<OVRConfigurationTaskProcessor> onCompleted = null
    )
    {
	    // TODO : A bit overkill for just one task
	    var filter = (Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> )(tasks => tasks.Where(otherTask => otherTask == task).ToList());
	    var fixer = new OVRConfigurationTaskFixer(Registry, buildTargetGroup, filter, logMessages, blocking, onCompleted);
	    ProcessorQueue.Request(fixer);
    }

    internal static void UpdateTasks(
	    BuildTargetGroup buildTargetGroup,
	    Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter = null,
	    LogMessages logMessages = LogMessages.Disabled,
	    bool blocking = true,
	    Action<OVRConfigurationTaskProcessor> onCompleted = null)
    {
	    var updater = new OVRConfigurationTaskUpdater(Registry, buildTargetGroup, filter, logMessages, blocking, onCompleted);
	    ProcessorQueue.Request(updater);
    }
}
