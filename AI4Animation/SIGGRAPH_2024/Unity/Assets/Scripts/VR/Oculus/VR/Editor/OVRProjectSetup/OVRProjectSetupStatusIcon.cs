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
using System.Reflection;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

[InitializeOnLoad]
internal static class OVRProjectSetupStatusIcon
{
    private static readonly Type _toolbarType;
    private static readonly PropertyInfo _guiBackend;
    private static readonly PropertyInfo _visualTree;
    private static readonly FieldInfo _onGuiHandler;
    private static readonly GUIContent _iconSuccess;
    private static readonly GUIContent _iconNeutral;
    private static readonly GUIContent _iconWarning;
    private static readonly GUIContent _iconError;
    private static readonly string OpenOculusSettings = "Open Oculus Settings";

    private static GUIStyle _iconStyle;
    private static VisualElement _container;
    private static GUIContent _currentIcon;

    internal static GUIContent CurrentIcon => _currentIcon;



    static OVRProjectSetupStatusIcon()
    {
        var editorAssembly = typeof(UnityEditor.Editor).Assembly;
        var bindingFlags = BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance;

        _toolbarType = editorAssembly.GetType("UnityEditor.AppStatusBar");
        var guiViewType = editorAssembly.GetType("UnityEditor.GUIView");
        var backendType = editorAssembly.GetType("UnityEditor.IWindowBackend");
        var containerType = typeof(IMGUIContainer);

        _guiBackend = guiViewType?.GetProperty("windowBackend", bindingFlags);
        _visualTree = backendType?.GetProperty("visualTree", bindingFlags);
        _onGuiHandler = containerType?.GetField("m_OnGUIHandler", bindingFlags);

        _iconSuccess = OVRProjectSetupUtils.CreateIcon("ovr_icon_success.png", null);
        _iconNeutral = OVRProjectSetupUtils.CreateIcon("ovr_icon_neutral.png", null);
        _iconWarning = OVRProjectSetupUtils.CreateIcon("ovr_icon_warning.png", null);
        _iconError = OVRProjectSetupUtils.CreateIcon("ovr_icon_error.png", null);
        _currentIcon = _iconSuccess;

        OVRProjectSetup.ProcessorQueue.OnProcessorCompleted += RefreshData;
        EditorApplication.update += RefreshContainer;
    }

    private static void RefreshContainer()
    {
        if (_container != null)
        {
            return;
        }
        var toolbars = Resources.FindObjectsOfTypeAll(_toolbarType);
        if (toolbars == null || toolbars.Length == 0)
        {
            return;
        }

        var toolbar = toolbars[0];
        if (toolbar == null)
        {
            return;
        }

        var backend = _guiBackend?.GetValue(toolbar);
        if (backend == null)
        {
            return;
        }

        var elements = _visualTree?.GetValue(backend, null) as VisualElement;
        _container = elements?[0];
        if (_container == null)
        {
            return;
        }

        var handler = _onGuiHandler?.GetValue(_container) as Action;
        if (handler == null)
        {
            return;
        }

        handler -= RefreshGUI;
        handler += RefreshGUI;
        _onGuiHandler.SetValue(_container, handler);

        EditorApplication.update -= RefreshContainer;
    }

    private static void RefreshStyles()
    {
        if (_iconStyle != null)
        {
            return;
        }

        _iconStyle = new GUIStyle("StatusBarIcon");
    }

    public static GUIContent ComputeIcon(OVRConfigurationTaskUpdaterSummary summary)
    {
	    if (summary == null)
	    {
		    return _iconSuccess;
	    }

	    var icon = summary.HighestFixLevel switch
	    {
		    OVRConfigurationTask.TaskLevel.Optional => _iconNeutral,
		    OVRConfigurationTask.TaskLevel.Recommended => _iconWarning,
		    OVRConfigurationTask.TaskLevel.Required => _iconError,
		    _ => _iconSuccess
	    };

	    icon.tooltip = $"{summary.ComputeNoticeMessage()}\n{OpenOculusSettings}";

	    return icon;
    }

    private static void RefreshData(OVRConfigurationTaskProcessor processor)
    {
	    var activeBuildTargetGroup = BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget);
	    if (processor.Type == OVRConfigurationTaskProcessor.ProcessorType.Updater
	        && processor.BuildTargetGroup == activeBuildTargetGroup)
	    {
		    var updater = processor as OVRConfigurationTaskUpdater;
		    _currentIcon = ComputeIcon(updater?.Summary);
	    }
    }

    private static void RefreshGUI()
    {
        if (!OVRProjectSetup.ShowStatusIcon.Value)
        {
            return;
        }

        RefreshStyles();

        var screenWidth = EditorGUIUtility.currentViewWidth;
        // Hardcoded position
        // Currently overlaps with progress bar, and works with 2020 status bar icons
        // TODO: Better hook to dynamically position the button
        var currentRect = new Rect(screenWidth - 130, 0, 26, 30); // Hardcoded position
        GUILayout.BeginArea(currentRect);
        if (GUILayout.Button(_currentIcon, _iconStyle))
        {
            OVRProjectSetupSettingsProvider.OpenSettingsWindow();
        }
        var buttonRect = GUILayoutUtility.GetLastRect();
        EditorGUIUtility.AddCursorRect(buttonRect, MouseCursor.Link);
        GUILayout.EndArea();
    }
}
