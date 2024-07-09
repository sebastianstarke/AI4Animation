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
using System.Linq;
using UnityEditor;
using UnityEngine;

internal class OVRProjectSetupDrawer
{
    private class Styles
    {
        private const float SmallIconSize = 16.0f;
        private const float FixButtonWidth = 64.0f;
        private const float FixAllButtonWidth = 80.0f;
        internal const float GroupSelectionWidth = 244.0f;
        internal const float LabelWidth = 96f;
        internal const float TitleLabelWidth = 196f;

        internal readonly GUIStyle Wrap = new GUIStyle(EditorStyles.label)
        {
            wordWrap = true,
            alignment = TextAnchor.MiddleLeft,
            padding = new RectOffset(0, 5, 1, 1)
        };

        internal readonly GUIStyle IssuesBackground = new GUIStyle("ScrollViewAlt")
        {

        };

        internal readonly GUIStyle ListLabel = new GUIStyle("TV Selection")
        {
            border = new RectOffset(0, 0, 0, 0),
            padding = new RectOffset(5, 5, 5, 3),
            margin = new RectOffset(4, 4, 4, 5)
        };

        internal readonly GUIStyle IssuesTitleLabel = new GUIStyle(EditorStyles.label)
        {
            fontSize = 14,
            wordWrap = false,
            stretchWidth = false,
            fontStyle = FontStyle.Bold,
            padding = new RectOffset(10, 10, 0, 0)
        };

        internal readonly GUIStyle FixButton = new GUIStyle(EditorStyles.miniButton)
        {
            margin = new RectOffset(0, 10, 2, 2),
            stretchWidth = false,
            fixedWidth = FixButtonWidth,
        };

        internal readonly GUIStyle FixAllButton = new GUIStyle(EditorStyles.miniButton)
        {
            margin = new RectOffset(0, 10, 2, 2),
            stretchWidth = false,
            fixedWidth = FixAllButtonWidth,
        };

        internal readonly GUIStyle InlinedIconStyle = new GUIStyle(EditorStyles.label)
        {
            margin = new RectOffset(0, 0, 0, 0),
            padding = new RectOffset(0, 0, 0, 0),
            fixedWidth = SmallIconSize,
            fixedHeight = SmallIconSize
        };

        internal readonly GUIStyle IconStyle = new GUIStyle(EditorStyles.label)
        {
            margin = new RectOffset(5, 5, 4, 5),
            padding = new RectOffset(0,0,0,0),
            fixedWidth = SmallIconSize,
            fixedHeight = SmallIconSize
        };

        internal readonly GUIStyle SubtitleHelpText = new GUIStyle(EditorStyles.miniLabel)
        {
            margin = new RectOffset(10,0,0,0),
            wordWrap = true
        };

        internal readonly GUIStyle InternalHelpBox = new GUIStyle(EditorStyles.helpBox)
        {
            margin = new RectOffset(5,5,5,5)
        };

        internal readonly GUIStyle InternalHelpText = new GUIStyle(EditorStyles.miniLabel)
        {
            margin = new RectOffset(10,0,0,0),
            wordWrap = true,
            fontStyle = FontStyle.Italic,
            normal =
            {
                textColor = new Color(0.58f, 0.72f, 0.95f)
            }
        };

        internal readonly GUIStyle NormalStyle = new GUIStyle(EditorStyles.label)
        {
            margin = new RectOffset(10, 0, 0, 0),
            wordWrap = true,
            stretchWidth = false
        };

        internal readonly GUIStyle BoldStyle = new GUIStyle(EditorStyles.label)
        {
            margin = new RectOffset(10, 0, 0, 0),
            stretchWidth = false,
            wordWrap = true,
            fontStyle = FontStyle.Bold
        };

        internal readonly GUIStyle MiniButton = new GUIStyle(EditorStyles.miniButton)
        {
            clipping = TextClipping.Overflow,
            fixedHeight = 18.0f,
            fixedWidth = 18.0f,
            margin = new RectOffset(2,2,2,2),
            padding = new RectOffset(1,1,1,1)
        };

        internal readonly GUIStyle Foldout = new GUIStyle(EditorStyles.foldoutHeader)
        {
            margin = new RectOffset(0, 0, 0, 0),
            padding = new RectOffset(16, 5, 5, 5),
            fixedHeight = 26.0f
        };

        internal readonly GUIStyle FoldoutHorizontal = new GUIStyle(EditorStyles.label)
        {
            fixedHeight = 26.0f
        };

        internal readonly GUIStyle List = new GUIStyle(EditorStyles.helpBox)
        {
            margin = new RectOffset(3,3,3,3),
            padding = new RectOffset(3,3,3,3)
        };
    }

    private static Styles _styles;
    private static Styles styles => _styles ??= new Styles();

    private readonly OVRProjectSetupSettingBool _showOutstandingItems =
        new OVRProjectSetupUserSettingBool("ShowOutstandingItems", "ShowOutstandingItems", true);
    private readonly OVRProjectSetupSettingBool _showRecommendedItems =
        new OVRProjectSetupUserSettingBool("ShowRecommendedItems", "ShowRecommendedItems", true);
    private readonly OVRProjectSetupSettingBool _showVerifiedItems =
        new OVRProjectSetupUserSettingBool("ShowVerifiedItems", "ShowVerifiedItems", false);
    private readonly OVRProjectSetupSettingBool _showIgnoredItems =
        new OVRProjectSetupUserSettingBool("ShowIgnoredItems", "ShowIgnoredItems", false);

    private static readonly GUIContent Title = new GUIContent("Project Setup Tool");
    private static readonly GUIContent Description = new GUIContent("This tool maintains a checklist of required setup tasks as well as best practices to ensure your project is ready to go. Follow our suggestions and fixes to quickly setup your project.");
    private static readonly GUIContent SummaryLabel = new GUIContent("Current project status: ");
    private static readonly GUIContent ListTitle = new GUIContent("Checklist");
    private static readonly GUIContent UnsupportedTitle = new GUIContent("Unsupported Platform");
    private static readonly GUIContent Filter = new GUIContent("Filter by Group :", "Filters the task to the selected group.");
    private static readonly GUIContent FixButtonContent = new GUIContent("Fix", "Fix with recommended settings");
    private static readonly GUIContent FixAllButtonContent = new GUIContent("Fix All", "Fix all the issues from this category");
    private static readonly GUIContent ApplyButtonContent = new GUIContent("Apply", "Apply the recommended settings");
    private static readonly GUIContent ApplyAllButtonContent = new GUIContent("Apply All", "Apply the recommended settings for all the items in this category");
    private static readonly GUIContent WarningIcon = OVRProjectSetupUtils.CreateIcon("ovr_icon_category_warning.png");
    private static readonly GUIContent ErrorIcon = OVRProjectSetupUtils.CreateIcon("ovr_icon_category_error.png");
    private static readonly GUIContent InfoIcon = OVRProjectSetupUtils.CreateIcon("ovr_icon_category_neutral.png");
    private static readonly GUIContent TestPassedIcon = OVRProjectSetupUtils.CreateIcon("ovr_icon_category_success.png");
    private static readonly GUIContent ConfigIcon = OVRProjectSetupUtils.CreateIcon("_Popup", "Additional options", builtIn:true);

    private const string OutstandingItems = "Outstanding Issues";
    private const string RecommendedItems = "Recommended Items";
    private const string VerifiedItems = "Verified Items";
    private const string IgnoredItems = "Ignored Items";


    // Internals
    private OVRConfigurationTask.TaskGroup _selectedTaskGroup;
    private BuildTargetGroup _selectedBuildTargetGroup = BuildTargetGroup.Unknown;
    private Vector2 _scrollViewPos = Vector2.zero;
    private OVRConfigurationTaskUpdaterSummary _lastSummary;

    internal OVRProjectSetupDrawer()
    {
        _selectedTaskGroup = OVRConfigurationTask.TaskGroup.All;
    }

    private class BuildTargetSelectionScope : GUI.Scope
    {
        public BuildTargetGroup BuildTargetGroup { get; protected set; }

        public BuildTargetSelectionScope()
        {
            BuildTargetGroup = EditorGUILayout.BeginBuildTargetSelectionGrouping();
            if (BuildTargetGroup == BuildTargetGroup.Unknown)
            {
                BuildTargetGroup = BuildPipeline.GetBuildTargetGroup(EditorUserBuildSettings.activeBuildTarget);
            }
        }
        protected override void CloseScope() => EditorGUILayout.EndVertical();
    }

    private TEnumType EnumPopup<TEnumType>(GUIContent content, TEnumType currentValue, Action<TEnumType> onChanged)
        where TEnumType : Enum, IComparable
    {
        var previousLabelWidth = EditorGUIUtility.labelWidth;
        EditorGUIUtility.labelWidth = Styles.LabelWidth;
        TEnumType newValue = (TEnumType)EditorGUILayout.EnumPopup(content, currentValue, GUILayout.Width(Styles.GroupSelectionWidth));
        EditorGUIUtility.labelWidth = previousLabelWidth;

        if(!newValue.Equals(currentValue))
        {
            onChanged(newValue);
        }

        return newValue;
    }

    private bool FoldoutWithAdditionalAction(OVRProjectSetupSettingBool key, string label, Rect rect, Action inlineAdditionalAction)
    {
        var previousLabelWidth = EditorGUIUtility.labelWidth;
        EditorGUIUtility.labelWidth = rect.width - 8;

        bool foldout;
        using (new EditorGUILayout.HorizontalScope(styles.FoldoutHorizontal))
        {
            foldout = Foldout(key, label);
            inlineAdditionalAction?.Invoke();
        }

        EditorGUIUtility.labelWidth = previousLabelWidth;
        return foldout;
    }

    private bool Foldout(OVRProjectSetupSettingBool key, string label)
    {
        var currentValue = key.Value;
        var newValue = EditorGUILayout.Foldout(currentValue, label, true, styles.Foldout);
        if (newValue != currentValue)
        {
            key.Value = newValue;
        }

        return newValue;
    }

    private GUIContent GetTaskIcon(OVRConfigurationTask task, BuildTargetGroup buildTargetGroup)
    {
        if (task.IsDone(buildTargetGroup))
        {
            return TestPassedIcon;
        }

        return task.Level.GetValue(buildTargetGroup) switch
        {
            OVRConfigurationTask.TaskLevel.Required => ErrorIcon,
            OVRConfigurationTask.TaskLevel.Recommended => WarningIcon,
            OVRConfigurationTask.TaskLevel.Optional => InfoIcon,
            _ => throw new ArgumentOutOfRangeException()
        };
    }

    private void UpdateTasks(BuildTargetGroup buildTargetGroup)
    {
        OVRProjectSetup.UpdateTasks(buildTargetGroup, logMessages:OVRProjectSetup.LogMessages.Disabled, blocking:false, onCompleted:OnUpdated);
    }

    private void OnUpdated(OVRConfigurationTaskProcessor processor)
    {
	    var updater = processor as OVRConfigurationTaskUpdater;
	    _lastSummary = updater?.Summary;
    }

    private void ShowSettingsMenu()
    {
        var menu = new GenericMenu();
        OVRProjectSetup.Enabled.AppendToMenu(menu);
        OVRProjectSetup.RequiredThrowErrors.AppendToMenu(menu);
        OVRProjectSetup.AllowLogs.AppendToMenu(menu);
        OVRProjectSetup.ShowStatusIcon.AppendToMenu(menu);
        menu.ShowAsContext();
    }

    private void ShowItemMenu(BuildTargetGroup buildTargetGroup, OVRConfigurationTask task)
    {
        var menu = new GenericMenu();
        var hasDocumentation = !string.IsNullOrEmpty(task.URL.GetValue(buildTargetGroup));
        if (hasDocumentation)
        {
            menu.AddItem(new GUIContent("Documentation"), false, OnDocumentation, new object[]{buildTargetGroup, task});
        }

        var hasSourceCode = task.SourceCode.Valid;
        if (hasSourceCode)
        {
            menu.AddItem(new GUIContent("Go to Source Code"), false, OnGoToSourceCode, new object[]{buildTargetGroup, task});
        }

        menu.AddItem(new GUIContent("Ignore"), task.IsIgnored(buildTargetGroup), OnIgnore, new object[]{buildTargetGroup, task});
        menu.ShowAsContext();
    }

    internal void OnTitleBarGUI()
    {
        if(GUILayout.Button(ConfigIcon, styles.MiniButton))
        {
            ShowSettingsMenu();
        }

    }

    internal void OnGUI()
    {
        // Title
        GUILayout.Label(Title, styles.IssuesTitleLabel);

        // Short Description
        GUILayout.Label(Description, styles.SubtitleHelpText);


        EditorGUILayout.Space();

        var enabled = OVRProjectSetup.Enabled.Value;
        using (new EditorGUI.DisabledScope(!enabled))
        {
	        // Summary
	        using (new EditorGUILayout.HorizontalScope())
	        {
		        GUILayout.Label(SummaryLabel, styles.NormalStyle);
		        if (enabled)
		        {
			        GUILayout.Label(OVRProjectSetupStatusIcon.ComputeIcon(_lastSummary), styles.InlinedIconStyle);
			        GUILayout.Label(_lastSummary?.ComputeNoticeMessage() ?? "", styles.BoldStyle);
		        }
		        else
		        {
			        GUILayout.Label("Setup Tool is disabled", styles.BoldStyle);
		        }
	        }

	        // Checklist
	        using (var buildTargetSelection = new BuildTargetSelectionScope())
	        {
		        var buildTargetGroup = buildTargetSelection.BuildTargetGroup;
		        if (_selectedBuildTargetGroup != buildTargetGroup)
		        {
			        _selectedBuildTargetGroup = buildTargetGroup;
			        UpdateTasks(buildTargetGroup);
		        }

		        using (new EditorGUILayout.VerticalScope())
		        {
			        EditorGUILayout.Space();
			        DrawTasksList(_selectedBuildTargetGroup);
		        }
	        }
        }
    }

    private void DrawTasksList(BuildTargetGroup buildTargetGroup)
    {
	    var disableTasksList = EditorApplication.isPlaying;

	    using (new EditorGUI.DisabledGroupScope(disableTasksList))
	    {
		    // Header
		    using (new EditorGUILayout.HorizontalScope())
		    {
			    // Title
			    GUILayout.Label(ListTitle,
				    styles.IssuesTitleLabel, GUILayout.Width(Styles.TitleLabelWidth));

			    GUILayout.FlexibleSpace();

			    // Filter
			    EnumPopup<OVRConfigurationTask.TaskGroup>(Filter, _selectedTaskGroup,
				    group => _selectedTaskGroup = group);
		    }

            // Scroll View
            _scrollViewPos = EditorGUILayout.BeginScrollView(_scrollViewPos, styles.IssuesBackground,
                GUILayout.ExpandHeight(true));

            DrawCategory(_showOutstandingItems, tasks => tasks
                    .Where(task =>
                        (_selectedTaskGroup == OVRConfigurationTask.TaskGroup.All || task.Group == _selectedTaskGroup)
                        && !task.IsDone(buildTargetGroup)
                        && !task.IsIgnored(buildTargetGroup)
                        && (task.Level.GetValue(buildTargetGroup) == OVRConfigurationTask.TaskLevel.Required))
                    .OrderByDescending(task => task.FixAction == null)
                    .ToList(),
                buildTargetGroup, OutstandingItems, true);

            DrawCategory(_showRecommendedItems, tasks => tasks
                    .Where(task =>
                        (_selectedTaskGroup == OVRConfigurationTask.TaskGroup.All || task.Group == _selectedTaskGroup)
                        && !task.IsDone(buildTargetGroup)
                        && !task.IsIgnored(buildTargetGroup)
                        && (task.Level.GetValue(buildTargetGroup) != OVRConfigurationTask.TaskLevel.Required))
                    .OrderByDescending(task => task.Level.GetValue(buildTargetGroup))
                    .ThenBy(task => task.FixAction == null)
                    .ToList(),
                buildTargetGroup, RecommendedItems, true);

            DrawCategory(_showVerifiedItems, tasks => tasks
                    .Where(task =>
                        (_selectedTaskGroup == OVRConfigurationTask.TaskGroup.All || task.Group == _selectedTaskGroup)
                        && task.IsDone(buildTargetGroup)
                        && !task.IsIgnored(buildTargetGroup))
                    .OrderByDescending(task => task.FixAction == null)
                    .ThenBy(task => task.Level.GetValue(buildTargetGroup))
                    .ToList(),
                buildTargetGroup, VerifiedItems, false);

            DrawCategory(_showIgnoredItems, tasks => tasks
                    .Where(task =>
                        (_selectedTaskGroup == OVRConfigurationTask.TaskGroup.All || task.Group == _selectedTaskGroup)
                        && task.IsIgnored(buildTargetGroup))
                    .OrderByDescending(task => task.Level.GetValue(buildTargetGroup))
                    .ThenBy(task => task.FixAction != null)
                    .ToList(),
                buildTargetGroup, IgnoredItems, false);

            EditorGUILayout.EndScrollView();
        }
    }

    private void DrawCategory(OVRProjectSetupSettingBool key, Func<IEnumerable<OVRConfigurationTask>, List<OVRConfigurationTask>> filter, BuildTargetGroup buildTargetGroup, string title, bool fixAllButton)
    {
        var tasks = filter(OVRProjectSetup.GetTasks(buildTargetGroup, false));

        if (key == null || tasks == null || tasks.Count == 0)
        {
            return;
        }

        using (var scope = new EditorGUILayout.VerticalScope(styles.List))
        {
            var rect = scope.rect;

            // Foldout
            title = $"{title} ({tasks.Count})";

            var foldout = FoldoutWithAdditionalAction(key, title, rect, () =>
            {
                if (fixAllButton)
                {
	                if (tasks.Any(task => task.FixAction != null))
	                {
		                var content = tasks[0].Level.GetValue(buildTargetGroup) == OVRConfigurationTask.TaskLevel.Required
			                ? FixAllButtonContent
			                : ApplyAllButtonContent;
		                EditorGUI.BeginDisabledGroup(OVRProjectSetup.ProcessorQueue.BusyWith(OVRConfigurationTaskProcessor.ProcessorType.Fixer));
		                if (GUILayout.Button(content, styles.FixAllButton))
		                {
			                OVRProjectSetup.FixTasks(buildTargetGroup, filter, blocking:false, onCompleted:AfterFixApply);
		                }
		                EditorGUI.EndDisabledGroup();
	                }
                }
            });

            if (foldout)
            {
                DrawIssues(tasks, buildTargetGroup);
            }
        }
    }

    private void AfterFixApply(OVRConfigurationTaskProcessor processor)
    {
        AssetDatabase.SaveAssets();
        UpdateTasks(processor.BuildTargetGroup);
    }

    private void DrawIssues(List<OVRConfigurationTask> tasks, BuildTargetGroup buildTargetGroup)
    {
        foreach (var task in tasks)
        {
            DrawIssue(task, buildTargetGroup);
        }
    }

    private void DrawIssue(OVRConfigurationTask task, BuildTargetGroup buildTargetGroup)
    {
        var ignored = task.IsIgnored(buildTargetGroup);
        var cannotBeFixed = task.IsDone(buildTargetGroup) || OVRProjectSetup.ProcessorQueue.BusyWith(OVRConfigurationTaskProcessor.ProcessorType.Fixer);
        var disabled = cannotBeFixed || ignored;

        // Note : We're not using scopes, because in this very case, we've got a cross of scopes
        EditorGUI.BeginDisabledGroup(disabled);
        var clickArea = EditorGUILayout.BeginHorizontal(styles.ListLabel);

        // Icon
        GUILayout.Label(GetTaskIcon(task, buildTargetGroup), styles.IconStyle);

        // Message
        GUILayout.Label(new GUIContent(task.Message.GetValue(buildTargetGroup)), styles.Wrap);

        EditorGUI.EndDisabledGroup();

        if (task.FixAction != null)
        {
	        EditorGUI.BeginDisabledGroup(cannotBeFixed);
	        var content = task.Level.GetValue(buildTargetGroup) == OVRConfigurationTask.TaskLevel.Required
		        ? FixButtonContent
		        : ApplyButtonContent;

            var fixMessage = task.FixMessage.GetValue(buildTargetGroup);
            var tooltip = fixMessage != null ? $"{content.tooltip} :\n{fixMessage}" : content.tooltip;
            content = new GUIContent(content.text, tooltip);
	        if (GUILayout.Button(content, styles.FixButton))
	        {
		        OVRProjectSetup.FixTask(buildTargetGroup, task, blocking: false, onCompleted: AfterFixApply);
	        }

	        EditorGUI.EndDisabledGroup();
        }

        var current = Event.current;
        if(GUILayout.Button("", EditorStyles.foldoutHeaderIcon, GUILayout.Width(16.0f))
           || (clickArea.Contains(current.mousePosition) && current.type == EventType.ContextClick))
        {
            ShowItemMenu(buildTargetGroup, task);
            if (current.type == EventType.ContextClick)
            {
                current.Use();
            }
        }

        EditorGUILayout.EndHorizontal();
    }

    private void ReadContextMenuArguments(
        object arg,
        out BuildTargetGroup buildTargetGroup,
        out OVRConfigurationTask task)
    {
        var args = arg as object[];
        buildTargetGroup = args != null ? (BuildTargetGroup)args[0] : BuildTargetGroup.Unknown;
        task = args?[1] as OVRConfigurationTask;
    }

    private void OnIgnore(object args)
    {
        ReadContextMenuArguments(args, out var buildTargetGroup, out var task);
        task?.SetIgnored(buildTargetGroup, !task.IsIgnored(buildTargetGroup));
    }

    private void OnDocumentation(object args)
    {
        ReadContextMenuArguments(args, out var buildTargetGroup, out var task);
        var url = task?.URL.GetValue(buildTargetGroup);

        Application.OpenURL(url);
    }

    private void OnGoToSourceCode(object args)
    {
        ReadContextMenuArguments(args, out var buildTargetGroup, out var task);
        task?.SourceCode.Open();
    }
}
