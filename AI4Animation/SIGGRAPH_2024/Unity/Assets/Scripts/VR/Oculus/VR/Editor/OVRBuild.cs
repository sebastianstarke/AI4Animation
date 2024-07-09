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

#if USING_XR_MANAGEMENT && (USING_XR_SDK_OCULUS || USING_XR_SDK_OPENXR)
#define USING_XR_SDK
#endif

#if UNITY_2020_1_OR_NEWER
#define REQUIRES_XR_SDK
#endif

// Unit made a change that broke `BuildOptions.AcceptExternalModificationsToPlayer`
// Thread detailing issue: https://forum.unity.com/threads/oculus-build-invalidoperationexception-the-build-target-does-not-support-build-appending.994930/
#if UNITY_2020_1_OR_NEWER || UNITY_2019_4_OR_NEWER
#define DONT_USE_BUILD_OPTIONS_EXTERNAL_MODIFICATIONS_FLAG
#endif

using UnityEngine;
using UnityEditor;
using System;
using System.IO;
using System.Diagnostics;
using System.Collections.Generic;
using System.Threading;

/// <summary>
/// Allows Oculus to build apps from the command line.
/// </summary>
[InitializeOnLoadAttribute]
partial class OculusBuildApp : EditorWindow
{
    static void SetPCTarget()
    {
        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.StandaloneWindows)
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Standalone, BuildTarget.StandaloneWindows);
        }
#if !USING_XR_SDK && !REQUIRES_XR_SDK
        UnityEditorInternal.VR.VREditor.SetVREnabledOnTargetGroup(BuildTargetGroup.Standalone, true);
#pragma warning disable 618
        PlayerSettings.virtualRealitySupported = true;
#pragma warning restore 618
#endif
        AssetDatabase.SaveAssets();
    }

    static void SetAndroidTarget()
    {
        EditorUserBuildSettings.androidBuildSubtarget = MobileTextureSubtarget.ASTC;
        EditorUserBuildSettings.androidBuildSystem = AndroidBuildSystem.Gradle;

        if (EditorUserBuildSettings.activeBuildTarget != BuildTarget.Android)
        {
            EditorUserBuildSettings.SwitchActiveBuildTarget(BuildTargetGroup.Android, BuildTarget.Android);
        }

#if !USING_XR_SDK && !REQUIRES_XR_SDK
        UnityEditorInternal.VR.VREditor.SetVREnabledOnTargetGroup(BuildTargetGroup.Standalone, true);
#pragma warning disable 618
        PlayerSettings.virtualRealitySupported = true;
#pragma warning restore 618
#endif
        AssetDatabase.SaveAssets();
    }

#if UNITY_EDITOR_WIN && UNITY_ANDROID
    // Build setting constants
    const string REMOTE_APK_PATH = "/data/local/tmp";
    const float USB_TRANSFER_SPEED_THRES = 25.0f;
    const float USB_3_TRANSFER_SPEED = 32.0f;
    const float TRANSFER_SPEED_CHECK_THRESHOLD = 4.0f;
    const int NUM_BUILD_AND_RUN_STEPS = 9;
    const int BYTES_TO_MEGABYTES = 1048576;

    // Build window variables (not saved)
    GUIStyle windowStyle;
    GUIStyle calloutStyle;
    string currConnectedDevice;
    Vector2 scrollViewPos;

    // Build window variables (saved)
    static bool saveKeystorePasswords;
    static bool isRunOnDevice;
    static string outputApkPath;

    // Progress bar variables
    static int totalBuildSteps;
    static int currentStep;
    static string progressMessage;

    // Build setting varaibles
    static string gradlePath;
    static string jdkPath;
    static string androidSdkPath;
    static string applicationIdentifier;
    static bool isDevelopmentBuild;
    static string productName;
    static string dataPath;

    static string gradleTempExport;
    static string gradleExport;
    static bool showCancel;
    static bool buildInProgress;

    static DirectorySyncer.CancellationTokenSource syncCancelToken;
    static Process gradleBuildProcess;
    static Thread buildThread;

    static bool? apkOutputSuccessful;

    [MenuItem("Oculus/OVR Build/OVR Build APK... %#k", false, 20)]
    static void Init()
    {
        EditorWindow.GetWindow<OculusBuildApp>(false, "OVR Build APK", true);
        OnBuildComplete();
    }

    [MenuItem("Oculus/OVR Build/OVR Build APK And Run %k", false, 21)]
    static void InitAndRun()
    {
        var window = EditorWindow.GetWindow<OculusBuildApp>(false, "OVR Build APK", true);
        isRunOnDevice = true; // make sure the "And Run" part of the menu name is true
        window.StartBuild();
    }

    private void OnEnable()
    {
        isRunOnDevice = EditorPrefs.GetBool("OVRBuild_RunOnDevice", false);
        saveKeystorePasswords = EditorPrefs.GetBool("OVRBuild_SaveKeystorePasswords", false);
        if (saveKeystorePasswords)
        {
            if (EditorPrefs.HasKey("OVRBuild_KeystorePassword"))
                PlayerSettings.Android.keystorePass = EditorPrefs.GetString("OVRBuild_KeystorePassword");
            if (EditorPrefs.HasKey("OVRBuild_KeyAliasPassword"))
                PlayerSettings.Android.keyaliasPass = EditorPrefs.GetString("OVRBuild_KeyAliasPassword");
        }
        outputApkPath = EditorPrefs.GetString("OVRBuild_BuiltAPKPath", "");

        CheckADBDevices(out currConnectedDevice);
    }

    private void OnDisable()
    {
        EditorPrefs.SetBool("OVRBuild_RunOnDevice", isRunOnDevice);
        EditorPrefs.SetBool("OVRBuild_SaveKeystorePasswords", saveKeystorePasswords);
        if (saveKeystorePasswords)
        {
            EditorPrefs.SetString("OVRBuild_KeystorePassword", PlayerSettings.Android.keystorePass);
            EditorPrefs.SetString("OVRBuild_KeyAliasPassword", PlayerSettings.Android.keyaliasPass);
        }
        else
        {
            EditorPrefs.DeleteKey("OVRBuild_KeystorePassword");
            EditorPrefs.DeleteKey("OVRBuild_KeyAliasPassword");
        }
        EditorPrefs.SetString("OVRBuild_BuiltAPKPath", outputApkPath);
    }

    private void OnGUI()
    {
        if (windowStyle == null)
        {
            windowStyle = new GUIStyle();
            windowStyle.margin = new RectOffset(10, 10, 10, 10);
        }

        if (calloutStyle == null)
        {
            calloutStyle = new GUIStyle(EditorStyles.label);
            calloutStyle.richText = true;
            calloutStyle.wordWrap = true;
        }

        // Fix progress bar window size
        minSize = new Vector2(500, 305);

        float oldLabelWidth = EditorGUIUtility.labelWidth;
        EditorGUIUtility.labelWidth = 160;

        EditorGUILayout.BeginVertical(windowStyle);

        GUILayout.BeginHorizontal(EditorStyles.helpBox);
        GUILayout.BeginVertical();
        EditorGUILayout.LabelField("Builds created in the <b>OVR Build APK</b> window are identical to Unity-built APKs, but use the Gradle cache to only touch changed files, resulting in shorter build times.",
            calloutStyle);

#if UNITY_2021_1_OR_NEWER
        if (EditorGUILayout.LinkButton("Documentation"))
#else
        if (GUILayout.Button("Documentation", GUILayout.ExpandWidth(false)))
#endif
        {
            Application.OpenURL("https://developer.oculus.com/documentation/unity/unity-build-android-tools/");
        }
        GUILayout.EndVertical();
        GUILayout.EndHorizontal();
        EditorGUILayout.Space(15f);

        scrollViewPos = EditorGUILayout.BeginScrollView(scrollViewPos, GUIStyle.none, GUI.skin.verticalScrollbar);
        using (new EditorGUI.DisabledScope(buildInProgress))
        {
            EditorGUILayout.BeginHorizontal();
            outputApkPath = EditorGUILayout.TextField("Built APK Path", outputApkPath);
            if (GUILayout.Button("Browse...", GUILayout.Width(80)))
                DisplayAPKPathDialog();
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.BeginHorizontal();

            PlayerSettings.Android.bundleVersionCode = EditorGUILayout.IntField(new GUIContent("Version Number",
                "Builds uploaded to the Oculus storefront are required to have incrementing version numbers.\nThis value is exposed to players."),
                PlayerSettings.Android.bundleVersionCode, GUILayout.Width(220));

            EditorGUI.BeginChangeCheck();
            bool isAutoIncrement = PlayerPrefs.GetInt(OVRGradleGeneration.prefName, 0) != 0;
            isAutoIncrement = EditorGUILayout.ToggleLeft(new GUIContent("Auto-Increment?",
                "If true, version number will be automatically incremented after every successful build."),
                isAutoIncrement, EditorStyles.miniLabel, GUILayout.Width(120));
            if (EditorGUI.EndChangeCheck())
                OVRGradleGeneration.ToggleUtilities();
            EditorGUILayout.EndHorizontal();

            EditorGUILayout.Space(15f);

            EditorGUILayout.BeginHorizontal();
            using (new EditorGUI.DisabledScope(string.IsNullOrEmpty(currConnectedDevice)))
            {
                isRunOnDevice = EditorGUILayout.Toggle("Install & Run on Device?", isRunOnDevice);
                GUILayout.FlexibleSpace();
                EditorGUILayout.LabelField(string.IsNullOrEmpty(currConnectedDevice) ? "No device connected" : $"Device: {currConnectedDevice}");
            }
            if (GUILayout.Button("Refresh", GUILayout.Width(80)))
                CheckADBDevices(out currConnectedDevice);
            EditorGUILayout.EndHorizontal();

            EditorUserBuildSettings.development = EditorGUILayout.Toggle(new GUIContent("Development Build?",
                "Development builds allow you to debug scripts. However, they're slightly slower, and they're not allowed on the Oculus storefront."),
                EditorUserBuildSettings.development);

            EditorGUILayout.Space(15f);

            EditorGUILayout.BeginHorizontal();
            saveKeystorePasswords = EditorGUILayout.Toggle(new GUIContent("Save Keystore Passwords?",
                "These values are also found in Project Settings > Player > [Android] > Publishing Settings > Project Keystore.\nStoring passwords is convenient, but reduces security."),
                saveKeystorePasswords);
            if (GUILayout.Button("Select Keystore...", GUILayout.Width(150)))
                SettingsService.OpenProjectSettings("Project/Player");
            EditorGUILayout.EndHorizontal();

            if (saveKeystorePasswords)
            {
                EditorGUI.indentLevel++;

                EditorGUILayout.LabelField("Keystore Path", PlayerSettings.Android.keystoreName);
                PlayerSettings.Android.keystorePass = EditorGUILayout.PasswordField("Keystore Password", PlayerSettings.Android.keystorePass);

                EditorGUILayout.LabelField("Key Alias Name", PlayerSettings.Android.keyaliasName);
                PlayerSettings.Android.keyaliasPass = EditorGUILayout.PasswordField("Alias Password", PlayerSettings.Android.keyaliasPass);

                EditorGUI.indentLevel--;
            }
        }
        EditorGUILayout.EndScrollView();
        EditorGUILayout.Space(10);

        EditorGUILayout.BeginHorizontal();
        GUILayout.FlexibleSpace();

        //better to perform these at end of GUI
        bool shouldCancel = false;
        bool shouldBuild = false;
        if (showCancel)
            shouldCancel = GUILayout.Button("Cancel", GUILayout.Height(30), GUILayout.Width(100));
        else
        {
            using (new EditorGUI.DisabledScope(buildInProgress))
            {
                shouldBuild = GUILayout.Button("Build", GUILayout.Height(30), GUILayout.Width(100));
            }
        }
        GUILayout.FlexibleSpace();
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.Space(10);

        // Show progress bar
        Rect progressRect = EditorGUILayout.GetControlRect(GUILayout.Height(25));
        float progress = currentStep / (float)totalBuildSteps;
        EditorGUI.ProgressBar(progressRect, progress, progressMessage);

        EditorGUIUtility.labelWidth = oldLabelWidth;
        EditorGUILayout.EndVertical();

        if (shouldBuild)
            StartBuild();
        else if (shouldCancel)
            CancelBuild();
    }

    void Update()
    {
        // Force window update if not in focus to ensure progress bar still updates
        var window = EditorWindow.focusedWindow;
        if (window != null && window.ToString().Contains("OculusBuildApp"))
        {
            Repaint();
        }
    }

    void DisplayAPKPathDialog()
    {
        string fileName = "build.apk";
        string path = "";
        if (!string.IsNullOrEmpty(outputApkPath))
        {
            try
            {
                path = Path.GetDirectoryName(outputApkPath);
                fileName = Path.GetFileName(outputApkPath);
            }
            catch (Exception)
            {
                //do nothing, we just have a malformed apkPath and should accept defaults
            }
        }

        outputApkPath = EditorUtility.SaveFilePanel("APK Path", path, fileName, "apk");
        Repaint();
    }

    void CancelBuild()
    {
        SetProgressBarMessage("Canceling . . .");

        if (syncCancelToken != null)
        {
            syncCancelToken.Cancel();
        }

        if (apkOutputSuccessful.HasValue && apkOutputSuccessful.Value)
        {
            buildThread.Abort();
            OnBuildComplete();
        }

        if (gradleBuildProcess != null && !gradleBuildProcess.HasExited)
        {
            var cancelThread = new Thread(delegate ()
            {
                CancelGradleBuild();
            });
            cancelThread.Start();
        }
    }

    void CancelGradleBuild()
    {
        Process cancelGradleProcess = new Process();
        string arguments = "-Xmx1024m -classpath \"" + gradlePath +
            "\" org.gradle.launcher.GradleMain --stop";
        var processInfo = new System.Diagnostics.ProcessStartInfo
        {
            WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal,
            FileName = jdkPath,
            Arguments = arguments,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
        };

        cancelGradleProcess.StartInfo = processInfo;
        cancelGradleProcess.EnableRaisingEvents = true;

        cancelGradleProcess.OutputDataReceived += new DataReceivedEventHandler(
            (s, e) =>
            {
                if (e != null && e.Data != null && e.Data.Length != 0)
                {
                    UnityEngine.Debug.LogFormat("Gradle: {0}", e.Data);
                }
            }
        );

        apkOutputSuccessful = false;

        cancelGradleProcess.Start();
        cancelGradleProcess.BeginOutputReadLine();
        cancelGradleProcess.WaitForExit();

        OnBuildComplete();
    }

    public void StartBuild()
    {
        showCancel = false;
        buildInProgress = true;

        InitializeProgressBar(NUM_BUILD_AND_RUN_STEPS);
        IncrementProgressBar("Exporting Unity Project . . .");

        apkOutputSuccessful = null;
        syncCancelToken = null;
        gradleBuildProcess = null;

        UnityEngine.Debug.Log("OVRBuild: Starting Unity build ...");

        SetupDirectories();

        // 1. Get scenes to build in Unity, and export gradle project
        var buildResult = UnityBuildPlayer();

        if (buildResult.summary.result == UnityEditor.Build.Reporting.BuildResult.Succeeded)
        {
            // Set static variables so build thread has updated data
            showCancel = true;
            gradlePath = OVRConfig.Instance.GetGradlePath();
            jdkPath = OVRConfig.Instance.GetJDKPath();
            androidSdkPath = OVRConfig.Instance.GetAndroidSDKPath();
            applicationIdentifier = PlayerSettings.GetApplicationIdentifier(BuildTargetGroup.Android);
            isDevelopmentBuild = EditorUserBuildSettings.development;
#if UNITY_2019_3_OR_NEWER
            productName = "launcher";
#else
			productName = Application.productName;
#endif
            dataPath = Application.dataPath;

            buildThread = new Thread(delegate ()
            {
                OVRBuildRun();
            });
            buildThread.Start();
            return;
        }
        else if (buildResult.summary.result == UnityEditor.Build.Reporting.BuildResult.Cancelled)
        {
            UnityEngine.Debug.Log("Build cancelled.");
        }
        else
        {
            UnityEngine.Debug.Log("Build failed.");
        }
        OnBuildComplete();
    }

    private UnityEditor.Build.Reporting.BuildReport UnityBuildPlayer()
    {
        // Unity introduced a possible bug with Unity 2020.1.10f1 that causes an exception
        // when building with the option 'BuildOptions.AcceptExternalModificationsToPlayer'
        // In order to maintain the same logic, we are using `EditorUserBuildSettings.exportAsGoogleAndroidProject`
#if DONT_USE_BUILD_OPTIONS_EXTERNAL_MODIFICATIONS_FLAG
        bool previousExportAsGoogleAndroidProject = EditorUserBuildSettings.exportAsGoogleAndroidProject;
        EditorUserBuildSettings.exportAsGoogleAndroidProject = true;
#endif
        try
        {
            var sceneList = GetScenesToBuild();

            var buildOptions = BuildOptions.None;
            if (isDevelopmentBuild)
                buildOptions |= (BuildOptions.Development | BuildOptions.AllowDebugging);
            if (isRunOnDevice)
                buildOptions |= BuildOptions.AutoRunPlayer;
#if !DONT_USE_BUILD_OPTIONS_EXTERNAL_MODIFICATIONS_FLAG
            buildOptions |= BuildOptions.AcceptExternalModificationsToPlayer;
#endif

            var buildPlayerOptions = new BuildPlayerOptions
            {
                scenes = sceneList.ToArray(),
                locationPathName = gradleTempExport,
                target = BuildTarget.Android,
                options = buildOptions
            };

            var buildResult = BuildPipeline.BuildPlayer(buildPlayerOptions);

            UnityEngine.Debug.Log(UnityBuildPlayerSummary(buildResult));

            return buildResult;
        }
        finally
        {
#if DONT_USE_BUILD_OPTIONS_EXTERNAL_MODIFICATIONS_FLAG
            EditorUserBuildSettings.exportAsGoogleAndroidProject = previousExportAsGoogleAndroidProject;
#endif
        }
    }

    private static string UnityBuildPlayerSummary(UnityEditor.Build.Reporting.BuildReport report)
    {
        var sb = new System.Text.StringBuilder();

        sb.Append($"Unity Build Player: Build {report.summary.result} ({report.summary.totalSize} bytes) in {report.summary.totalTime.TotalSeconds:0.00}s");

        foreach (var step in report.steps)
        {
            sb.AppendLine();
            if (step.depth > 0)
            {
                sb.Append(new String('-', step.depth));
                sb.Append(' ');
            }
            sb.Append($"{step.name}: {step.duration:g}");
        }

        return sb.ToString();
    }

    private static void OVRBuildRun()
    {
        // 2. Process gradle project
        IncrementProgressBar("Processing gradle project . . .");
        if (ProcessGradleProject())
        {
            // 3. Build gradle project
            IncrementProgressBar("Starting gradle build . . .");
            if (BuildGradleProject())
            {
                CopyAPK();

                // 4. Deploy and run
                if (isRunOnDevice)
                    DeployAPK();
            }
        }

        OnBuildComplete();
    }

    private static bool BuildGradleProject()
    {
        gradleBuildProcess = new Process();
        string arguments = "-Xmx4096m -classpath \"" + gradlePath + "\" org.gradle.launcher.GradleMain --profile ";
        if (isDevelopmentBuild)
            arguments += "assembleDebug -x validateSigningDebug";
        else
            arguments += "assembleRelease -x verifyReleaseResources";

#if UNITY_2019_3_OR_NEWER
        var gradleProjectPath = gradleExport;
#else
		var gradleProjectPath = Path.Combine(gradleExport, productName);
#endif

        var processInfo = new System.Diagnostics.ProcessStartInfo
        {
            WorkingDirectory = gradleProjectPath,
            WindowStyle = System.Diagnostics.ProcessWindowStyle.Normal,
            FileName = jdkPath,
            Arguments = arguments,
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardError = true,
            RedirectStandardOutput = true,
        };

        gradleBuildProcess.StartInfo = processInfo;
        gradleBuildProcess.EnableRaisingEvents = true;

        DateTime gradleStartTime = System.DateTime.Now;
        DateTime gradleEndTime = System.DateTime.MinValue;

        gradleBuildProcess.Exited += new System.EventHandler(
            (s, e) =>
            {
                UnityEngine.Debug.Log("Gradle: Exited");
            }
        );

        gradleBuildProcess.OutputDataReceived += new DataReceivedEventHandler(
            (s, e) =>
            {
                if (e != null && e.Data != null &&
                    e.Data.Length != 0 &&
                    (e.Data.Contains("BUILD") || e.Data.StartsWith("See the profiling report at:")))
                {
                    UnityEngine.Debug.LogFormat("Gradle: {0}", e.Data);
                    if (e.Data.Contains("SUCCESSFUL"))
                    {
                        string buildFlavor = isDevelopmentBuild ? "debug" : "release";
                        UnityEngine.Debug.LogFormat("APK Build Completed: {0}",
                            Path.Combine(gradleExport, $"build\\outputs\\apk\\{buildFlavor}", productName + $"-{buildFlavor}.apk").Replace("/", "\\"));
                        if (!apkOutputSuccessful.HasValue)
                        {
                            apkOutputSuccessful = true;
                        }
                        gradleEndTime = System.DateTime.Now;
                    }
                    else if (e.Data.Contains("FAILED"))
                    {
                        apkOutputSuccessful = false;
                    }
                }
            }
        );

        gradleBuildProcess.ErrorDataReceived += new DataReceivedEventHandler(
            (s, e) =>
            {
                if (e != null && e.Data != null &&
                    e.Data.Length != 0)
                {
                    UnityEngine.Debug.LogErrorFormat("Gradle: {0}", e.Data);
                }
                apkOutputSuccessful = false;
            }
        );

        gradleBuildProcess.Start();
        gradleBuildProcess.BeginOutputReadLine();
        IncrementProgressBar("Building gradle project . . .");

        gradleBuildProcess.WaitForExit();

        // Add a timeout for if gradle unexpectedlly exits or errors out
        Stopwatch timeout = new Stopwatch();
        timeout.Start();
        while (apkOutputSuccessful == null)
        {
            if (timeout.ElapsedMilliseconds > 5000)
            {
                UnityEngine.Debug.LogError("Gradle has exited unexpectedly.");
                apkOutputSuccessful = false;
            }
            System.Threading.Thread.Sleep(100);
        }

        return apkOutputSuccessful.HasValue && apkOutputSuccessful.Value;
    }

    private static bool ProcessGradleProject()
    {
        DateTime syncStartTime = System.DateTime.Now;
        DateTime syncEndTime = System.DateTime.MinValue;

        try
        {
            var ps = System.Text.RegularExpressions.Regex.Escape("" + Path.DirectorySeparatorChar);
            // ignore files .gradle/** build/** foo/.gradle/** and bar/build/**
            var ignorePattern = string.Format("^([^{0}]+{0})?(\\.gradle|build){0}", ps);

            var syncer = new DirectorySyncer(gradleTempExport,
                gradleExport, ignorePattern);

            syncCancelToken = new DirectorySyncer.CancellationTokenSource();
            var syncResult = syncer.Synchronize(syncCancelToken.Token);
            syncEndTime = System.DateTime.Now;
        }
        catch (Exception e)
        {
            UnityEngine.Debug.Log("OVRBuild: Processing gradle project failed with exception: " +
                e.Message);
            return false;
        }

        if (syncCancelToken.IsCancellationRequested)
        {
            return false;
        }

        return true;
    }

    private static List<string> GetScenesToBuild()
    {
        var sceneList = new List<string>();
        foreach (var scene in EditorBuildSettings.scenes)
        {
            // Enumerate scenes in project and check if scene is enabled to build
            if (scene.enabled)
            {
                sceneList.Add(scene.path);
            }
        }
        return sceneList;
    }

    public static bool CopyAPK()
    {
        string buildFlavor = isDevelopmentBuild ? "debug" : "release";
        string apkPathLocal = Path.Combine(gradleExport, productName, $"build\\outputs\\apk\\{buildFlavor}", productName + $"-{buildFlavor}.apk");
        if (File.Exists(apkPathLocal))
        {
            try
            {
                File.Copy(apkPathLocal, outputApkPath, true);
                UnityEngine.Debug.Log($"OVRBuild: Output APK generated at {outputApkPath}");
                Process.Start("explorer.exe", Path.GetDirectoryName(outputApkPath));
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }
        return false;
    }

    public static bool DeployAPK()
    {
        // Create new instance of ADB Tool
        var adbTool = new OVRADBTool(androidSdkPath);

        if (adbTool.isReady)
        {
            string apkPathLocal;
            string buildFlavor = isDevelopmentBuild ? "debug" : "release";
            string gradleExportFolder = Path.Combine(gradleExport, productName, $"build\\outputs\\apk\\{buildFlavor}");

            // Check to see if gradle output directory exists
            gradleExportFolder = gradleExportFolder.Replace("/", "\\");
            if (!Directory.Exists(gradleExportFolder))
            {
                UnityEngine.Debug.LogError("Could not find the gradle project at the expected path: " + gradleExportFolder);
                return false;
            }

            // Search for output APK in gradle output directory
            apkPathLocal = Path.Combine(gradleExportFolder, productName + $"-{buildFlavor}.apk");
            if (!System.IO.File.Exists(apkPathLocal))
            {
                UnityEngine.Debug.LogError(string.Format("Could not find {0} in the gradle project.", productName + $"-{buildFlavor}.apk"));
                return false;
            }

            string output, error;
            DateTime timerStart;

            // Ensure that the Oculus temp directory is on the device by making it
            IncrementProgressBar("Making Temp directory on device");
            string[] mkdirCommand = { "-d shell", "mkdir -p", REMOTE_APK_PATH };
            if (adbTool.RunCommand(mkdirCommand, null, out output, out error) != 0) return false;

            // Push APK to device, also time how long it takes
            timerStart = System.DateTime.Now;
            IncrementProgressBar("Pushing APK to device . . .");
            string[] pushCommand = { "-d push", "\"" + apkPathLocal + "\"", REMOTE_APK_PATH };
            if (adbTool.RunCommand(pushCommand, null, out output, out error) != 0) return false;

            // Calculate the transfer speed and determine if user is using USB 2.0 or 3.0
            // Only bother informing the user on non-trivial transfers, as for very short
            // periods of time, things like process creation overhead can dwarf the actual
            // transfer time.
            TimeSpan pushTime = System.DateTime.Now - timerStart;
            bool trivialPush = pushTime.TotalSeconds < TRANSFER_SPEED_CHECK_THRESHOLD;
            long? apkSize = (trivialPush ? (long?)null : new System.IO.FileInfo(apkPathLocal).Length);
            double? transferSpeed = (apkSize / pushTime.TotalSeconds) / BYTES_TO_MEGABYTES;
            bool informLog = transferSpeed.HasValue && transferSpeed.Value < USB_TRANSFER_SPEED_THRES;
            UnityEngine.Debug.Log("OVRADBTool: Push Success");

            // Install the APK package on the device
            IncrementProgressBar("Installing APK . . .");
            string apkPath = REMOTE_APK_PATH + "/" + productName + "-debug.apk";
            apkPath = apkPath.Replace(" ", "\\ ");
            string[] installCommand = { "-d shell", "pm install -r", apkPath };

            timerStart = System.DateTime.Now;
            if (adbTool.RunCommand(installCommand, null, out output, out error) != 0) return false;
            TimeSpan installTime = System.DateTime.Now - timerStart;
            UnityEngine.Debug.Log("OVRADBTool: Install Success");

            // Start the application on the device
            IncrementProgressBar("Launching application on device . . .");
#if UNITY_2019_3_OR_NEWER
            string playerActivityName = "\"" + applicationIdentifier + "/com.unity3d.player.UnityPlayerActivity\"";
#else
			string playerActivityName = "\"" + applicationIdentifier + "/" + applicationIdentifier + ".UnityPlayerActivity\"";
#endif
            string[] appStartCommand = { "-d shell", "am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER -S -f 0x10200000 -n", playerActivityName };
            if (adbTool.RunCommand(appStartCommand, null, out output, out error) != 0) return false;
            UnityEngine.Debug.Log("OVRADBTool: Application Start Success");

            IncrementProgressBar("Success!");

            // If the user is using a USB 2.0 cable, inform them about improved transfer speeds and estimate time saved
            if (informLog)
            {
                float usb3Time = apkSize.Value / (USB_3_TRANSFER_SPEED * BYTES_TO_MEGABYTES); // `informLog` can't be true if `apkSize` is null.
                UnityEngine.Debug.Log(string.Format("OVRBuild has detected slow transfer speeds. A USB 3.0 cable is recommended to reduce the time it takes to deploy your project by approximatly {0:0.0} seconds", pushTime.TotalSeconds - usb3Time));
                return true;
            }
        }
        else
        {
            UnityEngine.Debug.LogError("Could not find the ADB executable in the specified Android SDK directory.");
        }
		return false;
    }

    private static void OnBuildComplete()
    {
        showCancel = false;
        buildInProgress = false;
        currentStep = 0;
        SetProgressBarMessage("Waiting to build . . .", false);
    }

    private static bool CheckADBDevices(out string connectedDeviceName)
    {
        // Check if there are any ADB devices connected before starting the build process
        var adbTool = new OVRADBTool(OVRConfig.Instance.GetAndroidSDKPath());
        connectedDeviceName = null;

        if (adbTool.isReady)
        {
            List<string> devices = adbTool.GetDevices();
            if (devices.Count == 0)
            {
                UnityEngine.Debug.LogError("No ADB devices connected. Connect a device to this computer to run APK.");
                return false;
            }
            else if (devices.Count > 1)
            {
                UnityEngine.Debug.LogError("Multiple ADB devices connected. Disconnect extra devices from this computer to run APK.");
                return false;
            }
            else
            {
                connectedDeviceName = devices[0];
                return true;
            }
        }
        else
        {
            UnityEngine.Debug.LogError("OVR ADB Tool failed to initialize. Check the Android SDK path in [Edit -> Preferences -> External Tools]");
            return false;
        }
    }

    private static void SetupDirectories()
    {
        gradleTempExport = Path.Combine(Path.Combine(Application.dataPath, "../Temp"), "OVRGradleTempExport");
        gradleExport = Path.Combine(Path.Combine(Application.dataPath, "../Temp"), "OVRGradleExport");
        if (!Directory.Exists(gradleExport))
        {
            Directory.CreateDirectory(gradleExport);
        }
    }

    private static void InitializeProgressBar(int stepCount)
    {
        currentStep = 0;
        totalBuildSteps = stepCount;
    }

    private static void IncrementProgressBar(string message)
    {
        currentStep++;
        progressMessage = message;
        UnityEngine.Debug.Log("OVRBuild: " + message);
    }

    private static void SetProgressBarMessage(string message, bool log = true)
    {
        progressMessage = message;
        if (log)
            UnityEngine.Debug.Log("OVRBuild: " + message);
    }
#endif //UNITY_EDITOR_WIN && UNITY_ANDROID
}
