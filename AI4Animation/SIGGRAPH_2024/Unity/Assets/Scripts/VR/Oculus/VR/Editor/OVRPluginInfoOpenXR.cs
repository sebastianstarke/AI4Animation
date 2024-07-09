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


#if OVR_UNITY_PACKAGE_MANAGER

using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace Oculus.VR.Editor
{
    [InitializeOnLoad]
    public class OVRPluginInfoOpenXR : IOVRPluginInfoSupplier
    {
        private const string PackageName =
            "com.meta.xr.sdk.utilities";

        private static readonly string PluginsRelPath =
            Path.Combine("Packages", PackageName, "Plugins");

        private static readonly string PluginRelPathWin64 =
            Path.Combine(PluginsRelPath, "Win64OpenXR", "OVRPlugin.dll");

        private static readonly string PluginRelPathAndroid =
            Path.Combine(PluginsRelPath, "AndroidOpenXR", "OVRPlugin.aar");

        private static bool _unityRunningInBatchMode;
        private static bool _restartPending;

        public bool IsOVRPluginOpenXRActivated() => true;

        public bool IsOVRPluginUnityProvidedActivated() => false;

        static OVRPluginInfoOpenXR()
        {
            EditorApplication.delayCall += DelayCall;
        }

        private static void DelayCall()
        {
            if (Environment.CommandLine.Contains("-batchmode"))
            {
                _unityRunningInBatchMode = true;
            }

            if (_restartPending)
            {
                return;
            }

            CheckHasPluginChanged();
        }

        private static void CheckHasPluginChanged()
        {
            var projectConfig = OVRProjectConfig.GetProjectConfig();
            string md5Win64Actual = GetFileChecksum(Path.GetFullPath(PluginRelPathWin64));
            string md5AndroidActual = GetFileChecksum(Path.GetFullPath(PluginRelPathAndroid));
            if (projectConfig.ovrPluginMd5Win64 == md5Win64Actual &&
                projectConfig.ovrPluginMd5Android == md5AndroidActual)
            {
                return;
            }
            bool userAgreedToRestart = EditorUtility.DisplayDialog(
                "Restart Unity",
                "Changes to OVRPlugin detected. Plugin updates require a restart. Please restart Unity to complete the update.",
                "Restart Editor",
                "Not Now");
            projectConfig.ovrPluginMd5Win64 = md5Win64Actual;
            projectConfig.ovrPluginMd5Android = md5AndroidActual;
            OVRProjectConfig.CommitProjectConfig(projectConfig);
            if (userAgreedToRestart)
            {
                RestartUnityEditor();
            }
            else
            {
                Debug.LogWarning("OVRPlugin not updated. Restart the editor to update.");
            }
        }

        private static string GetFileChecksum(string filePath)
        {
            using var md5 = new MD5CryptoServiceProvider();
            byte[] buffer = md5.ComputeHash(File.ReadAllBytes(filePath));
            return string.Join(null, buffer.Select(b => b.ToString("x2")));
        }

        private static void RestartUnityEditor()
        {
            if (_unityRunningInBatchMode)
            {
                Debug.LogWarning("Restarting editor is not supported in batch mode");
                return;
            }

            _restartPending = true;
            EditorApplication.OpenProject(GetCurrentProjectPath());
        }

        private static string GetCurrentProjectPath()
        {
            DirectoryInfo projectPath = Directory.GetParent(Application.dataPath);
            if (projectPath == null)
            {
                throw new DirectoryNotFoundException("Unable to find project path.");
            }
            return projectPath.FullName;
        }
    }
}

#endif
