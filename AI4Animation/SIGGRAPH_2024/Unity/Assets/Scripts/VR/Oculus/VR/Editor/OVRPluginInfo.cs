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
using System.IO;
using UnityEditor;
using UnityEngine;

namespace Oculus.VR.Editor
{
    public interface IOVRPluginInfoSupplier
    {
        bool IsOVRPluginOpenXRActivated();
        bool IsOVRPluginUnityProvidedActivated();
    }

    public class OVRPluginInfo : ScriptableObject
    {
        private static readonly IOVRPluginInfoSupplier Supplier =
#if OVR_UNITY_PACKAGE_MANAGER
            new OVRPluginInfoOpenXR();
#elif OVR_UNITY_ASSET_STORE
            new OVRPluginUpdater();
#else
            new OVRPluginInfoStub();
#endif

        public static bool IsOVRPluginOpenXRActivated() => Supplier.IsOVRPluginOpenXRActivated();

        public static bool IsOVRPluginUnityProvidedActivated() => Supplier.IsOVRPluginUnityProvidedActivated();

        public static string GetUtilitiesRootPath()
        {
            var so = ScriptableObject.CreateInstance(typeof(OVRPluginInfo));
            var script = MonoScript.FromScriptableObject(so);
            string assetPath = AssetDatabase.GetAssetPath(script);

            var editorDir = Directory.GetParent(assetPath);
            if (editorDir == null)
            {
                throw new DirectoryNotFoundException($"Unable to find parent directory of {assetPath}");
            }
            string editorPath = editorDir.FullName;

            var ovrDir = Directory.GetParent(editorPath);
            if (ovrDir == null)
            {
                throw new DirectoryNotFoundException($"Unable to find parent directory of {editorPath}");
            }
            return ovrDir.FullName;
        }

        public static bool IsInsidePackageDistribution()
        {
            var so = CreateInstance(typeof(OVRPluginInfo));
            var script = MonoScript.FromScriptableObject(so);
            string assetPath = AssetDatabase.GetAssetPath(script);
            return assetPath.StartsWith("Packages\\", StringComparison.InvariantCultureIgnoreCase) ||
                   assetPath.StartsWith("Packages/", StringComparison.InvariantCultureIgnoreCase);
        }

        private class OVRPluginInfoStub : IOVRPluginInfoSupplier
        {
            public bool IsOVRPluginOpenXRActivated() => true;

            public bool IsOVRPluginUnityProvidedActivated() => false;
        }
    }
}
