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
using System.Reflection;
using UnityEditor;

[InitializeOnLoad]
internal static class ConsoleLinkEventHandler
{
    public static event Action<Dictionary<string, string>> OnConsoleLink;

#if UNITY_2021_2_OR_NEWER
    // Much simpler code in 2021 as the API became public
    static ConsoleLinkEventHandler()
    {
		EditorGUI.hyperLinkClicked += OnConsoleLinkInternal;
    }

    private static void OnConsoleLinkInternal(EditorWindow window, HyperLinkClickedEventArgs arguments)
    {
        OnConsoleLink?.Invoke(arguments.hyperLinkData);
    }
#else
    // Using reflection to hack into the internal hyperLinkClicked event before 2021
    private static PropertyInfo _parametersProperty;

    static ConsoleLinkEventHandler()
    {
        var evt = typeof(EditorGUI).GetEvent("hyperLinkClicked", BindingFlags.Static | BindingFlags.NonPublic);
        if (evt != null)
        {
            var method = typeof(ConsoleLinkEventHandler).GetMethod("OnConsoleLinkInternal", BindingFlags.Static | BindingFlags.NonPublic);
            if (method != null)
            {
                var handler = Delegate.CreateDelegate(evt.EventHandlerType, method);
                evt.AddMethod.Invoke(null, new object[] {handler});
            }
        }
    }

    private static bool InitialiseParametersProperty(EventArgs arguments)
    {
        if (_parametersProperty == null)
        {
            _parametersProperty = arguments.GetType().GetProperty("hyperlinkInfos", BindingFlags.Instance | BindingFlags.Public);
        }

        return _parametersProperty != null;
    }

    private static void OnConsoleLinkInternal(object sender, EventArgs arguments)
    {
        if (!InitialiseParametersProperty(arguments))
        {
            return;
        }

        if (_parametersProperty.GetValue(arguments) is Dictionary<string, string> argumentsAsDictionary)
        {
            OnConsoleLink?.Invoke(argumentsAsDictionary);
        }
    }
#endif
}
