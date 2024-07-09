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

using UnityEditor;

[CustomEditor(typeof(OVRSceneManager))]
internal class OVRSceneManagerEditor : Editor
{
    private SerializedProperty _planePrefab;
    private SerializedProperty _volumePrefab;
    private SerializedProperty _prefabOverrides;
    private SerializedProperty _verboseLogging;
    private SerializedProperty _maxSceneAnchorUpdatesPerFrame;
    private SerializedProperty _initialAnchorParent;
    private bool _showAdvanced;

    private void OnEnable()
    {
        _planePrefab = serializedObject.FindProperty(nameof(OVRSceneManager.PlanePrefab));
        _volumePrefab = serializedObject.FindProperty(nameof(OVRSceneManager.VolumePrefab));
        _prefabOverrides = serializedObject.FindProperty(nameof(OVRSceneManager.PrefabOverrides));
        _verboseLogging = serializedObject.FindProperty(nameof(OVRSceneManager.VerboseLogging));
        _maxSceneAnchorUpdatesPerFrame = serializedObject.FindProperty(nameof(OVRSceneManager.MaxSceneAnchorUpdatesPerFrame));
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.PropertyField(_planePrefab);
        EditorGUILayout.PropertyField(_volumePrefab);
        EditorGUILayout.PropertyField(_prefabOverrides);
        _showAdvanced = EditorGUILayout.Foldout(_showAdvanced, "Advanced");
        if (_showAdvanced)
        {
            EditorGUILayout.PropertyField(_verboseLogging);
            EditorGUILayout.PropertyField(_maxSceneAnchorUpdatesPerFrame);
        }
        serializedObject.ApplyModifiedProperties();
    }
}
