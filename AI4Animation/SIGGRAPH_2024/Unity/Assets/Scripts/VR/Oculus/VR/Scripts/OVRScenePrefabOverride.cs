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

using UnityEngine;
using UnityEditor;
using System.Collections.Generic;
using System.Linq;
using UnityEngine.Serialization;

/// <summary>
/// Represents a prefab that overrides the <see cref="OVRSceneManager.PlanePrefab"/> or
/// <see cref="OVRSceneManager.VolumePrefab"/> based on a semantic <see cref="OVRSceneManager.Classification"/>.
/// </summary>
[System.Serializable]
public class OVRScenePrefabOverride : ISerializationCallbackReceiver
{
  /// <summary>
  /// The prefab to instantiate instead of the default one specified in the <see cref="OVRSceneManager"/>.
  /// </summary>
  [FormerlySerializedAs("prefab")]
  public OVRSceneAnchor Prefab;

  /// <summary>
  /// The classification label that must be associated with a scene entity in order to instantiate <see cref="Prefab"/>.
  /// </summary>
  [FormerlySerializedAs("classificationLabel")]
  public string ClassificationLabel = "";

  // We use a custom property drawer to allow the user to select a label among a set of options.
  // Because the prefabOverrides is an array, and we want each entry to have their own
  // classification, we need to store an index. That cannot be stored in the custom property drawer
  // as it would be shared among all entries. We store it here instead. However, to ensure that
  // this value does not break over time, we update it after de-serialization based on the
  // classification label and the available classification options.
  [FormerlySerializedAs("editorClassificationIndex")]
  [SerializeField]
  private int _editorClassificationIndex;

  void ISerializationCallbackReceiver.OnBeforeSerialize()
  {
  }

  void ISerializationCallbackReceiver.OnAfterDeserialize()
  {
    if (ClassificationLabel != "")
    {
      int IndexOf(string label, IEnumerable<string> collection)
      {
        var index = 0;
        foreach (var item in collection)
        {
          if (item == label)
          {
            return index;
          }
          index++;
        }

        return -1;
      }

      // We do this ever time we deserialize in case the classification options have been updated
      // This ensures that the label displayed
      _editorClassificationIndex = IndexOf(ClassificationLabel, OVRSceneManager.Classification.List);

      if (_editorClassificationIndex < 0)
      {
        Debug.LogError($"[{nameof(OVRScenePrefabOverride)}] OnAfterDeserialize() " + ClassificationLabel +
          " not found. The Classification list in OVRSceneManager has likely changed");
      }
    }
    else
    {
      // No classification was selected, so we can just assign a default
      // This typically happens this object was just created
      _editorClassificationIndex = 0;
    }
  }
}

#if UNITY_EDITOR
[CustomPropertyDrawer(typeof(OVRScenePrefabOverride))]
internal class OVRSceneManagerEditor : PropertyDrawer
{
  private static readonly string[] ClassificationList = OVRSceneManager.Classification.List.ToArray();

  public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
  {
    return base.GetPropertyHeight(property, label) * 2.2f;
  }

  public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
  {
    SerializedProperty labelProperty = property.FindPropertyRelative(nameof(OVRScenePrefabOverride.ClassificationLabel));
    SerializedProperty editorClassificationIndex = property.FindPropertyRelative("_editorClassificationIndex");
    SerializedProperty prefab = property.FindPropertyRelative(nameof(OVRScenePrefabOverride.Prefab));

    EditorGUI.BeginProperty(position, label, property);

    float y = position.y;
    float h = position.height / 2;

    Rect rect = new Rect(position.x, y, position.width, h);
    if (editorClassificationIndex.intValue == -1)
    {
      var list = new List<string>
      {
          labelProperty.stringValue + " (invalid)"
      };
      list.AddRange(OVRSceneManager.Classification.List);
      editorClassificationIndex.intValue = EditorGUI.Popup(rect, 0, list.ToArray())-1;
    }
    else
    {
      editorClassificationIndex.intValue = EditorGUI.Popup(
        rect,
        editorClassificationIndex.intValue,
        ClassificationList);
    }

    if (editorClassificationIndex.intValue >= 0 &&
        editorClassificationIndex.intValue < ClassificationList.Length)
    {
      labelProperty.stringValue = OVRSceneManager.Classification.List[editorClassificationIndex.intValue];
    }

    EditorGUI.ObjectField(new Rect(position.x, y + EditorGUI.GetPropertyHeight(labelProperty), position.width, h), prefab);
    EditorGUI.EndProperty();
  }
}
#endif
