using System;
using System.Collections.Generic;
using System.Linq;
using UnityEditor;
using UnityEngine;

[Serializable]
public class Spawnable : ISerializationCallbackReceiver
{
    public SimpleResizable ResizablePrefab;
    public string ClassificationLabel = "";

    [SerializeField] private int _editorClassificationIndex;

    public void OnBeforeSerialize()
    {
    }

    public void OnAfterDeserialize()
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

            // We do this every time we deserialize in case the classification options have been updated
            // This ensures that the label displayed
            _editorClassificationIndex = IndexOf(ClassificationLabel, OVRSceneManager.Classification.List);

            if (_editorClassificationIndex < 0)
            {
                Debug.LogError($"[{nameof(Spawnable)}] OnAfterDeserialize() " + ClassificationLabel +
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
[CustomPropertyDrawer(typeof(Spawnable))]
internal class SpawnableEditor : PropertyDrawer
{
    private static readonly string[] ClassificationList = OVRSceneManager.Classification.List.ToArray();

    public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
    {
        return base.GetPropertyHeight(property, label) * 2.2f;
    }

    public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
    {
        SerializedProperty labelProperty = property.FindPropertyRelative(nameof(Spawnable.ClassificationLabel));
        SerializedProperty editorClassificationIndex = property.FindPropertyRelative("_editorClassificationIndex");
        SerializedProperty prefab = property.FindPropertyRelative(nameof(Spawnable.ResizablePrefab));

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
            editorClassificationIndex.intValue = EditorGUI.Popup(rect, 0, list.ToArray()) - 1;
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

        EditorGUI.ObjectField(new Rect(position.x, y + EditorGUI.GetPropertyHeight(labelProperty), position.width, h),
            prefab);
        EditorGUI.EndProperty();
    }
}
#endif
