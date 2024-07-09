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
using System;
using System.Linq;
using UnityEditor;
using UnityEngine;

/// <summary>
/// Custom Editor for <see cref="OVRCustomFace">
/// </summary>
/// <remarks>
/// Custom Editor for <see cref="OVRCustomFace"> that supports:
/// - attempting to find an <see cref="OVRFaceExpressions"/>in the parent hierarchy to match to if none is chosen
/// - supporting string matching to attempt to automatically match <see cref="OVRFaceExpressions.FaceExpression"/> to blend shapes on the shared mesh
/// The string matching algorithm will tokenize the blend shape names and the <see cref="OVRFaceExpressions.FaceExpression"/> names and for each
/// blend shape find the <see cref="OVRFaceExpressions.FaceExpression"/> with the most total characters in matching tokens.
/// To match tokens it currently uses case invariant matching.
/// The tokenization is based on some common string seperation characters and seperation by camel case.
/// </remarks>
[CustomEditor(typeof(OVRCustomFace))]
internal sealed class OVRCustomFaceEditor : Editor
{
    private SerializedProperty _expressionsProp;
    private SerializedProperty _mappings;
    private SerializedProperty _strengthMultiplier;
    private bool _showBlendshapes = true;

    private void OnEnable()
    {
        _expressionsProp = serializedObject.FindProperty(nameof(OVRCustomFace._faceExpressions));
        _mappings = serializedObject.FindProperty(nameof(OVRCustomFace._mappings));
        _strengthMultiplier = serializedObject.FindProperty(nameof(OVRCustomFace._blendShapeStrengthMultiplier));
    }

    public override void OnInspectorGUI()
    {
        serializedObject.Update();

        EditorGUILayout.PropertyField(_expressionsProp, new GUIContent(nameof(OVRFaceExpressions)));

        EditorGUILayout.PropertyField(_strengthMultiplier, new GUIContent("Blend Shape Strength Multiplier"));

        if(_expressionsProp.objectReferenceValue == null)
        {
            _expressionsProp.objectReferenceValue = FindFaceExpressionsComponent(_expressionsProp);
        }

        SkinnedMeshRenderer renderer = GetSkinnedMeshRenderer(_expressionsProp);//need to pass out some property to find the component from

        if(renderer == null || renderer.sharedMesh == null)
        {
            if (_mappings.arraySize > 0)
            {
                _mappings.ClearArray();
            }
            serializedObject.ApplyModifiedProperties();
            return;
        }

        if(_mappings.arraySize != renderer.sharedMesh.blendShapeCount)
        {
            _mappings.ClearArray();
            _mappings.arraySize = renderer.sharedMesh.blendShapeCount;
            for (int i = 0; i < renderer.sharedMesh.blendShapeCount; ++i)
            {
                _mappings.GetArrayElementAtIndex(i).enumValueIndex = (int)OVRFaceExpressions.FaceExpression.Max;
            }
        }

        EditorGUILayout.Space();

        _showBlendshapes = EditorGUILayout.BeginFoldoutHeaderGroup(_showBlendshapes, "Blendshapes");

        if(_showBlendshapes)
        {
            if(GUILayout.Button("Auto Generate Mapping"))
            {
                OVRFaceExpressions.FaceExpression[] generatedMapping = AutoGenerateMapping(renderer.sharedMesh);

                for (int i = 0; i < renderer.sharedMesh.blendShapeCount; ++i)
                {
                    _mappings.GetArrayElementAtIndex(i).enumValueIndex = (int)generatedMapping[i];
                }
            }

            EditorGUILayout.Space();

            for (int i = 0; i < renderer.sharedMesh.blendShapeCount; ++i)
            {
                EditorGUILayout.PropertyField(_mappings.GetArrayElementAtIndex(i), new GUIContent(renderer.sharedMesh.GetBlendShapeName(i)));
            }
        }

        EditorGUILayout.EndFoldoutHeaderGroup();

        serializedObject.ApplyModifiedProperties();
    }

    private static OVRFaceExpressions FindFaceExpressionsComponent(SerializedProperty property)
    {
        GameObject targetObject = GetGameObject(property);

        if(!targetObject)
            return null;

        return (OVRFaceExpressions)targetObject.GetComponentInParent(typeof(OVRFaceExpressions));
    }

    private static SkinnedMeshRenderer GetSkinnedMeshRenderer(SerializedProperty property)
    {
        GameObject targetObject = GetGameObject(property);

        if(!targetObject)
            return null;

        return targetObject.GetComponent<SkinnedMeshRenderer>();
    }

    private static GameObject GetGameObject(SerializedProperty property)
    {
        Component targetComponent = property.serializedObject.targetObject as Component;

        if (targetComponent && targetComponent.gameObject)
        {
            return targetComponent.gameObject;
        }
        return null;
    }

    /// <summary>
    /// Find the best matching blend shape for each facial expression based on their names
    /// </summary>
    /// <remarks>
    /// Auto generation idea is to tokenize expression enum strings and blend shape name strings and find matching tokens
    /// We quantify the quality of the match by the total number of characters in the matching tokens
    /// We require at least a total of more than 2 characters to match, to avoid matching just L/R LB/RB etc.
    /// A better technique might be to use Levenshtein distance to match the tokens to allow some typos while still being loose on order of tokens
    /// </remarks>
    /// <param name="skinnedMesh">The mesh to find a mapping for.</param>
    /// <returns>Returns an array of <see cref="OVRFaceExpressions.FaceExpression"/> of the same length as the number of blendshapes on the <paramref name="skinnedMesh"/> with each element identifying the closest found match</returns>
    private static OVRFaceExpressions.FaceExpression[] AutoGenerateMapping(Mesh skinnedMesh)
    {
        var result = new OVRFaceExpressions.FaceExpression[skinnedMesh.blendShapeCount];

        var expressionTokens = new HashSet<string>[(int)OVRFaceExpressions.FaceExpression.Max];
        string[] enumNames = Enum.GetNames(typeof(OVRFaceExpressions.FaceExpression));

        for(int i = 0; i < (int)OVRFaceExpressions.FaceExpression.Max; ++i)
        {
            expressionTokens[i] = TokenizeString(enumNames[i]);
        }

        for (int i = 0; i < skinnedMesh.blendShapeCount; ++i)
        {
            result[i] = (OVRFaceExpressions.FaceExpression)FindBestMatch(expressionTokens, skinnedMesh.GetBlendShapeName(i), (int)OVRFaceExpressions.FaceExpression.Max);
        }

        return result;
    }

    private static int FindBestMatch(HashSet<string>[] tokenizedOptions, string searchString, int defaultReturn)
    {
        HashSet<string> blendShapeTokens = TokenizeString(searchString);

        int bestMatchEnumIndex = defaultReturn;
        int bestMatchCount = 2; // require more than two characters to match in an expression, to avoid just matching L/ LB/ R/RB

        for (int j = 0; j < tokenizedOptions.Length; ++j)
        {
            int thisMatchCount = 0;
            HashSet<string> thisSet = tokenizedOptions[j];
            // Currently we only allow exact matches, using Levenshtein distance for fuzzy matches
            // would allow for handling of common typos and other slight mismatches
            foreach(string matchingToken in blendShapeTokens.Intersect(thisSet))
            {
                thisMatchCount += matchingToken.Length;
            }

            if(thisMatchCount > bestMatchCount)
            {
                bestMatchCount = thisMatchCount;
                bestMatchEnumIndex = j;
            }
        }

        return bestMatchEnumIndex;
    }

    private static HashSet<string> TokenizeString(string s)
    {
        var separators = new char[]{' ','_','-',',','.',';'};
        // add both the camel case and non-camel case split versions since the
        // camel case split doesn't handle all caps
        //(it's fundamentally ambigous without natural language comprehension)
        // duplicates don't matter as we later will hash them and they should match
        var splitTokens = SplitCamelCase(s).Split(separators).Concat(s.Split(separators));

        var hashCodes = new HashSet<string>();
        foreach(string token in splitTokens)
        {
            hashCodes.Add(token.ToLowerInvariant());
        }

        return hashCodes;
    }

    private static string SplitCamelCase(string input) => System.Text.RegularExpressions.Regex.Replace(input, "([A-Z])", " $1", System.Text.RegularExpressions.RegexOptions.Compiled).Trim();
}
