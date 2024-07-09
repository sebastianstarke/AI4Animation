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

using System.Collections;
using System.Collections.Generic;
using System.IO;
using System;
using UnityEngine;
using OVRSimpleJSON;

public enum OVRGLTFInputNode
{
    None,
    Button_A_X,
    Button_B_Y,
    Button_Oculus_Menu,
    Trigger_Grip,
    Trigger_Front,
    ThumbStick
};

public class OVRGLTFAnimatinonNode
{
    private OVRGLTFInputNode m_intputNodeType;
    private JSONNode m_jsonData;
    private OVRBinaryChunk m_binaryChunk;
    private GameObject m_gameObj;
    private InputNodeState m_inputNodeState = new InputNodeState();

    private List<Vector3> m_translations = new List<Vector3>();
    private List<Quaternion> m_rotations = new List<Quaternion>();
    private List<Vector3> m_scales = new List<Vector3>();

    private static Dictionary<OVRGLTFInputNode, int> InputNodeKeyFrames = new Dictionary<OVRGLTFInputNode, int>{
        {OVRGLTFInputNode.Button_A_X, 5},
        {OVRGLTFInputNode.Button_B_Y, 8},
        {OVRGLTFInputNode.Button_Oculus_Menu, 24},
        {OVRGLTFInputNode.Trigger_Grip, 21},
        {OVRGLTFInputNode.Trigger_Front, 16},
        {OVRGLTFInputNode.ThumbStick, 0},
    };
    private static List<int> ThumbStickKeyFrames = new List<int> { 29, 39, 34, 40, 31, 36, 32, 37 };
    private static Vector2[] CardDirections = new[]{
      new Vector2(0.0f, 0.0f), // none
      new Vector2(0.0f, 1.0f), // N
      new Vector2(1.0f, 1.0f), // NE
      new Vector2(1.0f, 0.0f), // E
      new Vector2(1.0f, -1.0f), // SE
      new Vector2(0.0f, -1.0f), // S
      new Vector2(-1.0f, -1.0f), // SW
      new Vector2(-1.0f, 0.0f), // W
      new Vector2(-1.0f, 1.0f)}; // NW

    private enum ThumbstickDirection
    {
        None,
        North,
        NorthEast,
        East,
        SouthEast,
        South,
        SouthWest,
        West,
        NorthWest,
    };

    private enum OVRGLTFTransformType
    {
        None,
        Translation,
        Rotation,
        Scale
    };

    private enum OVRInterpolationType
    {
        None,
        LINEAR,
        STEP,
        CUBICSPLINE
    };

    private struct InputNodeState
    {
        public bool down;
        public float t;
        public Vector2 vecT;
    }

    public OVRGLTFAnimatinonNode(JSONNode jsonData, OVRBinaryChunk binaryChunk, OVRGLTFInputNode inputNodeType, GameObject gameObj)
    {
        m_jsonData = jsonData;
        m_binaryChunk = binaryChunk;
        m_intputNodeType = inputNodeType;
        m_gameObj = gameObj;
        m_translations.Add(CloneVector3(m_gameObj.transform.localPosition));
        m_rotations.Add(CloneQuaternion(m_gameObj.transform.localRotation));
        m_scales.Add(CloneVector3(m_gameObj.transform.localScale));
    }

    public void AddChannel(JSONNode channel, JSONNode samplers)
    {
        int samplerId = channel["sampler"].AsInt;
        var target = channel["target"];
        int nodeId = target["node"].AsInt;
        OVRGLTFTransformType transformType = GetTransformType(target["path"].Value);
        ProcessAnimationSampler(samplers[samplerId], nodeId, transformType);
        return;
    }

    public void UpdatePose(bool down)
    {
        if (m_inputNodeState.down == down)
            return;
        m_inputNodeState.down = down;

        if(m_translations.Count > 1)
            m_gameObj.transform.localPosition = (down ? m_translations[1] : m_translations[0]);
        if (m_rotations.Count > 1)
            m_gameObj.transform.localRotation = (down ? m_rotations[1] : m_rotations[0]);
        if (m_scales.Count > 1)
            m_gameObj.transform.localScale = (down ? m_scales[1] : m_scales[0]);
    }

    public void UpdatePose(float t)
    {
        const float deadZone = 0.05f;
        if (Math.Abs(m_inputNodeState.t - t) < deadZone)
            return;
        m_inputNodeState.t = t;

        if (m_translations.Count > 1)
            m_gameObj.transform.localPosition = Vector3.Lerp(m_translations[0], m_translations[1], t);
        if (m_rotations.Count > 1)
            m_gameObj.transform.localRotation = Quaternion.Lerp(m_rotations[0], m_rotations[1], t);
        if (m_scales.Count > 1)
            m_gameObj.transform.localScale = Vector3.Lerp(m_scales[0], m_scales[1], t);
    }

    public void UpdatePose(Vector2 joystick)
    {
        const float deadZone = 0.05f;
        if (Math.Abs((m_inputNodeState.vecT - joystick).magnitude) < deadZone)
            return;
        m_inputNodeState.vecT.x = joystick.x;
        m_inputNodeState.vecT.y = joystick.y;

        if(m_rotations.Count != (int)ThumbstickDirection.NorthWest + 1)
        {
            Debug.LogError("Wrong joystick animation data.");
            return;
        }

        Tuple<ThumbstickDirection, ThumbstickDirection> dir = GetCardinalThumbsticks(joystick);
        Vector2 weights = GetCardinalWeights(joystick, dir);
        Quaternion a = CloneQuaternion(m_rotations[0]);
        for (int i = 0; i < 2; i++)
        {
            float t = weights[i];
            if (t != 0)
            {
                int poseIndex = (i == 0 ? (int)dir.Item1 : (int)dir.Item2) - (int)ThumbstickDirection.North;
                Quaternion b = m_rotations[poseIndex+1];
                a = Quaternion.Slerp(a, b, t);
            }
        }
        m_gameObj.transform.localRotation = a;
        if (m_translations.Count > 1 || m_scales.Count > 1)
            Debug.LogError("Unsupported pose.");
    }

    // We will blend the 2 closest animations, this picks which 2.
    private Tuple<ThumbstickDirection, ThumbstickDirection> GetCardinalThumbsticks(Vector2 joystick)
    {
        const float deadZone = 0.005f;
        if (joystick.magnitude < deadZone)
        {
            return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.None, ThumbstickDirection.None);
        }

        // East half
        if (joystick.x >= 0.0f)
        {
            // Northeast quadrant
            if (joystick.y >= 0.0f)
            {
                // North-Northeast
                if (joystick.y > joystick.x)
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.North, ThumbstickDirection.NorthEast);
                }
                // East-Northeast
                else
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.NorthEast, ThumbstickDirection.East);
                }
            }
            // Southeast quadrant
            else
            {
                // East-Southeast
                if (joystick.x > -joystick.y)
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.East, ThumbstickDirection.SouthEast);
                }
                // South-southeast
                else
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.SouthEast, ThumbstickDirection.South);
                }
            }
        }
        // West half
        else
        {
            // Southwest quadrant
            if (joystick.y < 0.0f)
            {
                // South-Southwest
                if (joystick.x > joystick.y)
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.South, ThumbstickDirection.SouthWest);
                }
                // West-Southwest
                else
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.SouthWest, ThumbstickDirection.West);
                }
            }
            // Northwest quadrant
            else
            {
                // West-Northwest
                if (-joystick.x > joystick.y)
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.West, ThumbstickDirection.NorthWest);
                }
                // North-Northwest
                else
                {
                    return new Tuple<ThumbstickDirection, ThumbstickDirection>(ThumbstickDirection.NorthWest, ThumbstickDirection.North);
                }
            }
        }
    }

    // This figures out how much of each of the 2 animations to blend, based on where in between the 2
    // cardinal directions the user is actually pushing the thumbstick, and how far they are pushing
    // the thumbstick("animations" themselves are a fixed pose for a "maximum" push.
    private Vector2 GetCardinalWeights(Vector2 joystick, Tuple<ThumbstickDirection, ThumbstickDirection> cardinals)
    {
        // follows ThumbstickDirection, can use ThumbstickDirection to directly index into this

        if (cardinals.Item1 == ThumbstickDirection.None || cardinals.Item2 == ThumbstickDirection.None)
        {
            return new Vector2(0.0f, 0.0f);
        }

        // Compute the barycentric coordinates of the joystick position in the triangle formed by the 2
        // cardinal directions
        Vector2 triangleEdge1 = CardDirections[(int)(cardinals.Item1)];
        Vector2 triangleEdge2 = CardDirections[(int)(cardinals.Item2)];
        float dot11 = Vector2.Dot(triangleEdge1, triangleEdge1);
        float dot12 = Vector2.Dot(triangleEdge1, triangleEdge2);
        float dot1j = Vector2.Dot(triangleEdge1, joystick);
        float dot22 = Vector2.Dot(triangleEdge2, triangleEdge2);
        float dot2j = Vector2.Dot(triangleEdge2, joystick);

        float invDenom = 1.0f / (dot11 * dot22 - dot12 * dot12);
        float weight1 = (dot22 * dot1j - dot12 * dot2j) * invDenom;
        float weight2 = (dot11 * dot2j - dot12 * dot1j) * invDenom;

        return new Vector2(weight1, weight2);
    }

    private void ProcessAnimationSampler(JSONNode samplerNode, int nodeId, OVRGLTFTransformType transformType)
    {
        //We don't need input at this moment
        //int inputId = samplerNode["input"].AsInt;

        int outputId = samplerNode["output"].AsInt;
        OVRInterpolationType interpolationId = ToOVRInterpolationType(samplerNode["interpolation"].Value);
        if(interpolationId == OVRInterpolationType.None)
        {
            Debug.LogError("Unsupported interpolation type: " + samplerNode["interpolation"].Value);
            return;
        }

        var jsonAccessor = m_jsonData["accessors"][outputId];
        OVRGLTFAccessor outputReader = new OVRGLTFAccessor(jsonAccessor, m_jsonData);
        switch (transformType)
        {
            case OVRGLTFTransformType.Translation:
                Vector3[] translations = new Vector3[outputReader.GetDataCount()];
                outputReader.ReadAsVector3(m_binaryChunk, ref translations, 0, OVRGLTFLoader.GLTFToUnitySpace);
                CopyData(m_translations, translations);
                break;
            case OVRGLTFTransformType.Rotation:
                Vector4[] rotations = new Vector4[outputReader.GetDataCount()];
                outputReader.ReadAsVector4(m_binaryChunk, ref rotations, 0, OVRGLTFLoader.GLTFToUnitySpace_Rotation);
                List<Vector4> rotationDest = new List<Vector4>();
                CopyData(rotationDest, rotations);
                foreach (Vector4 v in rotationDest)
                {
                    m_rotations.Add(new Quaternion(v.x, v.y, v.z, v.w));
                }
                break;
            case OVRGLTFTransformType.Scale:
                Vector3[] scales = new Vector3[outputReader.GetDataCount()];
                outputReader.ReadAsVector3(m_binaryChunk, ref scales, 0, new Vector3(1, 1, 1));
                CopyData(m_scales, scales);
                break;
            default:
                Debug.LogError("Unsupported transform type: " + transformType.ToString());
                break;
        }
    }

    private OVRGLTFTransformType GetTransformType(string transform)
    {
        switch (transform)
        {
            case "translation":
                return OVRGLTFTransformType.Translation;
            case "rotation":
                return OVRGLTFTransformType.Rotation;
            case "scale":
                return OVRGLTFTransformType.Scale;
            default:
                Debug.LogError("Unsupported transform type: " + transform);
                return OVRGLTFTransformType.None;
        }
    }

    private OVRInterpolationType ToOVRInterpolationType(string interpolationType)
    {
        switch (interpolationType)
        {
            case "LINEAR":
                return OVRInterpolationType.LINEAR;
            case "STEP":
                Debug.LogError("Unsupported interpolationType type." + interpolationType);
                return OVRInterpolationType.STEP;
            case "CUBICSPLINE":
                Debug.LogError("Unsupported interpolationType type." + interpolationType);
                return OVRInterpolationType.CUBICSPLINE;
            default:
                Debug.LogError("Unsupported interpolationType type." + interpolationType);
                return OVRInterpolationType.None;
        }
    }

    private void CopyData<T>(List<T> dest, T[] src)
    {
        if (m_intputNodeType == OVRGLTFInputNode.ThumbStick)
        {
            foreach (int idx in ThumbStickKeyFrames)
            {
                if (idx < src.Length)
                    dest.Add(src[idx]);
            }
        }
        else
        {
            int idx = InputNodeKeyFrames[m_intputNodeType];
            if (idx < src.Length)
                dest.Add(src[idx]);
        }
    }

    private Vector3 CloneVector3(Vector3 v)
    {
        return new Vector3(v.x, v.y, v.z);
    }
    private Quaternion CloneQuaternion(Quaternion q)
    {
        return new Quaternion(q.x, q.y, q.z, q.w);
    }
}
