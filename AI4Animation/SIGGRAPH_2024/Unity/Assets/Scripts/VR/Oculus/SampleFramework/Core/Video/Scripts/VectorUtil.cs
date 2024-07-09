// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public static class VectorUtil {

    public static Vector4 ToVector(this Rect rect)
    {
        return new Vector4(rect.x, rect.y, rect.width, rect.height);
    }

}
