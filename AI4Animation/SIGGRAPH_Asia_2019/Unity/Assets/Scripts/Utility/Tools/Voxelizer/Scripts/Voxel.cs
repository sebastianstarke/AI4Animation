using System;
using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;

using UnityEngine;

namespace VoxelSystem
{

    [StructLayout(LayoutKind.Sequential)]
    public struct Voxel_t
    {
        public Vector3 position;
        public Vector2 uv;
        public uint fill;
        public uint front;

        public bool IsFrontFace()
        {
            return fill > 0 && front > 0;
        }

        public bool IsBackFace()
        {
            return fill > 0 && front < 1;
        }

        public bool IsEmpty()
        {
            return fill < 1;
        }
    }

}

