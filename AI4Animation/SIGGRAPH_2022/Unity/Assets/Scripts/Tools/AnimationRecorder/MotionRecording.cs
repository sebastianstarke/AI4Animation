using System.Collections.Generic;
using UnityEngine;
using AI4Animation;

public class MotionRecording : ScriptableObject {

    public Frame[] Frames;

    public void Reset() {
        ArrayExtensions.Clear(ref Frames);
    }

    public void AddFrame(Actor[] actors) {
        ArrayExtensions.Append(ref Frames, new Frame(actors));
    }

    public Matrix4x4[] GetTransformations(int index, int actor) {
        return Frames[index].Entities[actor].Transformations;
    }

    [System.Serializable]
    public class Frame {
        public Entity[] Entities;
        public Frame(Actor[] actors) {
            Entities = new Entity[actors.Length];
            for(int i=0; i<Entities.Length; i++) {
                Entities[i] = new Entity(actors[i].GetBoneTransformations());
            }
        }
    }

    [System.Serializable]
    public class Entity {
        public Matrix4x4[] Transformations;
        public Entity(Matrix4x4[] transformations) {
            Transformations = transformations;
        }
    }

}
