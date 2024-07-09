using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class MonoCallbacks : MonoBehaviour {

    private static MonoCallbacks Instance;
    private Action RenderCallback;

    public static MonoCallbacks GetInstance() {
        if(Instance == null) {
            Instance = FindObjectOfType<MonoCallbacks>();
        }
        if(Instance == null) {
            Instance = new GameObject("[MonoCallbacks]").AddComponent<MonoCallbacks>();
        }
        return Instance;
    }

    public static void AddRenderCallback(Action method) {
        GetInstance().RenderCallback += method;
    }

    void Verify() {
        if(RenderCallback == null) {
            Utility.Destroy(gameObject);
        }
    }

    void OnDrawGizmos() {
        if(!Application.isPlaying) {
            OnRenderObject();
        }
    }

    void OnRenderObject() {
        Verify();
        if(RenderCallback != null) {
            RenderCallback();
        }
    }

}