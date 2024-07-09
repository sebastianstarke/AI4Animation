using UnityEngine;

public interface IRecorder {
   public Transform Target {get;}
   public float ExportFramerate {get; set;}
   public float Framerate {get;}
   public int MaxFrame {get;}
   public float MaxTimestamp {get;}
   public void Callback(float timestamp);
   public void Animate();
   public float GetTimestamp();
}
