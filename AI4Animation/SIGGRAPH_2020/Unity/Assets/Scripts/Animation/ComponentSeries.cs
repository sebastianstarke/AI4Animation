using UnityEngine;

public abstract class ComponentSeries : TimeSeries {
	public bool DrawGUI = true;
	public bool DrawScene = true;
	public ComponentSeries(TimeSeries global) : base(global) {}
    public abstract void Increment(int start, int end);
    public abstract void Interpolate(int start, int end);
	public abstract void GUI(Camera canvas=null);
	public abstract void Draw(Camera canvas=null);
}