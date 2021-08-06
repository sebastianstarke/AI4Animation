#if UNITY_EDITOR
public class Precomputable<T> {
    public T Value;
    public Precomputable(T value) {
        Value = value;
    }
}
#endif