using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using LibOptimization;

public class Example_Optimization : MonoBehaviour {

    public float Area = 10f;
    public int Resolution = 1000;
    public float TileScale = 1f;

    private int Dim = 2;
    private float Height = 0.25f;
    private LibOptimization.Optimization.clsOptDE Optimizer = null;

    private Vector3[][] OptimizationLandscape = null;

    void Start() {
        double[] min = new double[Dim];
        double[] max = new double[Dim];
        for(int i=0; i<Dim; i++) {
            min[i] = -Area/2f;
            max[i] = Area/2f;
        }
        ObjectiveFunction func = new ObjectiveFunction(Dim, Height);
        Optimizer = new LibOptimization.Optimization.clsOptDE(func);
        Optimizer.LowerBounds = min;
        Optimizer.UpperBounds = max;
        Optimizer.Init();

        OptimizationLandscape = new Vector3[Resolution][];
        for(int x=0; x<Resolution; x++) {
            OptimizationLandscape[x] = new Vector3[Resolution];
            for(int y=0; y<Resolution; y++) {
                float _x_ = (float)x*Area/Resolution - Area/2f;
                float _y_ = (float)y*Area/Resolution - Area/2f;
                OptimizationLandscape[x][y] = new Vector3(_x_, (float)func.F(new List<double>(){_x_,_y_}), _y_);
            }
        }
    }

    void Update() {
        Optimizer.DoIteration();
    }

    public class ObjectiveFunction : LibOptimization.Optimization.absObjectiveFunction {
    
        private int Dim = 0;
        private float Height = 0f;

        public ObjectiveFunction(int dim, float height) {
            Dim = dim;
            Height = height;
        }

        public override int NumberOfVariable() {
            return Dim;
        }    

        public override double F(List<double> args) {
            double x = args[0];
            double y = args[1];
            double dist = System.Math.Sqrt(x*x + y*y);
            return Height * (2.0*dist + (
                System.Math.Sin(2f * System.Math.PI * x)
                +
                System.Math.Cos(2f * System.Math.PI * y)
            ));
        }    

        public override List<double> Gradient(List<double> args) {
            Debug.Log("Should not be called for DE!");
            return new List<double>();
        }

        public override List<List<double>> Hessian(List<double> args) {
            Debug.Log("Should not be called for DE!");
            return new List<List<double>>();
        }   

    }

    void OnRenderObject() {
        UltiDraw.Begin();
        float size = TileScale*Area/Resolution;
        for(int x=0; x<Resolution-1; x++) {
            for(int y=0; y<Resolution-1; y++) {
                Vector3 a = OptimizationLandscape[x][y];
                Vector3 b = OptimizationLandscape[x+1][y];
                Vector3 c = OptimizationLandscape[x][y+1];
                Vector3 d = OptimizationLandscape[x+1][y+1];
                Vector3 center = (a+b+c+d)/4f;
                UltiDraw.DrawQuad(center, Quaternion.LookRotation(new Plane(a, b, c).normal), size, size, Color.black);
            }
        }
        double[] solution = new double[Dim];
        for(int i=0; i<solution.Length; i++) {
            solution[i] = Optimizer.Result[i];
        }
        Vector3 pos = new Vector3((float)solution[0], (float)Optimizer.ObjectiveFunction.F(new List<double>(solution)), (float)solution[1]);
        UltiDraw.DrawSphere(pos, Quaternion.identity, 0.25f, Color.red);
        UltiDraw.End();
    }

}