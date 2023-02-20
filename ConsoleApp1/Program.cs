using ConsoleApp1;
 
using System;

class Program
{
    static void Main(string[] args)
    {
        // Define neural network parameters
        int inputSize = 2;
        int hiddenSize = 4;
        int outputSize = 1;
        double learningRate = 0.1;

        // Create neural network
        NeuralNet net = new NeuralNet(inputSize, hiddenSize, outputSize);

        // Train neural network on dataset
        double[,] inputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 }
        };
        double[] targets = { 0, 1, 1, 0 };
        for (int epoch = 0; epoch < 10000; epoch++)
        {
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                double[] input = new double[inputSize];
                for (int j = 0; j < inputSize; j++)
                {
                    input[j] = inputs[i, j];
                }
                double[] target = { targets[i] };
                net.Train(input, target, learningRate);
            }
        }

        // Test neural network on new inputs
        Console.WriteLine("0 XOR 0 = " + net.Forward(new double[] { 0, 0 })[0]);
        Console.WriteLine("0 XOR 1 = " + net.Forward(new double[] { 0, 1 })[0]);
        Console.WriteLine("1 XOR 0 = " + net.Forward(new double[] { 1, 0 })[0]);
        Console.WriteLine("1 XOR 1 = " + net.Forward(new double[] { 1, 1 })[0]);



    }

}