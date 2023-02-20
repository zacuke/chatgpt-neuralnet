using System;
namespace ConsoleApp1;

class NeuralNet
{
    private int inputSize;
    private int hiddenSize;
    private int outputSize;
    private double[,] weights1;
    private double[,] weights2;
    private double[] bias1;
    private double[] bias2;

    public NeuralNet(int inputSize, int hiddenSize, int outputSize)
    {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        Random rand = new Random();

        // Initialize weights and biases for the hidden layer
        this.weights1 = new double[inputSize, hiddenSize];
        this.bias1 = new double[hiddenSize];
        for (int i = 0; i < inputSize; i++)
        {
            for (int j = 0; j < hiddenSize; j++)
            {
                this.weights1[i, j] = rand.NextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < hiddenSize; i++)
        {
            this.bias1[i] = rand.NextDouble() * 2 - 1;
        }

        // Initialize weights and biases for the output layer
        this.weights2 = new double[hiddenSize, outputSize];
        this.bias2 = new double[outputSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            for (int j = 0; j < outputSize; j++)
            {
                this.weights2[i, j] = rand.NextDouble() * 2 - 1;
            }
        }
        for (int i = 0; i < outputSize; i++)
        {
            this.bias2[i] = rand.NextDouble() * 2 - 1;
        }
    }

    public double[] Forward(double[] inputs)
    {
        // Calculate output of hidden layer
        double[] hidden = new double[hiddenSize];
        for (int i = 0; i < hiddenSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputSize; j++)
            {
                sum += inputs[j] * weights1[j, i];
            }
            hidden[i] = Sigmoid(sum + bias1[i]);
        }

        // Calculate output of output layer
        double[] outputs = new double[outputSize];
        for (int i = 0; i < outputSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++)
            {
                sum += hidden[j] * weights2[j, i];
            }
            outputs[i] = Sigmoid(sum + bias2[i]);
        }
        return outputs;
    }

    public void Train(double[] inputs, double[] targets, double learningRate)
    {
        double[] hidden = new double[hiddenSize];
        double[] outputs = new double[outputSize];

        // Forward pass
        for (int i = 0; i < hiddenSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < inputSize; j++)
            {
                sum += inputs[j] * weights1[j, i];
            }
            hidden[i] = Sigmoid(sum + bias1[i]);
        }

        for (int i = 0; i < outputSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < hiddenSize; j++)
            {
                sum += hidden[j] * weights2[j, i];
            }
            outputs[i] = Sigmoid(sum + bias2[i]);
        }

        // Backward pass
        double[] outputError = new double[outputSize];
        double[] hiddenError = new double[hiddenSize];

        for (int i = 0; i < outputSize; i++)
        {
            outputError[i] = (targets[i] - outputs[i]) * SigmoidDerivative(outputs[i]);
            for (int j = 0; j < hiddenSize; j++)
            {
                weights2[j, i] += learningRate * hidden[j] * outputError[i];
            }
            bias2[i] += learningRate * outputError[i];
        }

        for (int i = 0; i < hiddenSize; i++)
        {
            double sum = 0;
            for (int j = 0; j < outputSize; j++)
            {
                sum += outputError[j] * weights2[i, j];
            }
            hiddenError[i] = sum * SigmoidDerivative(hidden[i]);

            for (int j = 0; j < inputSize; j++)
            {
                weights1[j, i] += learningRate * inputs[j] * hiddenError[i];
            }
            bias1[i] += learningRate * hiddenError[i];
        }
    }

    private static double Sigmoid(double x)
    {
        return 1 / (1 + Math.Exp(-x));
    }

    private static double SigmoidDerivative(double x)
    {
        return x * (1 - x);
    }
}