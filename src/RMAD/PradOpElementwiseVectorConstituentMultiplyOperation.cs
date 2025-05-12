//------------------------------------------------------------------------------
// <copyright file="PradOpElementwiseVectorConstituentMultiplyOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace PradOpExample.RMAD
{
    using ParallelReverseAutoDiff.PRAD;
    using ParallelReverseAutoDiff.PRAD.VectorTools;
    using ParallelReverseAutoDiff.RMAD;
    using System;
    using System.Threading.Tasks;

    /// <summary>
    /// Element-wise multiplication operation.
    /// </summary>
    public class PradOpElementwiseVectorConstituentMultiplyOperation : Operation
    {
        private PradOp input1;
        private PradOp input2;
        private PradOp weights;
        private PradOp result;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PradOpElementwiseVectorConstituentMultiplyOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector constituent multiply function.
        /// </summary>
        /// <param name="input1">The first input to the vector constituent multiply operation.</param>
        /// <param name="input2">The second input to the vector constituent multiply operation.</param>
        /// <returns>The output of the vector constituent multiply operation.</returns>
        public Matrix Forward(Matrix input1, Matrix input2, Matrix weights)
        {
            var vectorTools = new PradVectorTools();
            this.input1 = new PradOp(input1.ToTensor());
            this.input2 = new PradOp(input2.ToTensor());
            this.weights = new PradOp(weights.ToTensor());

            var res = vectorTools.VectorBasedMatrixMultiplication(this.input1, this.input2, this.weights);

            this.result = res.PradOp;

            this.Output = res.Result.ToMatrix();

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            this.result.Back(dOutput.ToTensor());

            var dInput1 = this.input1.SeedGradient.ToMatrix();
            var dInput2 = this.input2.SeedGradient.ToMatrix();
            var dWeights = this.weights.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dInput1)
                .AddInputGradient(dInput2)
                .AddInputGradient(dWeights)
                .Build();
        }

    }
}
