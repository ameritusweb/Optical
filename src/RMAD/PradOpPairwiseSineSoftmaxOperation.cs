//------------------------------------------------------------------------------
// <copyright file="PradOpPairwiseSineSoftmaxOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace PradOpExample.RMAD
{
    using ParallelReverseAutoDiff.PRAD;
    using ParallelReverseAutoDiff.PRAD.VectorTools;
    using ParallelReverseAutoDiff.RMAD;
    using System;

    /// <summary>
    /// Sine Softmax operation.
    /// </summary>
    public class PradOpPairwiseSineSoftmaxOperation : Operation
    {
        private PradOp input;
        private PradOp result;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PradOpPairwiseSineSoftmaxOperation();
        }

        /// <summary>
        /// Performs the forward operation for the softmax function.
        /// </summary>
        /// <param name="input">The input to the softmax operation.</param>
        /// <returns>The output of the softmax operation.</returns>
        public Matrix Forward(Matrix input)
        {
            var vectorTools = new PradVectorTools();
            this.input = new PradOp(input.ToTensor());

            var res = vectorTools.PairwiseSineSoftmax(this.input);
            this.result = res.PradOp;

            this.Output = res.Result.ToMatrix();

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            this.result.Back(dLdOutput.ToTensor());

            var dLdInput = this.input.SeedGradient;

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
