//------------------------------------------------------------------------------
// <copyright file="PradOpVectorizeOperation.cs" author="ameritusweb" date="5/2/2023">
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
    /// Performs the forward and backward operations for the vectorize function.
    /// </summary>
    public class PradOpVectorizeOperation : Operation
    {
        private PradOp input;
        private PradOp angles;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PradOpVectorizeOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vectorize function.
        /// </summary>
        /// <param name="input">The input to the vectorize operation.</param>
        /// <returns>The output of the vectorize operation.</returns>
        public Matrix Forward(Matrix input, Matrix angles)
        {
            var vectorTools = new PradVectorTools();
            this.input = new PradOp(input.ToTensor());
            this.angles = new PradOp(angles.ToTensor());

            int rows = input.Length;
            int cols = input[0].Length;

            if (cols != angles[0].Length)
            {
                throw new ArgumentException("Input and angles matrices must have the same number of columns.");
            }

            var res = vectorTools.Vectorize(this.input, this.angles);
            this.resultOp = res.PradOp;

            this.Output = res.Result.ToMatrix();

            return this.Output;
        }


        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            this.resultOp.Back(dLdOutput.ToTensor());

            var dLdInput = this.input.SeedGradient.ToMatrix();
            var dLdAngles = this.angles.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .AddInputGradient(dLdAngles)
                .Build();
        }
    }
}
