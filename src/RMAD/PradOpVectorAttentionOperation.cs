//------------------------------------------------------------------------------
// <copyright file="PradOpVectorAttentionOperation.cs" author="ameritusweb" date="5/2/2023">
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
    /// Vector attention operation.
    /// </summary>
    public class PradOpVectorAttentionOperation : Operation
    {
        private PradOp vectors;
        private PradOp probabilities;
        private PradOp result;

        /// <summary>
        /// A common method for instantiating an operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PradOpVectorAttentionOperation();
        }

        /// <summary>
        /// Performs the forward operation for the vector attention function.
        /// </summary>
        /// <param name="vectors">The first input to the vector attention operation.</param>
        /// <param name="probabilities">The second input to the vector attention operation.</param>
        /// <returns>The output of the vector attention operation.</returns>
        public Matrix Forward(Matrix vectors, Matrix probabilities)
        {
            var vectorTools = new PradVectorTools();
            this.vectors = new PradOp(vectors.ToTensor());
            this.probabilities = new PradOp(probabilities.ToTensor());

            var res = vectorTools.VectorAttention(this.vectors, this.probabilities);

            this.result = res.PradOp;

            this.Output = res.Result.ToMatrix();

            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dOutput)
        {
            this.result.Back(dOutput.ToTensor());

            var dVectors = this.vectors.SeedGradient.ToMatrix();

            var dProbabilities = this.probabilities.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dVectors)
                .AddInputGradient(dProbabilities)
                .Build();
        }
    }
}
