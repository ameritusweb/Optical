//------------------------------------------------------------------------------
// <copyright file="ElementwiseSquareOperation.cs" author="ameritusweb" date="5/2/2023">
// Copyright (c) 2023 ameritusweb All rights reserved.
// </copyright>
//------------------------------------------------------------------------------
namespace PradOpExample.RMAD
{
    using ParallelReverseAutoDiff.PRAD;
    using ParallelReverseAutoDiff.RMAD;

    /// <summary>
    /// Performs the forward and backward operations for the element-wise square function.
    /// </summary>
    public class PradOpElementwiseSquareOperation : Operation
    {
        private PradOp input;
        private PradOp resultOp;

        /// <summary>
        /// A common factory method for instantiating this operation.
        /// </summary>
        /// <param name="net">The neural network.</param>
        /// <returns>The instantiated operation.</returns>
        public static IOperation Instantiate(NeuralNetwork net)
        {
            return new PradOpElementwiseSquareOperation();
        }

        /// <summary>
        /// Performs the forward operation for the element-wise square function.
        /// </summary>
        /// <param name="input">The input to the element-wise square operation.</param>
        /// <returns>The output of the element-wise square operation.</returns>
        public Matrix Forward(Matrix input)
        {
            this.input = new PradOp(input.ToTensor());
            var result = this.input.Square();
            this.resultOp = result.PradOp;
            this.Output = result.Result.ToMatrix();
            return this.Output;
        }

        /// <inheritdoc />
        public override BackwardResult Backward(Matrix dLdOutput)
        {
            this.resultOp.Back(dLdOutput.ToTensor());

            var dLdInput = this.input.SeedGradient.ToMatrix();

            return new BackwardResultBuilder()
                .AddInputGradient(dLdInput)
                .Build();
        }
    }
}
