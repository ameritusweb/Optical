using ParallelReverseAutoDiff.PRAD;

namespace PradOpExample.RMAD
{
    public class CosineSimilarityAngleLossOperation
    {
        public PradResult CalculateLoss(PradOp predictions, double targetAngle)
        {
            PradOp predictionsBranch = predictions.Branch();

            // Get x and y components of prediction
            var xPred = predictions.Indexer(":", "0:1");
            var yPred = predictionsBranch.Indexer(":", "1:2");

            PradOp xPredBranch = xPred.Branch();
            PradOp yPredBranch = yPred.Branch();

            // Create target vector components as tensors
            var xTarget = new Tensor(xPred.PradOp.CurrentShape, (float)Math.Cos(targetAngle));
            var yTarget = new Tensor(yPred.PradOp.CurrentShape, (float)Math.Sin(targetAngle));

            // Calculate dot product
            var xProduct = xPred.Then(PradOp.MulOp, xTarget);
            var yProduct = yPred.Then(PradOp.MulOp, yTarget);
            var dotProduct = xProduct.Then(PradOp.AddOp, yProduct.Result);

            // Calculate magnitudes
            var predXSquared = xPredBranch.Square();
            var predYSquared = yPredBranch.Square();
            var predMagnitude = predXSquared.Then(PradOp.AddOp, predYSquared.Result)
                                           .Then(PradOp.SquareRootOp);

            // Target magnitude is 1 since we used unit vector

            // Calculate cosine similarity
            var cosineSimilarity = dotProduct.Then(PradOp.DivOp, predMagnitude.Result);

            // Convert to loss (1 - cos_sim)
            var loss = cosineSimilarity.Then(PradOp.SubFromOp, new Tensor(cosineSimilarity.PradOp.CurrentShape, 1.0f));

            return loss;
        }
    }
}
