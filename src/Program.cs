namespace PradOpExample
{
    internal class Program
    {
        static async Task Main(string[] args)
        {
            var trainer = new VectorFieldNetTrainer();

            await trainer.Train();
        }
    }
}
