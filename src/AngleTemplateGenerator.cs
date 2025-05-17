using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Text;

namespace PradOpExample
{
    public class AngleTemplateGenerator
    {
        public double[,] GenerateAngleTemplate(char character)
        {
            int width = 50;
            int height = 50;
            using var bmp = new Bitmap(width, height);
            using var g = Graphics.FromImage(bmp);

            // Start with a small font size and measure
            float fontSize = 1;
            SizeF textSize;
            using (var font = new Font("Arial", fontSize))
            {
                textSize = g.MeasureString(character.ToString(), font);
            }

            // Calculate scaling factor to fill ~80% of the bitmap (leaving margin)
            float targetWidth = width * 0.8f;
            float targetHeight = height * 0.8f;
            float widthRatio = targetWidth / textSize.Width;
            float heightRatio = targetHeight / textSize.Height;

            // Use the smaller ratio to maintain aspect ratio
            float scaleFactor = Math.Min(widthRatio, heightRatio);
            fontSize *= scaleFactor;

            // Draw with calculated size
            using var finalFont = new Font("Arial", fontSize);
            g.Clear(Color.White);

            // Center the character
            textSize = g.MeasureString(character.ToString(), finalFont);
            float xF = (width - textSize.Width) / 2;
            float yF = (height - textSize.Height) / 2;

            g.DrawString(character.ToString(), finalFont, Brushes.Black, xF, yF);

            // Convert to angle template as before
            var template = new double[height, width];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    var pixel = bmp.GetPixel(x, y);
                    var grayscale = (pixel.R + pixel.G + pixel.B) / (3.0 * 255);
                    template[y, x] = grayscale < 0.5 ? (3 * Math.PI / 4) : (Math.PI / 4);
                }

            return template;
        }
    }
}
