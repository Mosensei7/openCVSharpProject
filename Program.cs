using OpenCvSharp;
using System;
using System.IO;

namespace ImageProcessingProject
{
    class Program
    {
        static void Main(String[] args)
        {
            Console.WriteLine("Enter the path of the image:");
            String imagePath = Console.ReadLine();

            // Normalize the image path
            imagePath = imagePath.Trim('\"');

            // Validate if the file exists
            if (!File.Exists(imagePath))
            {
                Console.WriteLine("File not found. Check the path.");
                return;
            }
            
            // Validate the file format
            String[] supportedFormats = { ".jpg", ".jpeg", ".png", ".bmp", ".tiff" };
            String fileExtension = Path.GetExtension(imagePath).ToLower();

            if (Array.IndexOf(supportedFormats, fileExtension) == -1)
            {
                Console.WriteLine("Unsupported file format. Please use .jpg, .png, .bmp, or .tiff.");
                return;
            }

            // 1. Open an image
            Mat originalImage = Cv2.ImRead(imagePath);

            if (originalImage.Empty())
            {
                Console.WriteLine("Failed to read the image. Check the file path or format.");
                return;
            }

            // Resize the image if it's too large
            const int maxWidth = 1920; // Full HD width
            const int maxHeight = 1080; // Full HD height
            if (originalImage.Width > maxWidth || originalImage.Height > maxHeight)
            {
                double scale = Math.Min((double)maxWidth / originalImage.Width, (double)maxHeight / originalImage.Height);
                Cv2.Resize(originalImage, originalImage, new OpenCvSharp.Size(0, 0), scale, scale);
                Console.WriteLine($"Resized large image to {originalImage.Width}x{originalImage.Height}.");
            }

            // Display Original Image
            Cv2.ImShow("Original Image", originalImage);

            // 2. Convert to Grayscale
            Mat grayImage = new Mat();
            Cv2.CvtColor(originalImage, grayImage, ColorConversionCodes.BGR2GRAY);
            Cv2.ImShow("Grayscale Image", grayImage);

            // Splitting Channels and Visualizing in Color
            if (originalImage.Channels() == 3)
            {
                Mat[] channels = Cv2.Split(originalImage);

                // Create individual color channel images
                Mat redChannel = new Mat();
                Mat greenChannel = new Mat();
                Mat blueChannel = new Mat();

                // Red channel (set green and blue to 0)
                Cv2.Merge(new Mat[] { Mat.Zeros(channels[0].Size(), channels[0].Type()), Mat.Zeros(channels[0].Size(), channels[0].Type()), channels[2] }, redChannel);

                // Green channel (set red and blue to 0)
                Cv2.Merge(new Mat[] { Mat.Zeros(channels[1].Size(), channels[1].Type()), channels[1], Mat.Zeros(channels[1].Size(), channels[1].Type()) }, greenChannel);

                // Blue channel (set red and green to 0)
                Cv2.Merge(new Mat[] { channels[0], Mat.Zeros(channels[0].Size(), channels[0].Type()), Mat.Zeros(channels[0].Size(), channels[0].Type()) }, blueChannel);

                // Display each channel in its respective color
                Cv2.ImShow("Red Channel", redChannel);
                Cv2.ImShow("Green Channel", greenChannel);
                Cv2.ImShow("Blue Channel", blueChannel);
            }
            else
            {
                Console.WriteLine("The image does not have 3 channels (BGR). Skipping channel display.");
            }

            // 3. Histogram and Histogram Equalization
            Mat equalizedGray = new Mat();
            Cv2.EqualizeHist(grayImage, equalizedGray);
            Cv2.ImShow("Equalized Grayscale Image", equalizedGray);

            // Calculate and Display Histogram
            PlotHistogram(grayImage, "Gray Image Histogram");

            // 4. Apply Filters
            // Canny Edge Detection
            Mat edges = new Mat();
            Cv2.Canny(grayImage, edges, 100, 200);
            Cv2.ImShow("Canny Edge Detection", edges);

            // Gaussian Blur
            Mat gaussianBlur = new Mat();
            Cv2.GaussianBlur(originalImage, gaussianBlur, new OpenCvSharp.Size(15, 15), 0);
            Cv2.ImShow("Gaussian Blur", gaussianBlur);

            // Smoothing
            Mat smoothed = new Mat();
            Cv2.Blur(originalImage, smoothed, new OpenCvSharp.Size(5, 5));
            Cv2.ImShow("Smoothing", smoothed);

            // Laplacian Filter
            Mat laplacian = new Mat();
            Cv2.Laplacian(grayImage, laplacian, MatType.CV_64F);
            Cv2.ImShow("Laplacian Filter", laplacian);

            // Wait for user to press a key before closing
            Cv2.WaitKey(0);
            Cv2.DestroyAllWindows();
        }

        // Function to plot and display histogram
        static void PlotHistogram(Mat image, string windowName)
        {
            int[] histSize = { 256 };
            Rangef[] ranges = { new Rangef(0, 256) };
            Mat hist = new Mat();

            Cv2.CalcHist(new Mat[] { image }, new int[] { 0 }, null, hist, 1, histSize, ranges);
            int histWidth = 512, histHeight = 400;
            int binWidth = (int)(histWidth / histSize[0]);

            Mat histImage = Mat.Zeros(histHeight, histWidth, MatType.CV_8UC3);
            Cv2.Normalize(hist, hist, 0, histImage.Rows, NormTypes.MinMax);

            for (int i = 1; i < histSize[0]; i++)
            {
                Cv2.Line(histImage,
                    new OpenCvSharp.Point(binWidth * (i - 1), histHeight - Math.Round(hist.At<float>(i - 1))),
                    new OpenCvSharp.Point(binWidth * i, histHeight - Math.Round(hist.At<float>(i))),
                    Scalar.All(255),
                    2,
                    LineTypes.Link8);
            }

            Cv2.ImShow(windowName, histImage);
        }
    }
}
