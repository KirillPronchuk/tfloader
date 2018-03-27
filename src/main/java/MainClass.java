import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.TensorFlow;
import sun.awt.image.ByteInterleavedRaster;

import javax.imageio.ImageIO;
import javax.imageio.stream.ImageOutputStream;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;

public class MainClass {

    public static void main(String[] args) {
        System.out.println("Hello World! I'm using tensorflow version " + TensorFlow.version());
//30km.jpg -- 1
//noEntry.jpg -- 17
//keepRight.jpg -- 38
//caution.jpg -- 18
//bumpyRoad.jpg -- 22
        try {
            float[] array = new float[1024];

            int IMG_WIDTH = 32;
            int IMG_HEIGHT = 32;
            BufferedImage source = ImageIO.read(new File("/home/kirill/IdeaProjects/tfloader/src/main/resources/examples/bumpyRoad.jpg"));
            BufferedImage gray = new BufferedImage(IMG_WIDTH, IMG_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
            Graphics2D g2d = gray.createGraphics();
            g2d.drawImage(source, 0, 0, IMG_WIDTH, IMG_HEIGHT, null);
            g2d.dispose();
//                array = ((ByteInterleavedRaster)gray.getRaster()).getDataStorage();
            for (int i = 0; i < gray.getHeight(); i++) {
                for (int j = 0; j < gray.getWidth(); j++) {
                    int rgb = gray.getRGB(i, j);
                    int r = (rgb >> 16) & 0xFF;
                    int g = (rgb >> 8) & 0xFF;
                    int b = (rgb & 0xFF);
                    int grayPixel = (r + g + b) / 3;
                    float floatPixel = (float)grayPixel;
                    array[j * IMG_WIDTH + i] =  floatPixel/255*2-1;
                }
            }

            System.out.println("Total bytes: " + array.length);


            SavedModelBundle smb = SavedModelBundle.load("/home/kirill/IdeaProjects/tfloader/src/main/resources", "serve");
            Session s = smb.session();

            FloatBuffer fb = FloatBuffer.allocate(1024);

            fb.put(array);
            fb.rewind();

            Tensor inputTensor = Tensor.create(new long[]{1, 32, 32, 1}, fb);

            Tensor result = s.runner()
                    .feed("input_tensor", inputTensor)
                    .fetch("output_tensor")
                    .run().get(0);

            float[][] m = new float[1][43];
            m[0] = new float[43];
            Arrays.fill(m[0], 0);


            float[][] matrix = (float[][]) result.copyTo(m);
            float maxVal = 0;
            int inc = 0;
            int predict = -1;
            for (float val : matrix[0]) {
                if (val > maxVal) {
                    predict = inc;
                    maxVal = val;
                }
                inc++;
            }
            System.out.println("Class № is " + predict);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
