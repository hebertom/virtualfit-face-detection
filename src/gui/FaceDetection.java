////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Author: Taha Emara
//
// Youtube Cahnnel : http://www.youtube.com/user/omar0103637
// Facebook Page : https://www.facebook.com/IcPublishes
// Linkedin Profile : http://eg.linkedin.com/pub/taha-emara/a4/1ab/524/
// E-mail: : tahaemara.eng@gmail.com
//
//                   Real time face detection using OpenCV with Java
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
package gui;

import java.awt.Color;
import java.awt.Frame;
import java.awt.GradientPaint;
import java.awt.Graphics;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.IOException;
import java.net.URISyntaxException;
import java.net.URL;
import java.security.CodeSource;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.Timer;
import java.util.TimerTask;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import org.opencv.objdetect.CascadeClassifier;
import org.opencv.videoio.VideoCapture;
import utils.BackgroundPanel;

/**
 *
 * @author Taha Emara
 */
public class FaceDetection extends javax.swing.JFrame {
///

    private Scalar CONTOUR_COLOR = new Scalar(255, 0, 0, 255);
    private int TEMPO_FOTO = 5;
    private int interval = 5;
    Timer timer;
    private final Scalar mLowerBound = new Scalar(0);
    private final Scalar mUpperBound = new Scalar(0);
    private double beta, alpha;
    private double WIDTHF = 4;
    private double HEIGHTF = 4;
    private BufferedImage qrCode, logo, maoSim, maoNao;
    private DaemonThread myThread = null;
    private List<MatOfPoint> mContours = new ArrayList<MatOfPoint>();
    private List<MatOfPoint> mContoursBlacked = new ArrayList<MatOfPoint>();
    private boolean snapShot = false;
    private BufferedImage buff, buffCamisa, buffSnap;
    int count = 0;
    VideoCapture webSource = null;
    Mat frame = new Mat();
    Mat frameBackup = new Mat();
    Mat mLogo = new Mat();
    Mat mMaoSim = new Mat();
    Mat mMaoNao = new Mat();
    Mat mHierarchy = new Mat();
    MatOfByte mem = new MatOfByte();
    Mat mMask = new Mat();
    Mat mLogoResized = new Mat();
    Mat mMaoResized = new Mat();
    Mat mMaoResizedSim = new Mat();
    Mat mLogoResizedCor = new Mat();
    MatOfByte memCamisa = new MatOfByte();
    CascadeClassifier faceDetector = new CascadeClassifier(FaceDetection.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1));
    CascadeClassifier handDetector = new CascadeClassifier(FaceDetection.class.getResource("fist.xml").getPath().substring(1));
    MatOfRect faceDetections = new MatOfRect();
    MatOfRect handsDetections = new MatOfRect();
    Rect ROI;
    Rect maiorRosto, maiorMao, ROISim, ROINao;
    boolean timerAtivo = false;
///    

    class DaemonThread implements Runnable {

        protected volatile boolean runnable = false;

        @Override
        public void run() {
            synchronized (this) {
                while (runnable) {
                    if (webSource.grab()) {
                        try {
                            Graphics g = jPanelSnapshot.getGraphics();
                            if (snapShot) {
                                if (buff != null) {
                                    Imgcodecs.imencode(".bmp", frameBackup, mem);
                                    Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                                    buffSnap = (BufferedImage) im;
                                    Graphics g3 = jPanel1.getGraphics();
                                    g3.drawImage(buffSnap, 0, 0, jPanel1.getWidth(), jPanel1.getHeight(), 0, 0, buffSnap.getWidth(), buffSnap.getHeight(), null);
                                    gerarQRCode();
                                }
                                snapShot = false;
                            } else if (buffSnap != null) {
                                Graphics g3 = jPanel1.getGraphics();
                                g3.drawImage(buffSnap, 0, 0, jPanel1.getWidth(), jPanel1.getHeight(), 0, 0, buffSnap.getWidth(), buffSnap.getHeight(), null);
                            } else if (qrCode != null) {
                                System.out.println("QRCODE");
                                Graphics g2 = jPanelQRCode.getGraphics();
                                g2.drawImage(qrCode, 0, 0, jPanelQRCode.getWidth(), jPanelQRCode.getHeight(), 0, 0, qrCode.getWidth(), qrCode.getHeight(), null);
                            }

                            webSource.retrieve(frame);
                            faceDetector.detectMultiScale(frame, faceDetections);
                            handDetector.detectMultiScale(frame, handsDetections);

                            double maxArea = 0;
                            maiorRosto = null;
                            for (Rect rect : faceDetections.toArray()) {
                                if (rect.area() > maxArea) {
                                    maxArea = rect.area();
                                    maiorRosto = rect;
                                }
                            }

                            if (maiorRosto != null) {
                                //Imgproc.rectangle(frame, new Point(maiorRosto.x, maiorRosto.y), new Point(maiorRosto.x + maiorRosto.width, maiorRosto.y + maiorRosto.height), new Scalar(0, 255, 0), 3);
                                configuraLogo();
                            }

                            frame.copyTo(frameBackup);

                            addMaos();

                            maxArea = 0;
                            maiorMao = null;
                            for (Rect rect : handsDetections.toArray()) {
                                if (rect.area() > maxArea) {
                                    maxArea = rect.area();
                                    maiorMao = rect;
                                }
                                //Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 3);
                            }

                            if (timerAtivo) {
                                Imgproc.putText(frame, "Tempo: " + interval, new org.opencv.core.Point((frame.width() / 3), (frame.height() / 4)), Core.FONT_HERSHEY_PLAIN, 3, CONTOUR_COLOR, 3);
                            }

                            Imgcodecs.imencode(".bmp", frame, mem);
                            Image im = ImageIO.read(new ByteArrayInputStream(mem.toArray()));
                            buff = (BufferedImage) im;
                            if (g.drawImage(buff, 0, 0, jPanelSnapshot.getWidth(), jPanelSnapshot.getHeight(), 0, 0, buff.getWidth(), buff.getHeight(), null)) {
                                if (runnable == false) {
                                    System.out.println("Paused ..... ");
                                    this.wait();
                                }
                            }

                            if (verificaSim() && !timerAtivo) {
                                timerAtivo = true;
                                interval = TEMPO_FOTO;
                                timer = new Timer();

                                timer.scheduleAtFixedRate(new TimerTask() {

                                    public void run() {
                                        setInterval();
                                    }
                                }, 1000, 1000);
                            }
                            //verificaNao();
                        } catch (Exception ex) {
                            Logger.getLogger(FaceDetection.class.getName()).log(Level.SEVERE, null, ex);
                        }
                    }
                }
            }
        }
    }
    
    public void carregaImagem() {
        ImageIcon natacao = new ImageIcon(getClass().getResource("/resources/background.png"));
        BackgroundPanel panel = new BackgroundPanel(natacao.getImage());
        GradientPaint paint = new GradientPaint(0, 0, Color.BLUE, 600, 0, Color.RED);
        panel.setPaint(paint);
        this.setContentPane(panel);
    }

/////////
    /**
     * Creates new form FaceDetection
     */
    public FaceDetection() {
        carregaImagem();
        initComponents();
        this.setExtendedState(Frame.MAXIMIZED_BOTH);
        try {
            logo = ImageIO.read(new File("camiseta50.jpg"));
            maoNao = ImageIO.read(new File("maoNao.jpg"));
            maoSim = ImageIO.read(new File("maoSim.jpg"));
        } catch (IOException ex) {
            Logger.getLogger(FaceDetection.class.getName()).log(Level.SEVERE, null, ex);
        }
        System.out.println(FaceDetection.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1));
    }

    private final int setInterval() {
        if (interval == 1) {
            timerAtivo = false;
            snapShot = true;
            timer.cancel();
        }
        return --interval;
    }

    private void configuraLogo() throws IOException {
        byte[] data = ((DataBufferByte) logo.getRaster().getDataBuffer()).getData();
        mLogo = new Mat(logo.getHeight(), logo.getWidth(), CvType.CV_8UC3);
        mLogo.put(0, 0, data);

        Double dLargura = mLogo.width() / WIDTHF;
        Double dAltura = mLogo.height() / HEIGHTF;

        double posX = (maiorRosto.x + (maiorRosto.width / 2)) - (dLargura.intValue() / 2);
        double posY = maiorRosto.y;
        Point inicial = new Point(posX, posY);

        ROI = new Rect(inicial, new Size(dLargura.intValue(), dAltura.intValue()));

        mLogoResized = new Mat();
        mLogoResizedCor = new Mat();

        Imgproc.resize(mLogo, mLogoResized, new Size((dLargura.intValue()), (dAltura.intValue())));
        Imgproc.resize(mLogo, mLogoResizedCor, new Size((dLargura.intValue()), (dAltura.intValue())));

        setHsvColor(0, 255, 'H');
        setHsvColor(0, 255, 'S');
        setHsvColor(10, 255, 'V');
        Core.inRange(mLogoResized, mLowerBound, mUpperBound, mMask);
        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
        Imgproc.findContours(mMask, contours, mHierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Find max contour area
        double maxArea = 0;
        Iterator<MatOfPoint> each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint wrapper = each.next();
            double area = Imgproc.contourArea(wrapper);
            if (area > maxArea) {
                maxArea = area;
            }
        }

        mContours.clear();
        mContoursBlacked.clear();
        each = contours.iterator();
        while (each.hasNext()) {
            MatOfPoint contour = each.next();
            if ((Imgproc.contourArea(contour) == maxArea) && (maxArea > 0)) {
                Core.multiply(contour, new Scalar(1, 1), contour);
                mContours.add(contour);
            } else {
                mContoursBlacked.add(contour);
            }
        }

        Imgproc.drawContours(mLogoResized, mContoursBlacked, -1, new Scalar(0), -1);
        Imgproc.drawContours(mLogoResized, mContours, -1, new Scalar(255), -1);
        inicial.y = inicial.y + (maiorRosto.height - 40);
        addLogo(inicial);
    }

    public void addLogo(Point inicial) {

        Double dX = inicial.x;
        Double dY = inicial.y;

        int i, j;
        double[] data = new double[3];

        for (i = 0; i < mLogoResized.width(); i++) {
            for (j = 0; j < mLogoResized.height(); j++) {
                double[] dataMask = mLogoResized.get(i, j);
                if (dataMask != null && dataMask[0] == 255) {

                    double[] dataCor = mLogoResizedCor.get(i, j);

                    int i2 = (i + dY.intValue());
                    int j2 = (j + dX.intValue());
                    frame.put(i2, j2, dataCor);
                }
            }
        }
        //Core.addWeighted(frame.submat(ROI), alpha, mLogoResized, beta, 1, frame.submat(ROI));
    }

    public void addMaos() {

        byte[] data = ((DataBufferByte) maoNao.getRaster().getDataBuffer()).getData();
        mMaoNao = new Mat(maoNao.getHeight(), maoNao.getWidth(), CvType.CV_8UC3);
        mMaoNao.put(0, 0, data);

        Double dLargura = mMaoNao.width() / 4.0;
        Double dAltura = mMaoNao.height() / 4.0;

        alpha = 0.3;
        beta = 1 - alpha;

        ROINao = new Rect(0, 0, dLargura.intValue(), dAltura.intValue());

        Imgproc.resize(mMaoNao, mMaoResized, new Size((dLargura.intValue()), (dAltura.intValue())));
        Core.addWeighted(frame.submat(ROINao), alpha, mMaoResized, beta, 1, frame.submat(ROINao));

        data = ((DataBufferByte) maoSim.getRaster().getDataBuffer()).getData();
        mMaoSim = new Mat(maoSim.getHeight(), maoSim.getWidth(), CvType.CV_8UC3);
        mMaoSim.put(0, 0, data);

        ROISim = new Rect((frame.width() - dLargura.intValue()), 0, dLargura.intValue(), dAltura.intValue());
        Imgproc.resize(mMaoSim, mMaoResizedSim, new Size((dLargura.intValue()), (dAltura.intValue())));
        Core.addWeighted(frame.submat(ROISim), alpha, mMaoResizedSim, beta, 1, frame.submat(ROISim));
    }

    public void gerarQRCode() {
        new Thread(new Runnable() {
            public void run() {
                try {
                    File outputfile = new File("image.jpg");
                    ImageIO.write(buffSnap, "jpg", outputfile);
                    System.out.println("GerandoQRCode");
                    TesteHackathon.enviaImagem();
                    Graphics g2 = jPanelQRCode.getGraphics();
                    qrCode = ImageIO.read(new URL("http://chart.apis.google.com/chart?cht=qr&chl=http://www.ejec.co/virtualfit/index.php?img=image.jpg&chs=120x120"));
                    g2.drawImage(qrCode, 0, 0, jPanelQRCode.getWidth(), jPanelQRCode.getHeight(), 0, 0, qrCode.getWidth(), qrCode.getHeight(), null);
                } catch (Exception ex) {
                    Logger.getLogger(FaceDetection.class.getName()).log(Level.SEVERE, null, ex);
                }
            }
        }).start();
    }

    public boolean verificaSim() {
        for (Rect rect : handsDetections.toArray()) {
            Point centerS = new Point(rect.x + (rect.width / 2), rect.y + ((rect.height / 2)));
            if ((centerS.x >= ROISim.x) && (centerS.y <= (ROISim.y + ROISim.height))) {
                System.out.println("SIM!");
                return true;
            }
        }
        return false;
    }

    public boolean verificaNao() {
        for (Rect rect : handsDetections.toArray()) {
            Point centerN = new Point(rect.x + (rect.width / 2), rect.y + ((rect.height / 2)));
            if ((centerN.x <= (ROINao.x + ROINao.width)) && (centerN.y <= (ROINao.y + ROINao.height))) {
                System.out.println("NAO!");
                return true;
            }
        }
        return false;
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel1 = new javax.swing.JPanel();
        jButton1 = new javax.swing.JButton();
        jButton2 = new javax.swing.JButton();
        jPanelQRCode = new javax.swing.JPanel();
        jButton3 = new javax.swing.JButton();
        jPanelSnapshot = new javax.swing.JPanel();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);

        jPanel1.setBorder(javax.swing.BorderFactory.createTitledBorder(""));

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 674, Short.MAX_VALUE)
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 379, Short.MAX_VALUE)
        );

        jButton1.setText("Start");
        jButton1.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton1ActionPerformed(evt);
            }
        });

        jButton2.setText("Pause");
        jButton2.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton2ActionPerformed(evt);
            }
        });

        jPanelQRCode.setBorder(javax.swing.BorderFactory.createTitledBorder(""));

        javax.swing.GroupLayout jPanelQRCodeLayout = new javax.swing.GroupLayout(jPanelQRCode);
        jPanelQRCode.setLayout(jPanelQRCodeLayout);
        jPanelQRCodeLayout.setHorizontalGroup(
            jPanelQRCodeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 217, Short.MAX_VALUE)
        );
        jPanelQRCodeLayout.setVerticalGroup(
            jPanelQRCodeLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 154, Short.MAX_VALUE)
        );

        jButton3.setText("Snapshot");
        jButton3.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButton3ActionPerformed(evt);
            }
        });

        jPanelSnapshot.setBorder(javax.swing.BorderFactory.createTitledBorder(""));

        javax.swing.GroupLayout jPanelSnapshotLayout = new javax.swing.GroupLayout(jPanelSnapshot);
        jPanelSnapshot.setLayout(jPanelSnapshotLayout);
        jPanelSnapshotLayout.setHorizontalGroup(
            jPanelSnapshotLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 462, Short.MAX_VALUE)
        );
        jPanelSnapshotLayout.setVerticalGroup(
            jPanelSnapshotLayout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 415, Short.MAX_VALUE)
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(0, 0, Short.MAX_VALUE)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jButton1)
                                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                .addComponent(jButton2))
                            .addComponent(jButton3, javax.swing.GroupLayout.PREFERRED_SIZE, 124, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGap(379, 379, 379)
                        .addComponent(jPanelQRCode, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(49, 49, 49)
                        .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                        .addGap(18, 18, 18)
                        .addComponent(jPanelSnapshot, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addGap(50, 50, 50))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(97, 97, 97)
                        .addComponent(jPanelSnapshot, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addGroup(layout.createSequentialGroup()
                        .addGap(119, 119, 119)
                        .addComponent(jPanel1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createSequentialGroup()
                        .addGap(125, 125, 125)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jButton1)
                            .addComponent(jButton2))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addComponent(jButton3))
                    .addComponent(jPanelQRCode, javax.swing.GroupLayout.Alignment.TRAILING, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButton2ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton2ActionPerformed
        myThread.runnable = false;            // stop thread
        jButton2.setEnabled(false);   // activate start button 
        jButton1.setEnabled(true);     // deactivate stop button

        webSource.release();  // stop caturing fron cam
    }//GEN-LAST:event_jButton2ActionPerformed

    private void jButton1ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton1ActionPerformed

        webSource = new VideoCapture(0); // video capture from default cam
        myThread = new DaemonThread(); //create object of threat class
        Thread t = new Thread(myThread);
        t.setDaemon(true);
        myThread.runnable = true;
        t.start();                 //start thrad
        jButton1.setEnabled(false);  // deactivate start button
        jButton2.setEnabled(true);  //  activate stop button


    }//GEN-LAST:event_jButton1ActionPerformed

    private void jButton3ActionPerformed(java.awt.event.ActionEvent evt) {//GEN-FIRST:event_jButton3ActionPerformed
        // TODO add your handling code here:
        snapShot = true;
    }//GEN-LAST:event_jButton3ActionPerformed

    /**
     * @param args the command line arguments
     */
    public static void main(String args[]) {
        //System.load("C:\\OpenCV\\opencv\\build\\java\\x64\\opencv_java300.dll");
        String directory;
        File dllFile;
        CodeSource codeSource = FaceDetection.class.getProtectionDomain().getCodeSource();
        File jarFile;
        try {
            jarFile = new File(codeSource.getLocation().toURI().getPath());
            File parentDir = jarFile.getParentFile();
            //if (System.getProperty("sun.arch.data.model").equalsIgnoreCase("64")) {
            dllFile = new File(parentDir, "opencv_java300.dll");
            /*} else {
                dllFile = new File(parentDir, "opencv_java300_32.dll");
            }*/
            System.out.println(dllFile.getPath());
            System.load(dllFile.getPath());
        } catch (URISyntaxException ex) {
            Logger.getLogger(FaceDetection.class.getName()).log(Level.SEVERE, null, ex);
        }

        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
         * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Nimbus".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;
                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(FaceDetection.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(FaceDetection.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(FaceDetection.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(FaceDetection.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                new FaceDetection().setVisible(true);
            }
        });
    }

    public void setHsvColor(int min, int max, char channel) {

        switch (channel) {
            case 'H':
                mLowerBound.val[0] = min;
                mUpperBound.val[0] = max;
                break;

            case 'S':
                mLowerBound.val[1] = min;
                mUpperBound.val[1] = max;
                break;

            case 'V':
                mLowerBound.val[2] = min;
                mUpperBound.val[2] = max;
                break;

            case 'A':
                mLowerBound.val[0] = min;
                mUpperBound.val[0] = max;
                mLowerBound.val[1] = min;
                mUpperBound.val[1] = max;
                mLowerBound.val[2] = min;
                mUpperBound.val[2] = max;
                break;
        }

        mLowerBound.val[3] = 0;
        mUpperBound.val[3] = 255;
    }
    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButton1;
    private javax.swing.JButton jButton2;
    private javax.swing.JButton jButton3;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanelQRCode;
    private javax.swing.JPanel jPanelSnapshot;
    // End of variables declaration//GEN-END:variables
}
