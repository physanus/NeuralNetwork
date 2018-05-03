package de.danielprinz.NeuralNetwork;

import org.apache.log4j.BasicConfigurator;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;

public class Main {

    public static void main(String[] args) throws IOException, InterruptedException {

        BasicConfigurator.configure();

        int seed = 123;
        double learningRate = 0.01;
        int batchSize = 50;
        int nEpochs = 30;

        int numInputs = 2;
        int numOutputs = 2;
        int numHiddenNodes = 20;

        File data_train = new File(new ClassPathResource("linear_data_train.csv").getFile().getPath());
        File data_test = new File(new ClassPathResource("linear_data_eval.csv").getFile().getPath());

        // load the training data
        RecordReader recordReader = new CSVRecordReader();
        recordReader.initialize(new FileSplit(data_train));
        DataSetIterator trainIter = new RecordReaderDataSetIterator(recordReader, batchSize, 0, 2);

        // load the test-evaluation data:
        RecordReader recordReaderTest = new CSVRecordReader();
        recordReaderTest.initialize(new FileSplit(data_test));
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest, batchSize, 0, 2);

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1) // custom
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // custom
                .updater(new Nesterovs(learningRate, 0.9))
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(numInputs)
                        .nOut(numHiddenNodes)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .weightInit(WeightInit.XAVIER)
                        .activation(Activation.SOFTMAX)
                        .nIn(numHiddenNodes)
                        .nOut(numOutputs)
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();

        //System.out.println(conf.toJson());

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(10)); // print score every 10 parameter updates

        for(int n = 0; n < nEpochs; n++) {
            model.fit(trainIter);
        }

        System.out.println("Evaluate model.......");
        Evaluation eval = new Evaluation(numOutputs);
        while(testIter.hasNext()) {
            DataSet dataSet = testIter.next();
            INDArray features = dataSet.getFeatureMatrix();
            INDArray labels = dataSet.getLabels();
            INDArray predicted = model.output(features, false);

            eval.eval(labels, predicted);
        }

        System.out.println(eval.stats());

        //------------------------------------------------------------------------------------
        //Training is complete.

    }

}
