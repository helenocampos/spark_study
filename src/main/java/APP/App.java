/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package APP;

import Kafka.ConsumerCreator;
import commons.TemperatureMeasurement;
import model.*;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.Locale;
import java.util.Scanner;
import org.apache.commons.lang.SystemUtils;
import org.apache.kafka.clients.consumer.Consumer;
import org.apache.kafka.clients.consumer.ConsumerRecords;
import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.classification.LogisticRegressionModel;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

/**
 *
 * @author helenocampos
 */
public class App {

    static LogisticRegression lr;
    static LogisticRegressionModel lrModel;
    public static Integer MAX_NO_MESSAGE_FOUND_COUNT = 10000;
    public static String HADOOP_WINUTILS_DIRECTORY = "C:\\hadoop";

    public static void main(String[] args) {
        if (SystemUtils.IS_OS_WINDOWS) {
            System.setProperty("hadoop.home.dir", HADOOP_WINUTILS_DIRECTORY);
        }
        Logger.getRootLogger().setLevel(Level.ERROR);
        SparkSession spark = SparkSession
                .builder().master("local")
                .appName("JavaRegressionExample")
                .getOrCreate();
        spark.sparkContext().setLogLevel("ERROR");
        String option = "";
        Scanner sc = new Scanner(System.in);
        sc.useLocale(Locale.US);
        String modelName = "";
        while (!"0".equalsIgnoreCase(option)) {
            System.out.println("Escolha uma opção: ");
            System.out.println("1 - Treinar modelo");
            System.out.println("2 - Carregar modelo salvo");
            System.out.println("3 - Salvar modelo atual");
            System.out.println("4 - Realizar nova inferência");
            System.out.println("5 - Inicializar inferência a partir de stream");
            System.out.println("0 - Sair");
            System.out.println("");
            option = sc.next();
            switch (option) {
                case "0":
                    System.out.println("Adeus");
                    break;
                case "1":
                    System.out.println("Qual o caminho dos dados de treinamento? (arquivo csv)");
                    String trainingDataPath = sc.next();
                    System.out.println("Deseja treinar o modelo com parâmetros padrão? S/N");
                    String padrao = sc.next();
                    if (padrao.equalsIgnoreCase("S")) {
                        trainModel(spark, 10, 0.001, trainingDataPath);
                    } else {
                        System.out.println("Qual o valor do parâmetro maxIter desejado?");
                        int maxIter = sc.nextInt();
                        System.out.println("Qual o valor do parâmetro regParam desejado?");
                        double regParam = sc.nextDouble();
                        trainModel(spark, maxIter, regParam, trainingDataPath);
                    }
                    System.out.println("Modelo treinado!");
                    break;
                case "2":
                    System.out.println("Qual o nome do modelo a ser carregado?");
                    modelName = sc.next();
                    loadModel(modelName);
                    break;
                case "3":
                    System.out.println("Qual nome deseja dar ao modelo?");
                    modelName = sc.next();
                    exportModel(modelName);
                    break;
                case "4":
                    if (lrModel != null) {
                        Dataset<Row> dataToEvaluate = getEntryFromConsole(spark, sc);
                        evaluateData(dataToEvaluate);
                    } else {
                        System.out.println("Primeiro é necessário instanciar um modelo!");
                    }
                    break;
                case "5":
                    System.out.println("Qual o endereço ip da stream?");
                    String address = sc.next();
                    listenToStream(spark, address);
                    break;

                default:
                    System.out.println("Opção inválida!");
                    break;
            }
        }

        spark.stop();
    }

    private static Dataset<Row> defineFeaturesColumns(Dataset<Row> data) {
        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(new String[]{
            "hora", "temperatura"
        })
                .setOutputCol("features");
        return assembler.transform(data);
    }

    private static void exportModel(String modelName) {
        try {
            modelName = "models/" + modelName;
            if (lrModel != null && lr != null) {
                lrModel.write().save(modelName + "/model");
                lr.write().save(modelName + "/lr");
                System.out.println("Modelo salvo com sucesso!");
            } else {
                System.out.println("Primeiro é necessário instanciar um modelo!s");
            }
        } catch (IOException ex) {
            java.util.logging.Logger.getLogger(App.class.getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
    }

    private static void printData(Dataset<Row> data) {
        for (Row r : data.select("hora", "temperatura", "prediction").collectAsList()) {
            boolean anomalia = r.get(2).equals(1.0);
            if (anomalia) {
                System.out.println("(hora: " + r.get(0) + ", temp:" + r.get(1) + ") -->  anomalia");
            } else {
                System.out.println("(hora: " + r.get(0) + ", temp:" + r.get(1) + ") -->  normal");
            }
        }
    }

    private static Dataset<Row> getEntryFromConsole(SparkSession spark, Scanner sc) {
        System.out.println("Digite a hora no formato float. Exemplo: 18.512");
        float hora = sc.nextFloat();
        System.out.println("Digite a temperatura (inteiro). Exemplo: 20");
        int temperatura = sc.nextInt();
        return spark.createDataFrame(Arrays.asList(new MeasurementModel(hora, temperatura)), MeasurementModel.class);
    }

    private static Dataset<Row> getEntry(SparkSession spark, float hora, int temp) {
        return spark.createDataFrame(Arrays.asList(new MeasurementModel(hora, temp)), MeasurementModel.class);
    }

    private static void evaluateData(Dataset<Row> data) {
        if (lrModel != null) {
            Dataset<Row> dataToEvaluate = data;
            dataToEvaluate = defineFeaturesColumns(dataToEvaluate);
            dataToEvaluate = lrModel.transform(dataToEvaluate);
            printData(dataToEvaluate);
        } else {
            System.out.println("Não há nenhum modelo carregado.");
        }
    }

    private static void trainModel(SparkSession spark, int maxIter, double regParam, String dataPath) {
        StructType customSchema = new StructType(new StructField[]{
            new StructField("hora", DataTypes.FloatType, true, Metadata.empty()),
            new StructField("temperatura", DataTypes.IntegerType, true, Metadata.empty()),
            new StructField("saida", DataTypes.IntegerType, true, Metadata.empty())
        });
        Dataset<Row> training = spark.read().format("csv").option("header", "true").schema(customSchema)
                .load(dataPath);
        training = defineFeaturesColumns(training);
        lr = new LogisticRegression()
                .setMaxIter(maxIter)
                .setRegParam(regParam)
                .setFeaturesCol("features")
                .setLabelCol("saida");

        lrModel = lr.fit(training);
    }

    private static void loadModel(String modelName) {
        File modelPath = new File("models/" + modelName);
        if (modelPath.exists()) {
            lr = LogisticRegression.load("models/" + modelName + "/lr");
            lrModel = LogisticRegressionModel.load("models/" + modelName + "/model");
            System.out.println("Modelo carregado!");
        } else {
            System.out.println("A model with this name does not exist.");
        }
    }

    private static void listenToStream(SparkSession spark, String address) {
        if (lrModel != null) {
            String clientId = Long.toString(System.currentTimeMillis());
            try (Consumer<String, TemperatureMeasurement> consumer = ConsumerCreator.createConsumer(address,clientId)) {
                System.out.println("Escutando stream em "+address+" client id:" + clientId);
                int noMessageFound = 0;
                while (true) {
                    ConsumerRecords<String, TemperatureMeasurement> consumerRecords = consumer.poll(java.time.Duration.ofMillis(510));

                    if (consumerRecords.count() == 0) {
                        noMessageFound++;
                        if (noMessageFound > MAX_NO_MESSAGE_FOUND_COUNT) {
                            break;
                        } else {
                            continue;
                        }
                    }
                    consumerRecords.forEach(record
                            -> {
                        System.out.println("Sensor: " + record.value().getSensorId());
                        float start = System.currentTimeMillis();
                        evaluateData(getEntry(spark, record.value().getDateFloat(), (int) record.value().getTemp()));
                        float finish = System.currentTimeMillis();
                        double evaluationTime = (finish - start);
                        System.out.println("Evaluated in: " + evaluationTime + " milliseconds.");
                        System.out.println("\n\n");
                    });
                    consumer.commitAsync();
                }
            }
        } else {
            System.out.println("Não há nenhum modelo treinado!");
        }
    }
}
